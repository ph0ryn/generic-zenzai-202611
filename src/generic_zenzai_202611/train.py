import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from .prompting import CONTEXT_CHARS, MAX_TOKENS, make_prompt

MODEL_ID = "llm-jp/llm-jp-3-150m"
MODEL_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"


def mixed_precision(cpu):
    if cpu:
        return "fp32"
    # Emulated bf16 support does not imply support for the CUDA GEMM used by TRL.
    return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"


def encode_example(row, tokenizer):
    if not isinstance(row, dict):
        raise TypeError("各行にはJSONオブジェクトが必要です。")
    for name in ("input", "output"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"{name} は空でない文字列が必要です。")
    if row["left_context"] is not None and not isinstance(row["left_context"], str):
        raise ValueError("left_context は文字列またはnullが必要です。")
    if tokenizer.eos_token_id is None:
        raise ValueError("終了記号のあるtokenizerが必要です。")

    prompt_ids = tokenizer.encode(make_prompt(row["input"], row["left_context"]))
    answer_ids = tokenizer.encode(row["output"], add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > MAX_TOKENS:
        raise ValueError(
            f"{len(input_ids)}トークンの例が上限{MAX_TOKENS}を超えました。"
        )
    return {
        "input_ids": input_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * len(answer_ids),
    }


def prepare_dataset(path, tokenizer):
    examples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                examples.append(encode_example(json.loads(line), tokenizer))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError("学習データが空です。")
    return Dataset.from_list(examples)


def resolve_run(output, resume=False, resume_from=None):
    if resume_from is not None:
        run = resume_from.parent
        if output.resolve() not in (run.resolve(), run.parent.resolve()):
            raise ValueError("resume-fromにはoutput内のcheckpointを指定してください。")
        if not valid_checkpoint(resume_from):
            raise ValueError("有効な再開用チェックポイントが見つかりません。")
        return run, resume_from
    if resume:
        runs = sorted(path for path in output.glob("run-*") if path.is_dir())
        if not runs:
            raise ValueError("再開するrunが見つかりません。")
        run = runs[-1]
        if (run / "train_results.json").exists():
            raise ValueError(f"最新のrunは完了済みです: {run}")
        checkpoints = [
            path
            for path in run.glob("checkpoint-*")
            if path.name.removeprefix("checkpoint-").isdigit()
            and valid_checkpoint(path)
        ]
        if not checkpoints:
            raise ValueError(f"最新のrunに有効なチェックポイントがありません: {run}")
        return run, max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    run = output / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run.mkdir(parents=True, exist_ok=False)
    return run, None


def valid_checkpoint(path):
    required = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "config.json")
    if not all((path / name).is_file() for name in required):
        return False
    weights = ("model.safetensors", "pytorch_model.bin")
    indexes = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    try:
        json.loads((path / "trainer_state.json").read_text())
        if any((path / name).is_file() for name in weights):
            return True
        for name in indexes:
            if (path / name).is_file():
                shards = json.loads((path / name).read_text())["weight_map"].values()
                return bool(shards) and all(
                    (path / shard).is_file() for shard in shards
                )
    except OSError, ValueError, KeyError, TypeError:
        return False
    return False


def main():
    parser = argparse.ArgumentParser(
        description="文脈付きかな漢字変換を全パラメータ学習する"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/llm-jp-150m"),
        help="日時付きrunを保存する親ディレクトリ",
    )
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    resuming = parser.add_mutually_exclusive_group()
    resuming.add_argument("--resume", action="store_true", help="最新runから再開")
    resuming.add_argument(
        "--resume-from", type=Path, help="再開するcheckpointディレクトリ"
    )
    parser.add_argument("--cpu", action="store_true", help="小規模な動作確認用")
    args = parser.parse_args()
    if args.epochs <= 0 or args.save_steps <= 0:
        parser.error("epochsとsave-stepsは正の値を指定してください。")
    if not args.cpu and not torch.cuda.is_available():
        parser.error("CUDA GPUが必要です。ColabのランタイムをGPUに変更してください。")
    try:
        args.output, args.resume_from = resolve_run(
            args.output, args.resume, args.resume_from
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"出力先: {args.output}", flush=True)
    if args.resume_from:
        print(f"再開元: {args.resume_from}", flush=True)

    set_seed(42)
    revision = MODEL_REVISION if args.model == MODEL_ID else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(args.data, tokenizer)
    settings = {
        "model": args.model,
        "revision": revision,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "examples": len(dataset),
        "max_tokens": MAX_TOKENS,
        "context_chars": CONTEXT_CHARS,
        "learning_rate": 5e-5,
        "batch_size": 2,
        "gradient_accumulation": 16,
        "seed": 42,
    }
    metadata_path = args.output / "run.json"
    if args.resume_from:
        try:
            original_settings = json.loads(metadata_path.read_text())
        except (OSError, ValueError) as error:
            parser.error(f"再開用のrun.jsonを読み込めません: {error}")
        if original_settings != settings:
            parser.error(
                "元の実行とモデル・データ・学習設定が異なるため再開できません。"
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=torch.float32, attn_implementation="sdpa"
    )
    precision = mixed_precision(args.cpu)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            learning_rate=settings["learning_rate"],
            per_device_train_batch_size=settings["batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation"],
            max_length=MAX_TOKENS,
            completion_only_loss=True,
            packing=False,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            use_cpu=args.cpu,
            gradient_checkpointing=True,
            optim="adamw_torch",
            warmup_steps=0.05,
            logging_steps=10,
            logging_nan_inf_filter=False,
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="no",
            report_to="none",
            seed=settings["seed"],
            data_seed=settings["seed"],
            dataloader_pin_memory=not args.cpu,
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    print(f"学習データ: {len(dataset)}件、実効バッチ: 32、周回数: {args.epochs}")
    print(f"計算精度: {precision}")
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None
    )
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(args.output / "model")
    trainer.save_metrics("train", result.metrics)
    print(f"学習済みモデル: {args.output / 'model'}")
