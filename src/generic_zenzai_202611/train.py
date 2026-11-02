"""CLI orchestration for full-parameter kana-kanji conversion training."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_callback import PrinterCallback
from trl import SFTConfig, SFTTrainer

from .checkpoints import remove_checkpoints, resolve_run, valid_checkpoint
from .config import DATASET_DIR, MODEL_DIR, MODEL_ID, MODEL_REVISION
from .generate_data import generate_train_jsonl
from .prepare import prepared_model_path
from .prompting import CONTEXT_CHARS, MAX_TOKENS, make_prompt
from .training_data import encode_example, prepare_dataset

# Preserve helper imports for existing callers of the training module.
__all__ = [
    "CONTEXT_CHARS",
    "MAX_TOKENS",
    "MODEL_ID",
    "MODEL_REVISION",
    "encode_example",
    "main",
    "make_prompt",
    "mixed_precision",
    "prepare_dataset",
    "resolve_run",
    "valid_checkpoint",
]


class TrainingLogCallback(TrainerCallback):
    """Use line-based metrics instead of Trainer's raw dictionary output."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        metrics = " | ".join(
            f"{name}: {value:.6g}" if isinstance(value, float) else f"{name}: {value}"
            for name, value in logs.items()
            if name != "total_flos"
        )
        print(f"step {state.global_step}/{state.max_steps} | {metrics}", flush=True)


def mixed_precision(cpu):
    if cpu:
        return "fp32"
    # Emulated bf16 support does not imply support for the CUDA GEMM used by TRL.
    return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"


def main():
    parser = argparse.ArgumentParser(
        description="文脈付きかな漢字変換を全パラメータ学習する"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="事前に保存した既定のベースモデルのディレクトリ",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Driveに保存した元データのディレクトリ",
    )
    parser.add_argument("--wikipedia-count", type=int, required=True)
    parser.add_argument("--corpus-count", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/llm-jp-150m"),
        help="日時付きrunを保存する親ディレクトリ",
    )
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--save-steps", type=int, default=1000)
    resuming = parser.add_mutually_exclusive_group()
    resuming.add_argument("--resume", action="store_true", help="最新runから再開")
    resuming.add_argument(
        "--resume-from", type=Path, help="再開するcheckpointディレクトリ"
    )
    parser.add_argument("--cpu", action="store_true", help="小規模な動作確認用")
    args = parser.parse_args()
    if args.epochs <= 0 or args.save_steps <= 0:
        parser.error("epochsとsave-stepsは正の値を指定してください。")
    if min(args.wikipedia_count, args.corpus_count) < 0 or not (
        args.wikipedia_count + args.corpus_count
    ):
        parser.error("各件数は0以上、合計は正の整数にしてください。")
    if not args.cpu and not torch.cuda.is_available():
        parser.error("CUDA GPUが必要です。ColabのランタイムをGPUに変更してください。")
    try:
        args.output, args.resume_from = resolve_run(
            args.output, args.resume, args.resume_from
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    model_path = Path("model-path.txt")
    model_path.write_text("", encoding="utf-8")
    print(f"出力先: {args.output}", flush=True)
    if args.resume_from:
        print(f"再開元: {args.resume_from}", flush=True)

    data_path = args.output / "train.jsonl"
    try:
        generate_train_jsonl(
            args.dataset_dir,
            data_path,
            args.wikipedia_count,
            args.corpus_count,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"学習データを生成できません: {error}")
    print(f"生成した学習データ: {data_path}", flush=True)

    set_seed(42)
    revision = MODEL_REVISION if args.model == MODEL_ID else None
    try:
        model_source = prepared_model_path(args.model_dir) if revision else args.model
    except (OSError, ValueError) as error:
        parser.error(str(error))
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=revision, local_files_only=revision is not None
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(data_path, tokenizer)
    settings = {
        "model": args.model,
        "revision": revision,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "examples": len(dataset),
        "max_tokens": MAX_TOKENS,
        "context_chars": CONTEXT_CHARS,
        "learning_rate": 5e-5,
        "batch_size": 32,
        "gradient_accumulation": 1,
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
        model_source,
        revision=revision,
        local_files_only=revision is not None,
        dtype=torch.float32,
        attn_implementation="sdpa",
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
            logging_first_step=True,
            logging_nan_inf_filter=False,
            disable_tqdm=True,
            save_steps=args.save_steps,
            save_total_limit=1,
            eval_strategy="no",
            report_to="none",
            seed=settings["seed"],
            data_seed=settings["seed"],
            dataloader_pin_memory=not args.cpu,
        ),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(TrainingLogCallback)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    effective_batch_size = (
        trainer.args.per_device_train_batch_size
        * trainer.args.gradient_accumulation_steps
        * trainer.args.world_size
    )
    print(
        f"学習データ: {len(dataset)}件 | 実効バッチ: {effective_batch_size} | 周回数: {args.epochs}",
        flush=True,
    )
    print(f"計算精度: {precision}", flush=True)
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None
    )
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(args.output / "model")
    result.metrics.update(
        global_step=trainer.state.global_step,
        precision=precision,
        effective_batch_size=effective_batch_size,
    )
    trainer.save_metrics("train", result.metrics)
    model_path.write_text(
        str((args.output / "model").resolve()) + "\n", encoding="utf-8"
    )
    # Keep resumable state until every final artifact has been saved successfully.
    remove_checkpoints(args.output)
    print(f"学習済みモデル: {args.output / 'model'}", flush=True)
