"""CLI for AJIMEE model-only evaluation and persisted results."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import transformers

from .benchmark import load_benchmark, score_prediction, summarize
from .inference import Converter
from .prompting import CONTEXT_CHARS, PROMPT_TEMPLATE


def main():
    parser = argparse.ArgumentParser(
        description="保存済みモデルをAJIMEE-Benchで評価する"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="保存済みmodelディレクトリ"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/AJIMEE-Bench"),
        help="AJIMEE-BenchのGit checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="評価結果の親ディレクトリ（既定: modelの隣のajimee）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cpu", action="store_true", help="GPUの代わりにCPUで評価")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokensは正の整数を指定してください。")
    try:
        items, benchmark = load_benchmark(args.benchmark)
        converter = Converter(
            args.model, cpu=args.cpu, max_new_tokens=args.max_new_tokens
        )
        inputs = converter.prepare_inputs(items)
    except (TypeError, ValueError, OSError) as error:
        parser.error(str(error))

    now = datetime.now().astimezone()
    parent = args.output if args.output is not None else args.model.parent / "ajimee"
    output = parent / f"eval-{now:%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": now.isoformat(),
        "model": str(args.model.resolve()),
        "benchmark": benchmark,
        "generation": converter.generation_config.to_dict(),
        "context_chars": CONTEXT_CHARS,
        "max_tokens": converter.limit,
        "prompt_template": PROMPT_TEMPLATE,
        "device": str(converter.device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    (output / "evaluation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"評価結果: {output}", flush=True)
    predictions = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for position, (item, encoded) in enumerate(
            zip(items, inputs, strict=True), start=1
        ):
            prediction = converter.predict(encoded)
            row = {
                "index": item["index"],
                "input": item["input"],
                "context_text": item["context_text"],
                "expected_output": item["expected_output"],
                **prediction,
                **score_prediction(item["expected_output"], prediction["prediction"]),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            predictions.append(row)
            verdict = "完全一致" if row["accuracy_at1"] else "不一致"
            print(
                f"{position}/{len(items)} | 判定: {verdict} | MinCER: {row['min_cer']:.4f}",
                flush=True,
            )
    scores = summarize(predictions)
    # Written only after every item succeeds; its presence marks a completed evaluation.
    (output / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, label in (
        ("all", "全体"),
        ("with_context", "文脈あり"),
        ("without_context", "文脈なし"),
    ):
        score = scores[name]
        if score["count"]:
            print(
                f"{label}: {score['count']}件 | 完全一致率: {score['accuracy_at1']:.2%} "
                f"({score['correct']}/{score['count']}) | "
                f"平均MinCER: {score['mean_min_cer']:.4f} | 生成上限到達: {score['truncated']}件",
                flush=True,
            )
        else:
            print(f"{label}: 0件（対象なし）", flush=True)


if __name__ == "__main__":
    main()
