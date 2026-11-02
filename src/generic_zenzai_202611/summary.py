"""Print and save a self-contained summary from completed evaluation records."""

import argparse
import json
from pathlib import Path

from .benchmark import summarize


def read_object(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: JSONオブジェクトが必要です。")
    return data


def read_training(model):
    settings_path = model.parent / "run.json"
    if not settings_path.exists():
        return None
    settings = read_object(settings_path)
    required = (
        "model",
        "examples",
        "learning_rate",
        "batch_size",
        "gradient_accumulation",
        "max_tokens",
        "context_chars",
        "seed",
    )
    for key in required:
        if key not in settings:
            raise ValueError(f"{settings_path}: {key}がありません。")
    state_path = model / "trainer_state.json"
    metrics_path = model.parent / "train_results.json"
    if state_path.exists():
        # A checkpoint's progress must not be taken from the completed run's metrics.
        state = read_object(state_path)
        metrics = {key: state[key] for key in ("epoch", "global_step")}
    else:
        metrics = read_object(metrics_path) if metrics_path.exists() else {}
    return {"settings": settings, "metrics": metrics}


def render_summary(output, metadata, scores, predictions):
    lines = ["# 実験サマリー", "", f"評価対象: {metadata['model']}", ""]
    training = metadata["training"]
    if training is None:
        lines.append("学習記録: なし（学習条件は不明）")
    else:
        settings, metrics = training["settings"], training["metrics"]
        lines += [
            "## 学習",
            "",
            f"- 元モデル: {settings['model']}",
            f"- 学習データ: {settings['examples']:,}件 | 学習率: {settings['learning_rate']:g} | seed: {settings['seed']}",
            f"- バッチ: {settings['batch_size']}件 | 勾配累積: {settings['gradient_accumulation']}回",
            f"- 文脈長: {settings['context_chars']}文字 | 入力＋正解の上限: {settings['max_tokens']}トークン",
        ]
        for label, key, spec, unit in (
            ("周回数", "epoch", "g", ""),
            ("更新回数", "global_step", "d", ""),
            ("実効バッチ", "effective_batch_size", "d", "件"),
            ("計算精度", "precision", "s", ""),
            ("今回の学習実行時間", "train_runtime", ".1f", "秒"),
            ("学習loss", "train_loss", ".4f", ""),
            ("最終ログ区間のトークン正解率", "mean_token_accuracy", ".2%", ""),
        ):
            value = f"{metrics[key]:{spec}}{unit}" if key in metrics else "未記録"
            lines.append(f"- {label}: {value}")

    generation = metadata["generation"]
    sampling = "あり" if generation["do_sample"] else "なし"
    lines += [
        "",
        "## 評価条件",
        "",
        f"- AJIMEE: {scores['all']['count']}件 | 文脈長: {metadata['context_chars']}文字",
        f"- サンプリング: {sampling} | beam数: {generation['num_beams']} | 生成上限: {generation['max_new_tokens']}トークン",
        f"- 入力＋生成の上限: {metadata['max_tokens']}トークン | 精度: {metadata['dtype']} | device: {metadata['device']}",
        f"- PyTorch: {metadata['torch_version']} | Transformers: {metadata['transformers_version']}",
        "- 採点: 許容解との完全一致とMinCER。空白除去・表記の正規化なし。",
        "",
        "## 評価結果",
        "",
    ]
    for name, label in (
        ("all", "全体"),
        ("with_context", "文脈あり"),
        ("without_context", "文脈なし"),
    ):
        score = scores[name]
        if score["count"]:
            lines.append(
                f"- {label}: {score['count']}件 | 完全一致率: {score['accuracy_at1']:.2%} "
                f"({score['correct']}/{score['count']}) | "
                f"平均MinCER: {score['mean_min_cer']:.4f} | 生成上限到達: {score['truncated']}件"
            )
        else:
            lines.append(f"- {label}: 0件（対象なし）")
    empty = sum(not row["prediction"].strip() for row in predictions)
    average_tokens = sum(row["generated_tokens"] for row in predictions) / len(
        predictions
    )
    lines += [
        f"- 空出力: {empty}件（空白のみを含む） | 平均生成トークン数: {average_tokens:.2f}（終了記号を含む）",
        "",
        "学習中のトークン正解率と、評価時の完全一致率は異なる指標。",
        "文脈あり・なしは別の問題群であり、この差だけでは文脈の効果を判断できない。",
        f"詳細結果: {output}",
    ]
    return "\n".join(lines) + "\n"


def write_summary(output):
    metadata = read_object(output / "evaluation.json")
    scores = read_object(output / "scores.json")
    with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream]
    if not predictions or summarize(predictions) != scores:
        raise ValueError("予測の集計とscores.jsonが一致しません。")
    if "training" not in metadata:
        metadata["training"] = read_training(Path(metadata["model"]))
    summary = render_summary(output, metadata, scores, predictions)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="保存済み評価から実験サマリーを出力する"
    )
    parser.add_argument(
        "evaluation", type=Path, help="scores.jsonなどがあるevalディレクトリ"
    )
    args = parser.parse_args()
    try:
        summary = write_summary(args.evaluation)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
