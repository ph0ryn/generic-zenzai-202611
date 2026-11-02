"""Read the AJIMEE checkout and score predictions without executing upstream code."""

import hashlib
import json
import subprocess
from pathlib import Path

BENCHMARK_REPOSITORY = "https://github.com/azooKey/AJIMEE-Bench"
DATA_PATH = Path("JWTD_v2/v1/evaluation_items.json")


def parse_items(raw):
    items = json.loads(raw)
    if not isinstance(items, list) or not items:
        raise ValueError("評価データには空でないJSON配列が必要です。")
    seen = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"評価データ{position}件目にはJSONオブジェクトが必要です。")
        for name in ("index", "input"):
            if not isinstance(item.get(name), str) or not item[name]:
                raise ValueError(
                    f"評価データ{position}件目の{name}は空でない文字列が必要です。"
                )
        if not isinstance(item.get("context_text"), str):
            raise TypeError(
                f"評価データ{position}件目のcontext_textは文字列が必要です。"
            )
        references = item.get("expected_output")
        if (
            not isinstance(references, list)
            or not references
            or not all(
                isinstance(reference, str) and reference for reference in references
            )
        ):
            raise ValueError(
                f"評価データ{position}件目には空でない許容解の配列が必要です。"
            )
        if item["index"] in seen:
            raise ValueError(f"評価データのindexが重複しています: {item['index']}")
        seen.add(item["index"])
    return items


def load_benchmark(root):
    root = root.resolve()
    path = root / DATA_PATH
    if not path.is_file():
        raise ValueError(
            f"評価データがありません: {path}。git submodule update --init を実行してください。"
        )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True
        ).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        if top != root:
            raise ValueError("benchmarkにはAJIMEE-BenchのGitルートを指定してください。")
        revision = git("rev-parse", "HEAD").decode().strip()
        committed = git("show", f"HEAD:{DATA_PATH.as_posix()}")
    except subprocess.CalledProcessError as error:
        raise ValueError("AJIMEE-Benchの初期化済みGit checkoutが必要です。") from error
    raw = path.read_bytes()
    if raw != committed:
        raise ValueError(
            "評価データがcommitから変更されています。変更を確認してください。"
        )
    return parse_items(raw), {
        "repository": BENCHMARK_REPOSITORY,
        "revision": revision,
        "data_path": DATA_PATH.as_posix(),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC-BY-SA-3.0",
    }


def edit_distance(reference, hypothesis):
    """Unit-cost Levenshtein distance over Unicode code points."""
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, start=1):
        current = [i]
        for j, target in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (source != target),
                )
            )
        previous = current
    return previous[-1]


def score_prediction(references, prediction):
    # Matches the official utils.py: choose the minimum *normalized* CER.
    return {
        "accuracy_at1": int(prediction in references),
        "min_cer": min(edit_distance(ref, prediction) / len(ref) for ref in references),
    }


def summarize(predictions):
    def aggregate(rows):
        count = len(rows)
        correct = sum(row["accuracy_at1"] for row in rows)
        return {
            "count": count,
            "correct": correct,
            "accuracy_at1": correct / count if count else None,
            "mean_min_cer": sum(row["min_cer"] for row in rows) / count
            if count
            else None,
            "truncated": sum(row["truncated"] for row in rows),
        }

    return {
        "all": aggregate(predictions),
        "with_context": aggregate([row for row in predictions if row["context_text"]]),
        "without_context": aggregate(
            [row for row in predictions if not row["context_text"]]
        ),
    }
