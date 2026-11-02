"""Build a run-local training JSONL from the two mounted source files."""

import json
import os
import tempfile

from .config import DATASET_SOURCES


def generate_train_jsonl(dataset_dir, output, wikipedia_count, corpus_count):
    """Take the first requested valid examples from each source, in source order."""
    counts = (wikipedia_count, corpus_count)
    if any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError("各件数は0以上、合計は正の整数にしてください。")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
            temporary = target.name
            for filename, count in zip(DATASET_SOURCES, counts, strict=True):
                if count == 0:
                    continue
                path = dataset_dir / filename
                with path.open("rb") as source:
                    accepted = 0
                    skipped = 0
                    for line_number, line in enumerate(source, start=1):
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError) as error:
                            raise ValueError(
                                f"{path}:{line_number}: {error}"
                            ) from error
                        if not isinstance(row, dict):
                            raise TypeError(
                                f"{path}:{line_number}: JSONオブジェクトが必要です。"
                            )
                        if any(
                            not isinstance(row.get(name), str) or not row[name]
                            for name in ("input", "output")
                        ):
                            skipped += 1
                            continue
                        target.write(line if line.endswith(b"\n") else line + b"\n")
                        accepted += 1
                        if accepted == count:
                            break
                    if accepted < count:
                        raise ValueError(
                            f"{path}: {count}件を要求しましたが、有効な例が{accepted}件で不足しました。"
                        )
                    if skipped:
                        print(
                            f"{path}: input/outputが空または文字列でない{skipped}行を除外しました。",
                            flush=True,
                        )
        os.replace(temporary, output)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
