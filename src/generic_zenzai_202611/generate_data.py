"""Build a run-local training JSONL from the two mounted source files."""

import os
import tempfile

SOURCES = ("train_wikipedia.jsonl", "train_llm-jp-corpus-v3.jsonl")


def generate_train_jsonl(dataset_dir, output, wikipedia_count, corpus_count):
    """Copy the requested number of first lines from each source, in source order."""
    counts = (wikipedia_count, corpus_count)
    if any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError("各件数は0以上、合計は正の整数にしてください。")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
            temporary = target.name
            for filename, count in zip(SOURCES, counts, strict=True):
                if count == 0:
                    continue
                path = dataset_dir / filename
                with path.open("rb") as source:
                    for index in range(count):
                        line = source.readline()
                        if not line:
                            raise ValueError(
                                f"{path}: {count}件を要求しましたが、{index}件で不足しました。"
                            )
                        target.write(line if line.endswith(b"\n") else line + b"\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
