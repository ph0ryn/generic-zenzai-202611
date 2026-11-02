"""Validate JSONL examples and encode completion-only training data."""

import json

from datasets import Dataset

from .prompting import MAX_TOKENS, make_prompt


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
