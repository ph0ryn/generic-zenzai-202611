"""Check the training JSONL built from the mounted dataset files."""

import json
import tempfile
import unittest
from pathlib import Path

from generic_zenzai_202611.generate_data import generate_train_jsonl


class GenerateDataTest(unittest.TestCase):
    def test_takes_exact_requested_prefix_from_each_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "dataset"
            sources.mkdir()
            for filename, prefix in (
                ("train_wikipedia.jsonl", "wiki"),
                ("train_llm-jp-corpus-v3.jsonl", "corpus"),
            ):
                with (sources / filename).open("w", encoding="utf-8") as stream:
                    for index in range(5):
                        row = {
                            "input": f"{prefix}-{index}",
                            "output": prefix,
                            "left_context": None,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            output = root / "run" / "train.jsonl"
            generate_train_jsonl(sources, output, wikipedia_count=2, corpus_count=3)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["input"] for row in rows],
                ["wiki-0", "wiki-1", "corpus-0", "corpus-1", "corpus-2"],
            )
            generate_train_jsonl(sources, output, wikipedia_count=2, corpus_count=3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)

            with self.assertRaisesRegex(ValueError, "不足"):
                generate_train_jsonl(sources, output, wikipedia_count=6, corpus_count=0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)

    def test_skips_empty_input_or_output_and_collects_requested_valid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "dataset"
            sources.mkdir()
            rows = [
                {"input": "", "output": "不正", "left_context": None},
                {"input": "ア", "output": "亜", "left_context": None},
                {"input": "イ", "output": "", "left_context": None},
                {"input": "ウ", "output": "右", "left_context": None},
            ]
            for filename in ("train_wikipedia.jsonl", "train_llm-jp-corpus-v3.jsonl"):
                (sources / filename).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
            output = root / "train.jsonl"
            generate_train_jsonl(sources, output, wikipedia_count=2, corpus_count=2)
            self.assertEqual(
                [json.loads(line)["input"] for line in output.read_text().splitlines()],
                ["ア", "ウ", "ア", "ウ"],
            )

    def test_insufficient_valid_rows_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "dataset"
            sources.mkdir()
            (sources / "train_wikipedia.jsonl").write_text(
                '{"input": "", "output": "正", "left_context": null}\n'
            )
            output = root / "train.jsonl"
            output.write_text("previous\n")
            with self.assertRaisesRegex(ValueError, "有効な例が0件"):
                generate_train_jsonl(sources, output, wikipedia_count=1, corpus_count=0)
            self.assertEqual(output.read_text(), "previous\n")

    def test_malformed_json_fails_with_source_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "dataset"
            sources.mkdir()
            path = sources / "train_wikipedia.jsonl"
            path.write_text('{"input": "ア", "output": "亜"}\n{\n')
            with self.assertRaises(ValueError) as raised:
                generate_train_jsonl(sources, root / "train.jsonl", 2, 0)
            self.assertIn(f"{path}:2:", str(raised.exception))

    def test_rejects_negative_and_zero_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for counts in ((-1, 2), (0, 0)):
                with self.subTest(counts=counts), self.assertRaises(ValueError):
                    generate_train_jsonl(root, root / "train.jsonl", *counts)
