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

    def test_rejects_negative_and_zero_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for counts in ((-1, 2), (0, 0)):
                with self.subTest(counts=counts), self.assertRaises(ValueError):
                    generate_train_jsonl(root, root / "train.jsonl", *counts)
