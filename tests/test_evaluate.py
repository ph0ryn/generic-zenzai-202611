import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        backend = Tokenizer(
            models.WordLevel({"熱い": 0, "[UNK]": 1, "[PAD]": 2, "[EOS]": 3}, "[UNK]")
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        tokenizer.save_pretrained(self.model)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=512,
                eos_token_id=3,
                pad_token_id=2,
            )
        )
        # All logits tie: real greedy generation always chooses token 0 (熱い).
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        # Evaluation must override settings inherited from a saved checkpoint.
        model.generation_config.do_sample = True
        model.generation_config.num_beams = 2
        model.generation_config.bad_words_ids = [[0]]
        model.save_pretrained(self.model)
        self.benchmark = self.root / "benchmark"
        self.data = self.benchmark / "JWTD_v2/v1/evaluation_items.json"
        self.data.parent.mkdir(parents=True)
        self.rows = [
            {
                "index": "a",
                "input": "アツイ",
                "context_text": "お湯に触れた。",
                "expected_output": ["暑い", "熱い"],
            },
            {
                "index": "b",
                "input": "サムイ",
                "context_text": "",
                "expected_output": ["寒い"],
            },
        ]
        self.data.write_text(
            json.dumps(self.rows, ensure_ascii=False), encoding="utf-8"
        )
        self.git("init", "-q")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        )
        self.output = self.root / "results"

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.benchmark), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def evaluate(self, *extra, benchmark=None):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "generic_zenzai_202611.evaluate",
                "--model",
                str(self.model),
                "--benchmark",
                str(benchmark or self.benchmark),
                "--output",
                str(self.output),
                "--cpu",
                "--max-new-tokens",
                "1",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
            },
        )

    def test_cli_generates_scores_and_preserves_repeat_runs(self):
        result = self.evaluate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        first = next(self.output.glob("eval-*"))
        predictions = [
            json.loads(line)
            for line in (first / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["index"] for row in predictions], ["a", "b"])
        self.assertEqual([row["prediction"] for row in predictions], ["熱い", "熱い"])
        self.assertEqual([row["accuracy_at1"] for row in predictions], [1, 0])
        self.assertEqual([row["min_cer"] for row in predictions], [0, 0.5])
        self.assertTrue(all(row["truncated"] for row in predictions))
        self.assertEqual(predictions[0]["context_text"], "お湯に触れた。")
        scores = json.loads((first / "scores.json").read_text())
        self.assertEqual(scores["all"]["count"], 2)
        self.assertEqual(scores["all"]["accuracy_at1"], 0.5)
        self.assertEqual(scores["all"]["mean_min_cer"], 0.25)
        self.assertEqual(scores["all"]["truncated"], 2)
        self.assertEqual(scores["with_context"]["accuracy_at1"], 1)
        self.assertEqual(scores["without_context"]["accuracy_at1"], 0)
        metadata = json.loads((first / "evaluation.json").read_text())
        self.assertEqual(
            metadata["benchmark"]["revision"], self.git("rev-parse", "HEAD")
        )
        self.assertEqual(
            metadata["benchmark"]["data_sha256"],
            hashlib.sha256(self.data.read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["generation"]["max_new_tokens"], 1)
        self.assertFalse(metadata["generation"]["do_sample"])
        self.assertEqual(metadata["generation"]["num_beams"], 1)
        self.assertIsNone(metadata["generation"]["bad_words_ids"])
        self.assertEqual(metadata["context_chars"], 40)
        self.assertEqual(metadata["model"], str(self.model.resolve()))
        self.assertIn("全体", result.stdout)
        repeated = self.evaluate()
        self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
        self.assertEqual(len(list(self.output.glob("eval-*"))), 2)
        self.assertEqual(
            [
                json.loads(line)
                for line in (first / "predictions.jsonl").read_text().splitlines()
            ],
            predictions,
        )

    def test_eos_stops_generation_and_is_not_marked_truncated(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        tokenizer.eos_token = "熱い"  # Token 0 is always generated by this model.
        tokenizer.save_pretrained(self.model)
        result = self.evaluate("--max-new-tokens", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = next(self.output.glob("eval-*"))
        predictions = [
            json.loads(line)
            for line in (output / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(row["prediction"] == "" for row in predictions))
        self.assertTrue(all(row["generated_tokens"] == 1 for row in predictions))
        self.assertTrue(all(not row["truncated"] for row in predictions))
        self.assertTrue(all(row["min_cer"] == 1 for row in predictions))

    def test_uninitialized_submodule_explains_how_to_initialize(self):
        result = self.evaluate(benchmark=self.root / "missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git submodule update --init", result.stderr)
        self.assertFalse(self.output.exists())

    def test_modified_benchmark_is_not_silently_reported_as_clean_revision(self):
        self.data.write_text("[]", encoding="utf-8")
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("変更", result.stderr)
        self.assertFalse(self.output.exists())

    def test_overlong_prompt_and_invalid_generation_limit_fail_before_output(self):
        for extra in (("--max-new-tokens", "0"), ("--max-new-tokens", "512")):
            with self.subTest(extra=extra):
                result = self.evaluate(*extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
