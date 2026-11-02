import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from generic_zenzai_202611.train import (
    encode_example,
    make_prompt,
    mixed_precision,
    prepare_dataset,
    resolve_run,
)


def make_tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["左文脈：お湯に触れた。読み：アツイ変換：熱い暑い"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[PAD]", "<s>", "</s>"],
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )


class RunTest(unittest.TestCase):
    def checkpoint(self, run, step):
        path = run / f"checkpoint-{step}"
        path.mkdir(parents=True)
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "config.json",
            "model.safetensors",
        ):
            (path / name).touch()
        (path / "trainer_state.json").write_text("{}")
        return path

    def test_new_runs_are_unique_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first, checkpoint = resolve_run(output)
            self.assertIsNone(checkpoint)
            (first / "run.json").write_text("original")
            second, _ = resolve_run(output)
            self.assertNotEqual(first, second)
            self.assertTrue(second.is_dir())
            self.assertEqual((first / "run.json").read_text(), "original")

    def test_resume_selects_latest_run_and_latest_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.checkpoint(output / "run-001", 100)
            run = output / "run-002"
            checkpoint = self.checkpoint(run, 50)
            self.checkpoint(run, 9)
            incomplete = self.checkpoint(run, 60)
            (incomplete / "optimizer.pt").unlink()
            self.assertEqual(resolve_run(output, resume=True), (run, checkpoint))
            self.assertEqual(
                resolve_run(output, resume_from=checkpoint), (run, checkpoint)
            )
            self.assertEqual(
                resolve_run(run, resume_from=checkpoint), (run, checkpoint)
            )

    def test_resume_failure_never_creates_run_or_falls_back_to_older_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                resolve_run(output, resume=True)
            self.assertFalse(output.exists())
            self.checkpoint(output / "run-001", 50)
            latest = output / "run-002"
            latest.mkdir()
            before = sorted(output.iterdir())
            with self.assertRaises(ValueError):
                resolve_run(output, resume=True)
            self.checkpoint(latest, 50)
            (latest / "train_results.json").write_text("{}")
            with self.assertRaises(ValueError):
                resolve_run(output, resume=True)
            self.assertEqual(sorted(output.iterdir()), before)

    def test_resume_accepts_each_supported_weight_format(self):
        for name, sharded in (
            ("model.safetensors", False),
            ("pytorch_model.bin", False),
            ("model.safetensors.index.json", True),
            ("pytorch_model.bin.index.json", True),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                checkpoint = self.checkpoint(output / "run-001", 1)
                (checkpoint / "model.safetensors").unlink()
                if sharded:
                    (checkpoint / name).write_text(
                        json.dumps({"weight_map": {"layer": "shard-001"}})
                    )
                    (checkpoint / "shard-001").touch()
                else:
                    (checkpoint / name).touch()
                self.assertEqual(
                    resolve_run(output, resume=True), (checkpoint.parent, checkpoint)
                )

    def test_resume_rejects_incomplete_or_invalid_checkpoint_metadata(self):
        for contents, shard_exists in (
            ("{", True),
            ("{}", True),
            ('{"weight_map": {}}', True),
            ('{"weight_map": {"layer": "shard-001"}}', False),
        ):
            with (
                self.subTest(contents=contents),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory)
                checkpoint = self.checkpoint(output / "run-001", 1)
                (checkpoint / "model.safetensors").unlink()
                (checkpoint / "model.safetensors.index.json").write_text(contents)
                if shard_exists:
                    (checkpoint / "shard-001").touch()
                with self.assertRaises(ValueError):
                    resolve_run(output, resume_from=checkpoint)
                with self.assertRaises(ValueError):
                    resolve_run(output, resume=True)

    def test_explicit_resume_rejects_unrelated_output_and_invalid_trainer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = self.checkpoint(output / "run-001", 1)
            with self.assertRaises(ValueError):
                resolve_run(output / "other", resume_from=checkpoint)
            (checkpoint / "trainer_state.json").write_text("{")
            with self.assertRaises(ValueError):
                resolve_run(output, resume_from=checkpoint)


class PrecisionTest(unittest.TestCase):
    def test_precision_requires_native_bf16_support(self):
        for cpu, native_bf16, expected in (
            (False, False, "fp16"),  # T4 can emulate bf16 but cannot run its CUDA GEMM.
            (False, True, "bf16"),
            (True, False, "fp32"),
        ):
            with (
                self.subTest(cpu=cpu, native_bf16=native_bf16),
                patch(
                    "torch.cuda.is_bf16_supported",
                    side_effect=lambda including_emulation=True, native=native_bf16: (
                        native or including_emulation
                    ),
                ),
            ):
                self.assertEqual(mixed_precision(cpu), expected)


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = make_tokenizer()
        self.row = {"input": "アツイ", "output": "熱い", "left_context": None}

    def test_mask_preserves_prompt_and_supervises_answer_and_eos(self):
        example = encode_example(self.row, self.tokenizer)
        boundary = example["completion_mask"].index(1)
        prompt = self.tokenizer.encode(make_prompt("アツイ", None))
        self.assertEqual(example["input_ids"][:boundary], prompt)
        self.assertEqual(
            self.tokenizer.decode(example["input_ids"][boundary:-1]), "熱い"
        )
        self.assertEqual(example["input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(example["completion_mask"][:boundary], [0] * len(prompt))
        self.assertEqual(
            example["completion_mask"][boundary:],
            [1] * (len(example["input_ids"]) - boundary),
        )

    def test_jsonl_keeps_unicode_line_separators_inside_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            self.row["output"] = "熱い\u0085お湯\u2028です"
            path.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
            dataset = prepare_dataset(path, self.tokenizer)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0], encode_example(self.row, self.tokenizer))

    def test_invalid_or_overlong_rows_fail_instead_of_being_dropped(self):
        for change in (
            {"input": ""},
            {"output": None},
            {"left_context": 123},
            {"output": "熱い" * 600},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                encode_example({**self.row, **change}, self.tokenizer)
        with self.assertRaises(KeyError):
            encode_example({"input": "アツイ", "output": "熱い"}, self.tokenizer)

    def test_jsonl_errors_include_path_and_physical_line_number(self):
        invalid_rows = ("{", "null", json.dumps({"input": "アツイ"}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            first = json.dumps(self.row, ensure_ascii=False) + "\n"
            for invalid in invalid_rows:
                with self.subTest(invalid=invalid):
                    path.write_text(first + invalid + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError) as raised:
                        prepare_dataset(path, self.tokenizer)
                    self.assertIn(f"{path}:2:", str(raised.exception))
                    self.assertIsNotNone(raised.exception.__cause__)

    def test_empty_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.touch()
            with self.assertRaisesRegex(ValueError, "学習データが空"):
                prepare_dataset(path, self.tokenizer)

    def test_context_uses_last_40_characters(self):
        self.assertEqual(
            make_prompt("アツイ", "古い文脈" + "あ" * 40),
            make_prompt("アツイ", "あ" * 40),
        )


class TrainingTest(unittest.TestCase):
    def test_cli_trains_saves_reloads_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            tokenizer = make_tokenizer()
            tokenizer.save_pretrained(base)
            model = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=512,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
            before = model.get_input_embeddings().weight.detach().clone()
            model.save_pretrained(base)
            data = root / "train.jsonl"
            rows = [
                {"input": "アツイ", "output": "熱い", "left_context": "お湯に触れた。"},
                {"input": "アツイ", "output": "暑い", "left_context": None},
            ] * 17
            data.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            output = root / "runs with spaces and 'quotes'"
            exported_model = root / "model-path.txt"
            command = [
                str(Path(sys.executable).with_name("generic-zenzai-202611")),
                "--model",
                str(base),
                "--data",
                str(data),
                "--output",
                str(output),
                "--cpu",
                "--save-steps",
                "1",
            ]
            env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
            }
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=root,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("計算精度: fp32", result.stdout)
            self.assertIn(
                "学習データ: 34件 | 実効バッチ: 32 | 周回数: 1", result.stdout
            )
            self.assertIn("step 1/2 | loss:", result.stdout)
            self.assertIn("step 2/2 | train_runtime:", result.stdout)
            self.assertIn("train_loss:", result.stdout)
            self.assertNotIn("{'loss':", result.stdout)
            self.assertNotIn("{'train_runtime':", result.stdout)
            self.assertNotIn("[学習]", result.stdout)
            output = next(output.glob("run-*"))
            self.assertIn(f"出力先: {output}", result.stdout)
            self.assertTrue((output / "model" / "model.safetensors").is_file())
            self.assertEqual(
                exported_model.read_text(encoding="utf-8"),
                str((output / "model").resolve()) + "\n",
            )
            trained = AutoModelForCausalLM.from_pretrained(output / "model")
            restored_tokenizer = AutoTokenizer.from_pretrained(output / "model")
            self.assertFalse(torch.equal(before, trained.get_input_embeddings().weight))
            inputs = restored_tokenizer(
                "読み：アツイ\n変換：\n",
                return_tensors="pt",
                return_token_type_ids=False,
            )
            with torch.no_grad():
                tokens = trained.generate(**inputs, max_new_tokens=2, do_sample=False)
            self.assertGreater(tokens.shape[-1], inputs.input_ids.shape[-1])
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["examples"], 34)
            metrics = json.loads((output / "train_results.json").read_text())
            self.assertEqual(metrics["global_step"], 2)
            self.assertEqual(metrics["precision"], "fp32")
            self.assertEqual(metrics["effective_batch_size"], 32)
            self.assertTrue(math.isfinite(metrics["train_loss"]))
            checkpoint = output / "checkpoint-2"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())

            # Automatic resume must not restart a completed run.
            repeated = subprocess.run(
                command + ["--resume"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=root,
                timeout=120,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("完了済み", repeated.stderr)
            # Simulate interruption after checkpoint save but before final metrics.
            (output / "train_results.json").unlink()
            resumed = subprocess.run(
                command + ["--resume", "--epochs", "2"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=root,
                timeout=120,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            state = json.loads(
                (output / "checkpoint-4" / "trainer_state.json").read_text()
            )
            self.assertEqual(state["global_step"], 4)
            self.assertEqual(
                exported_model.read_text(encoding="utf-8"),
                str((output / "model").resolve()) + "\n",
            )

            # A failed new training must not leave the previous model exported.
            failed = subprocess.run(
                command + ["--data", str(root / "missing.jsonl")],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=root,
                timeout=120,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(exported_model.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
