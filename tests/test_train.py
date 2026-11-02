import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from generic_zenzai_202611.train import encode_example, make_prompt, prepare_dataset


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
            output = root / "run"
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
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "model" / "model.safetensors").is_file())
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
            self.assertTrue(math.isfinite(metrics["train_loss"]))
            checkpoint = output / "checkpoint-2"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())

            # An accidental second run must not overwrite the saved experiment.
            repeated = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            self.assertNotEqual(repeated.returncode, 0)
            resumed = subprocess.run(
                command + ["--resume-from", str(checkpoint), "--epochs", "2"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            state = json.loads(
                (output / "checkpoint-4" / "trainer_state.json").read_text()
            )
            self.assertEqual(state["global_step"], 4)


if __name__ == "__main__":
    unittest.main()
