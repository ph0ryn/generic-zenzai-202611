import json
import os
import subprocess
import sys
import unittest

import test_evaluate

from generic_zenzai_202611.inference import Converter


class GenerationDefaultsTest(unittest.TestCase):
    def setUp(self):
        # Reuse the real local model and benchmark without inheriting their tests.
        self.fixture = test_evaluate.EvaluationTest()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def test_converter_defaults_to_64_generated_tokens(self):
        converter = Converter(self.fixture.model, cpu=True)
        inputs = converter.prepare_inputs(self.fixture.rows[:1])
        prediction = converter.predict(inputs[0])
        self.assertEqual(prediction["generated_tokens"], 64)
        self.assertTrue(prediction["truncated"])

    def test_cli_defaults_to_64_generated_tokens(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "generic_zenzai_202611.evaluate",
                "--model",
                str(self.fixture.model),
                "--benchmark",
                str(self.fixture.benchmark),
                "--output",
                str(self.fixture.output),
                "--cpu",
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
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = next(self.fixture.output.glob("eval-*"))
        metadata = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(metadata["generation"]["max_new_tokens"], 64)
        predictions = [
            json.loads(line)
            for line in (output / "predictions.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(row["generated_tokens"] == 64 for row in predictions))
        self.assertTrue(all(row["truncated"] for row in predictions))


if __name__ == "__main__":
    unittest.main()
