import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from generic_zenzai_202611.config import MODEL_ID, MODEL_REVISION
from generic_zenzai_202611.prepare import main as prepare
from generic_zenzai_202611.train import main as train


class ModelCliTest(unittest.TestCase):
    def test_model_options_are_absent_from_help_and_rejected(self):
        for command, main, required in (
            ("prepare", prepare, []),
            ("train", train, ["--wikipedia-count", "1", "--corpus-count", "0"]),
        ):
            with self.subTest(command=command):
                output = io.StringIO()
                with (
                    patch("sys.argv", [command, "--help"]),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 0)
                for option in ("--model", "--revision", "--model-dir"):
                    self.assertNotIn(option, output.getvalue())

                for option, value in (
                    ("--model", MODEL_ID),
                    ("--revision", MODEL_REVISION),
                    ("--model-dir", "/unused-model"),
                ):
                    with self.subTest(option=option):
                        errors = io.StringIO()
                        with (
                            patch("sys.argv", [command, *required, option, value]),
                            redirect_stderr(errors),
                            self.assertRaises(SystemExit) as raised,
                        ):
                            main()
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn("unrecognized arguments", errors.getvalue())
                        self.assertIn(option, errors.getvalue())


if __name__ == "__main__":
    unittest.main()
