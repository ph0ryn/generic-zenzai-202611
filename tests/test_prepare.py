import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generic_zenzai_202611.config import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SOURCES,
    MODEL_ID,
    MODEL_REVISION,
)
from generic_zenzai_202611.prepare import main


class PrepareTest(unittest.TestCase):
    def run_cli(self, root):
        with (
            patch("sys.argv", ["prepare", "--dataset-dir", str(root / "dataset")]),
            patch("generic_zenzai_202611.prepare.MODEL_DIR", root / "model"),
        ):
            main()

    def test_fixed_model_revision_is_saved_reused_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            for name in DATASET_SOURCES:
                (root / "dataset" / name).touch()
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=self.fake_download,
            ):
                self.run_cli(root)
            marker = root / "model" / ".base-model-revision"
            self.assertEqual(marker.read_text().strip(), MODEL_REVISION)
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=AssertionError("unexpected download"),
            ):
                self.run_cli(root)
                marker.write_text("a" * 40 + "\n")
                with self.assertRaisesRegex(ValueError, "revisionが異なります"):
                    self.run_cli(root)

    def fake_download(self, *, repo_id, revision, local_dir, **kwargs):
        local_dir.mkdir(parents=True, exist_ok=True)
        if repo_id == DATASET_ID:
            self.assertEqual(revision, DATASET_REVISION)
            self.assertEqual(
                kwargs,
                {"repo_type": "dataset", "allow_patterns": list(DATASET_SOURCES)},
            )
            for name in DATASET_SOURCES:
                (local_dir / name).write_text('{"input": "アツイ", "output": "熱い"}\n')
        else:
            self.assertEqual(repo_id, MODEL_ID)
            self.assertEqual(revision, MODEL_REVISION)
            self.assertEqual(kwargs, {})
            (local_dir / "config.json").write_text('{"model_type": "llama"}')
            (local_dir / "model.safetensors").write_bytes(b"weights")

    def test_download_then_offline_cli_reuses_files_without_changes(self):
        with tempfile.TemporaryDirectory(prefix="prepare with spaces ") as directory:
            root = Path(directory)
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=self.fake_download,
            ) as download:
                self.run_cli(root)
            self.assertEqual(download.call_count, 2)
            before = {
                str(path.relative_to(root)): (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
                for path in root.rglob("*")
                if path.is_file()
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path\n"
                        "from generic_zenzai_202611 import config\n"
                        f"config.MODEL_DIR = Path({str(root / 'model')!r})\n"
                        "from generic_zenzai_202611.prepare import main\n"
                        "main()\n"
                    ),
                    "--dataset-dir",
                    str(root / "dataset"),
                ],
                check=False,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "PATH": str(Path(sys.executable).parent)
                    + os.pathsep
                    + os.environ["PATH"],
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("保存済み", result.stdout)
            self.assertIn(str(root / "model"), result.stdout)
            self.assertIn(str(root / "dataset"), result.stdout)
            self.assertEqual(
                before,
                {
                    str(path.relative_to(root)): (
                        path.read_bytes(),
                        path.stat().st_mtime_ns,
                    )
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )

    def test_existing_dataset_is_reused_without_downloading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            for name in DATASET_SOURCES:
                (dataset / name).write_text("existing data")
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=self.fake_download,
            ) as download:
                self.run_cli(root)
            download.assert_called_once_with(
                repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=root / "model"
            )
            for name in DATASET_SOURCES:
                self.assertEqual((dataset / name).read_text(), "existing data")

    def test_interrupted_model_download_is_retried_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A complete dataset must not be downloaded again after model failure.
            self.fake_download(
                repo_id=DATASET_ID,
                revision=DATASET_REVISION,
                local_dir=root / "dataset",
                repo_type="dataset",
                allow_patterns=list(DATASET_SOURCES),
            )
            destination = root / "model"
            destination.mkdir()
            (destination / "config.json").write_text("{}")
            with (
                patch(
                    "generic_zenzai_202611.prepare.snapshot_download",
                    side_effect=OSError("interrupted"),
                ),
                self.assertRaisesRegex(OSError, "interrupted"),
            ):
                self.run_cli(root)
            self.assertEqual(
                [path.name for path in destination.iterdir()], ["config.json"]
            )
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=self.fake_download,
            ) as download:
                self.run_cli(root)
            download.assert_called_once_with(
                repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=destination
            )

    def test_partial_dataset_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / DATASET_SOURCES[0]).write_text("existing data")
            with patch(
                "generic_zenzai_202611.prepare.snapshot_download",
                side_effect=self.fake_download,
            ) as download:
                self.run_cli(root)
            self.assertEqual(download.call_count, 2)
            self.assertTrue(all((dataset / name).is_file() for name in DATASET_SOURCES))


if __name__ == "__main__":
    unittest.main()
