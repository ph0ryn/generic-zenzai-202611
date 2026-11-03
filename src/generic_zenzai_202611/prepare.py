"""Prepare reusable dataset and base-model snapshots on mounted Drive storage."""

import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import (
    DATASET_DIR,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SOURCES,
    MODEL_DIR,
    MODEL_ID,
    MODEL_REVISION,
)


def prepared_model_path(destination, expected_revision=MODEL_REVISION):
    """Validate a prepared snapshot without downloading or changing any files."""
    marker = destination / ".base-model-revision"
    try:
        revision = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError(
            f"{destination}: ベースモデルが未準備です。先にuv run --locked task prepareを実行してください。"
        ) from error
    if revision != expected_revision:
        raise ValueError(f"{destination}: モデルのrevisionが異なります。")
    return destination


def prepare_model(destination, model=MODEL_ID, revision=MODEL_REVISION):
    """Reuse a completed snapshot, or download/resume it at the pinned revision."""
    marker = destination / ".base-model-revision"
    if marker.is_file():
        prepared_model_path(destination, revision)
        print(f"ベースモデルは保存済み: {destination}", flush=True)
    else:
        print(f"ベースモデルを取得: {destination}", flush=True)
        snapshot_download(repo_id=model, revision=revision, local_dir=destination)
        # Directory existence alone cannot distinguish an interrupted download.
        marker.write_text(revision + "\n", encoding="utf-8")
    return destination


def prepare_dataset(destination):
    """Reuse both source files, including those saved by the original notebook."""
    if all((destination / name).is_file() for name in DATASET_SOURCES):
        print(f"データセットは保存済み: {destination}", flush=True)
    else:
        print(f"データセットを取得: {destination}", flush=True)
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=list(DATASET_SOURCES),
            local_dir=destination,
        )
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="データセットとベースモデルをDriveに保存し、保存済みなら再利用する"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="保存するモデルの固定commit SHA")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.model != MODEL_ID and (not args.revision or args.model_dir is None):
        parser.error("別モデルには--revisionと--model-dirを指定してください。")
    revision = args.revision or MODEL_REVISION
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("revisionには40桁のcommit SHAを指定してください。")
    prepare_dataset(args.dataset_dir)
    prepare_model(args.model_dir or MODEL_DIR, args.model, revision)
    print("準備完了", flush=True)
