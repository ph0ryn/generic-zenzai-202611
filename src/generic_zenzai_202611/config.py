"""Pinned input artifacts and shared Drive locations for preparation and training."""

from pathlib import Path

MODEL_ID = "llm-jp/llm-jp-3-150m"
MODEL_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"
MODEL_DIR = Path("/content/drive/MyDrive/models/llm-jp-3-150m")

DATASET_ID = "Miwa-Keita/zenz-v2.5-dataset"
DATASET_REVISION = "7d9c9ea36347b638a627c1d264da48ebc7eb38aa"
DATASET_DIR = Path("/content/drive/MyDrive/datasets/zenz-v2.5-dataset")
DATASET_SOURCES = ("train_wikipedia.jsonl", "train_llm-jp-corpus-v3.jsonl")
