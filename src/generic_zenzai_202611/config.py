"""Pinned input artifacts and shared Drive locations for preparation and training."""

from pathlib import Path

MODEL_ID = "LiquidAI/LFM2.5-350M-Base"
MODEL_REVISION = "9960764e30892e01f29a6dc23df2533fcd8bd5ae"
MODEL_DIR = Path("/content/drive/MyDrive/models/LFM2.5-350M-Base")

DATASET_ID = "Miwa-Keita/zenz-v2.5-dataset"
DATASET_REVISION = "7d9c9ea36347b638a627c1d264da48ebc7eb38aa"
DATASET_DIR = Path("/content/drive/MyDrive/datasets/zenz-v2.5-dataset")
DATASET_SOURCES = ("train_wikipedia.jsonl", "train_llm-jp-corpus-v3.jsonl")
