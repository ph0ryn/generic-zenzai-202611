# generic-zenzai-202611

LLM-jp-3-150mを追加学習し、読みと直前の文脈からかな漢字変換するモデルを作る。
Colab上で学習し、AJIMEE-Benchで変換精度を評価できる。

## Usage

Colabで新規ノートブックを作る。
ランタイムをGPUに変更し、以下の4ブロックを別々のセルにコピーして、
上から順に実行する。

```python
!git clone --recurse-submodules https://github.com/ph0ryn/generic-zenzai-202611.git
%cd /content/generic-zenzai-202611
%pip install uv==0.13.4
!uv sync --locked
!nvidia-smi
```

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!uv run --locked train \
    --wikipedia-count 7500 \
    --corpus-count 7500 \
    --epochs 1 \
    --output /content/drive/MyDrive/generic-zenzai-202611
```

```python
!uv run --locked evaluate --model "$(cat model-path.txt)"
```

評価の最後に出る「実験サマリー」を貼れば、学習条件と評価結果をまとめて共有できる。

学習データはDrive上の元データ2本の先頭から指定件数だけ読み、run内に`train.jsonl`を生成する。
事前に下記の手順でDriveにデータを保存する。モデルは初回にHugging Faceから取得する。
uvがプロジェクト用のPython 3.14と依存関係を用意する。
所要時間と消費コンピュートユニットは、割り当てられたGPUによって変わる。

## dataset

初回のみ、別のColabノートブックで以下を実行しDriveへ元データを保存する。
Driveに約36.5 GBの空きが必要。

```python
%pip -q install huggingface_hub

from pathlib import Path
from google.colab import drive
from huggingface_hub import snapshot_download

drive.mount("/content/drive")

destination = Path("/content/drive/MyDrive/datasets/zenz-v2.5-dataset")
revision = "7d9c9ea36347b638a627c1d264da48ebc7eb38aa"

snapshot_download(
    repo_id="Miwa-Keita/zenz-v2.5-dataset",
    repo_type="dataset",
    revision=revision,
    allow_patterns=[
        "train_wikipedia.jsonl",
        "train_llm-jp-corpus-v3.jsonl",
    ],
    local_dir=destination,
)

for name in ("train_wikipedia.jsonl", "train_llm-jp-corpus-v3.jsonl"):
    path = destination / name
    print(f"{path}: {path.stat().st_size / 1_000_000_000:.2f} GB")

print("準備完了:", destination)
```

## 詳しい使い方

- [学習設定・保存と再開](docs/training.md)
- [AJIMEE評価の実行・指標・結果](docs/evaluation.md)
- [ローカル開発・テスト](docs/development.md)
- [学習データの出典・ライセンス・制限](docs/training.md#学習データの出典と制限)
