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
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611
```

```python
!uv run --locked generic-zenzai-evaluate --model "$(cat model-path.txt)"
```

学習データ5,000件は同梱済み。モデルは初回にHugging Faceから取得する。
uvがプロジェクト用のPython 3.14と依存関係を用意する。
所要時間と消費コンピュートユニットは、割り当てられたGPUによって変わる。

## 詳しい使い方

- [学習設定・保存と再開](docs/training.md)
- [AJIMEE評価の実行・指標・結果](docs/evaluation.md)
- [ローカル開発・テスト](docs/development.md)
- [学習データの出典・ライセンス・制限](data/README.md)
