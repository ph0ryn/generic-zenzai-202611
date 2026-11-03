# generic-zenzai-202611

事前学習済みの小型LLMを追加学習し、読みと直前の文脈からかな漢字変換するモデルを作る。
Colab上で学習し、AJIMEE-Benchで変換精度を評価できる。
既定モデルはLFM2.5-350M-Base。選定理由と比較候補は[モデル選定](docs/models.md)を参照。

## Usage

Colabで新規ノートブックを作る。
ランタイムをGPUに変更し、以下の5ブロックを別々のセルにコピーして、
上から順に実行する。
以下はLFM2.5-350M-Baseを135,000件・3周で全パラメータ学習する手順。
バッチ32件・勾配累積1回を維持し、T4ではFP16で実行する。

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
!uv run --locked task prepare
```

```python
!uv run --locked train \
    --wikipedia-count 67500 \
    --corpus-count 67500 \
    --epochs 3 \
    --batch-size 32 \
    --gradient-accumulation 1 \
    --output /content/drive/MyDrive/generic-zenzai-202611/lfm2.5-350m
```

```python
!uv run --locked evaluate --model "$(cat model-path.txt)"
```

評価の最後に出る「実験サマリー」を貼れば、学習条件と評価結果をまとめて共有できる。

学習データはDrive上の元データ2本を先頭から読み、入力・正解が空の行を除外して各指定件数の有効な例を集め、run内に`train.jsonl`を生成する。
準備セルでデータセットとベースモデルをDriveに保存し、学習時に再利用する。
uvがプロジェクト用のPython 3.14と依存関係を用意する。
所要時間と消費コンピュートユニットは、割り当てられたGPUによって変わる。

## データセットとベースモデルの準備

準備セルは毎回同じコマンドを実行すればよい。
初回だけHugging Faceから取得し、保存済みなら通信せずにスキップする。
Driveには元データ約36.5 GBに加え、ベースモデル用の空きが必要。
別のデータセット準備用ノートブックは不要。

- データセット: `/content/drive/MyDrive/datasets/zenz-v2.5-dataset`
- ベースモデル: `/content/drive/MyDrive/models/LFM2.5-350M-Base`

従来の手順で保存したデータセットも、対象の2ファイルが揃っていればそのまま使う。
モデルはダウンロード成功後に完了印を保存し、中断した場合は次回に取得を再試行する。
モデル・データセットのID、固定revision、保存先、対象ファイルは
[共通設定](src/generic_zenzai_202611/config.py)で管理する。
データセットの保存先だけを変える場合は、準備・学習の両コマンドに同じ`--dataset-dir`を指定する。
ベースモデルのID・revision・保存先はCLI引数では変更せず、共通設定を使う。

## 詳しい使い方

- [学習設定・保存と再開](docs/training.md)
- [AJIMEE評価の実行・指標・結果](docs/evaluation.md)
- [ローカル開発・テスト](docs/development.md)
- [学習データの出典・ライセンス・制限](docs/training.md#学習データの出典と制限)
