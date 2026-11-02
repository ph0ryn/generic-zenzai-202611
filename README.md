# generic-zenzai-202611

LLM-jp-3-150mを、文脈と読みからかな漢字変換するモデルに追加学習する。
uvで環境を固定し、ColabからCLIで学習を実行する。

## Colabで学習

Colabで新規ノートブックを作る。
ランタイムをGPUに変更し、以下の3ブロックを別々のセルにコピーして、
上から順に実行する。

```python
!git clone https://github.com/ph0ryn/generic-zenzai-202611.git
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
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611/run-001
```

学習データ5,000件は同梱済み。モデルは初回にHugging Faceから取得する。
uvがプロジェクト用のPython 3.14と依存関係を用意する。
GPUの種類、VRAM、残りのコンピュートユニットは実行時に確認する。
所要時間と消費量はGPUの割り当てによって変わる。

## 何を学習するか

入力は「直前40文字の文脈＋カタカナの読み」、正解は変換後の文字列。
入力の再現には損失を付けず、正解と終了記号だけを予測するよう学習する。

| 設定 | 初期値・意味 |
| --- | --- |
| モデル | `llm-jp/llm-jp-3-150m`、全パラメータを更新 |
| 周回数 | 1 epoch。5,000件を1回ずつ学習 |
| 学習率 | `5e-5`。最初の5%で増やし、その後は線形に下げる |
| バッチ | 2件ずつ計算し、16回分の勾配をまとめて更新。実効32件 |
| 長さ上限 | 入力と正解を合わせて512トークン。超過時は切らずに停止 |
| 精度 | bf16のハードウェア対応GPUではbf16、T4などではfp16混合精度 |
| 保存 | 50更新ごと。最新2個のチェックポイントを保持 |

文脈がないデータも2,500件含む。
開始時に計算精度を表示する。T4では`計算精度: fp16`になる。
学習中の`loss`は正解の予測誤差であり、変換精度そのものではない。
まず学習と保存を完了し、その後にAJIMEEで評価する。
このプロジェクトには評価データの作成・評価実行を含めていない。

モデルのrevisionは
`b112feef602fff752e4dac4c30af6a2c2fa41c7a`に固定。
出典は[LLM-jpのモデルカード](https://huggingface.co/llm-jp/llm-jp-3-150m)。
データの出典・ライセンス・品質上の制限は[data/README.md](data/README.md)を参照。

## 保存と再開

指定した`--output`に次を保存する。

- `model/`: 学習済みモデルとtokenizer。`from_pretrained`で読み込める。
- `checkpoint-N/`: モデル、optimizer、学習進捗などの再開用状態。
- `run.json`: モデルrevision、データのSHA-256、学習設定。
- `train_results.json`: 完了時の学習損失、速度など。

途中で切断したらclone・環境準備・Driveの接続をやり直し、
学習コマンドに最後まで保存されたチェックポイントを指定する。
`checkpoint-50`は実際に存在する名前に置き換える。

```python
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611/run-001 --resume-from /content/drive/MyDrive/generic-zenzai-202611/run-001/checkpoint-50
```

再開は保存時点から。保存後に進んだ計算はやり直す。
モデル・データ・主要な学習設定が異なる場合は停止する。
新しい実験では`run-002`など別の出力先を使う。
既存の出力先への誤った上書きは拒否する。

学習開始直後のエラーでチェックポイントがない場合は、
コードを更新し、未使用の出力先で最初から実行する。
以下の`run-002`は、すでに使用していれば別の名前に変更する。

```python
%cd /content/generic-zenzai-202611
!git pull --ff-only
!uv sync --locked
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611/run-002
```

## ローカル開発

```sh
uv sync --locked
uv run task lint
uv run task test
```

テストはダウンロード不要の小さなモデルをCPUで実際に学習し、
重みの更新・保存・推論・再開・上書き拒否を確認する。
通常の学習コマンドにはCUDA GPUが必要。
`--cpu`は小規模な動作確認用。
