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
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611
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

指定した`--output`の下に`run-日時`ディレクトリを自動作成し、次を保存する。
実際の出力先は開始時に表示する。通常実行のたびに新しいrunを作り、既存のrunは上書きしない。

- `model/`: 学習済みモデルとtokenizer。`from_pretrained`で読み込める。
- `checkpoint-N/`: モデル、optimizer、学習進捗などの再開用状態。
- `run.json`: モデルrevision、データのSHA-256、学習設定。
- `train_results.json`: 完了時の学習損失、速度など。

途中で切断したらclone・環境準備・Driveの接続をやり直し、
同じ`--output`に`--resume`を付けて実行する。
最新のrun内で、再開用ファイルが揃った最後のチェックポイントを自動選択する。

```python
!uv run --locked generic-zenzai-202611 --output /content/drive/MyDrive/generic-zenzai-202611 --resume
```

再開は保存時点から。保存後に進んだ計算はやり直す。
最新のrunにチェックポイントがない、完了済み、またはモデル・データ・主要な学習設定が
異なる場合はエラー終了する。古いrunへの切り替えや、新規学習への切り替えはしない。
特定のチェックポイントから再開する場合は、`--resume`の代わりに
`--resume-from /実際のrun/checkpoint-N`を指定する。

学習開始直後のエラーでチェックポイントがない場合は、原因を解消してから
`--resume`なしで実行する。新しい日時付きrunで最初から学習し、失敗したrunは残す。
修正版のコードが必要な場合は、実行前に`git pull --ff-only`と`uv sync --locked`で更新する。

## ローカル開発

```sh
uv sync --locked
uv run task lint
uv run task test
```

テストはダウンロード不要の小さなモデルをCPUで実際に学習し、
重みの更新・保存・推論・再開・出力先の分離・再開できない場合の終了を確認する。
通常の学習コマンドにはCUDA GPUが必要。
`--cpu`は小規模な動作確認用。
