# 学習設定と再開

Colabでの開始手順は[README](../README.md#usage)を参照。

## 学習内容と既定値

入力は「直前40文字までの文脈＋カタカナの読み」、正解は変換後の文字列。
損失は正解と終了記号にだけ付け、入力部分には付けない。
Drive上の元データから各ファイルの先頭を読む。`--wikipedia-count`と
`--corpus-count`でそれぞれの件数を必ず指定する（例: 各7,500件で計15,000件）。
元データが別の場所なら`--dataset-dir`を指定する。
生成した`train.jsonl`はrun内に保存する。
各ソースを先頭から順に読み、`input`または`output`が空・文字列でない行は除外して、
有効な例を指定件数ずつ集める。除外行数は実行時に表示する。
有効な例が指定件数に足りなければエラーにする。並べ替え・重複除去はしない。
JSONの破損やその他の入力検証エラーは除外せずに停止する。

| 設定 | 既定値 |
| --- | --- |
| モデル | `llm-jp/llm-jp-3-150m`、全パラメータを更新 |
| 周回数 | 1 epoch（`--epochs`で変更） |
| 学習率 | `5e-5`。最初の5%で増やし、その後は線形に下げる |
| バッチ | 32件 × 勾配累積1回（累積なし）。実効32件 |
| 長さ上限 | 入力・正解・終了記号を合わせて512トークン。超過時は切らずに停止 |
| 計算精度 | bf16ハードウェア対応GPUではbf16、それ以外のCUDA GPUではfp16 |
| チェックポイント | 50更新ごと（`--save-steps`で変更）。最新2個を保持 |

開始時に計算精度を表示する。T4では`計算精度: fp16`になる。
最初の更新時と以降10更新ごとに、`step 現在/全体 | loss: … | …`の形式で指標を表示する。
終了時の学習損失・速度なども同じ形式で表示する。
学習中の`loss`は予測誤差であり、変換精度ではない。
学習完了後、別コマンドで[AJIMEE評価](evaluation.md)を実行する。

既定モデルのrevisionは固定し、使用した値を`run.json`に記録する。
モデルの出典は[LLM-jpのモデルカード](https://huggingface.co/llm-jp/llm-jp-3-150m)、
学習データの出典・ライセンス・品質上の制限は[以下の説明](#学習データの出典と制限)を参照。
その他のCLIオプションは`uv run --locked train --help`で確認できる。

## 学習速度・VRAMのベンチマーク

### 測定済みの基準値

[PR #1のコメント](https://github.com/ph0ryn/generic-zenzai-202611/pull/1#issuecomment-6582714831)で共有された、**同じColabランタイムとGPU**を使った既存の比較結果。学習データは同一の5,000件（SHA-256一致）、1 epoch、fp16、seed 42、実効バッチ32、157更新で揃えている。

| ミニバッチ × 累積 | 学習時間 | 処理速度 |
| --- | ---: | ---: |
| 8 × 4（1回目） | 143.4秒 | 34.9件/秒 |
| 8 × 4（2回目） | 140.0秒 | 35.7件/秒 |
| 16 × 2 | 119.7秒 | 41.8件/秒 |
| **32 × 1（既定）** | **98.8秒** | **50.6件/秒** |

バッチサイズを増やす比較は**既に実施済み**。最速の32×1を既定値に採用している。現時点で、これらの実行のピークVRAM記録はない。

同条件のAJIMEE評価では16×2から32×1への変更で完全一致3/200→2/200、平均MinCER 0.6297→0.6313だった。ただし各条件の単一評価であり、精度差の傾向を断定できない。

### 追加した実験オプション

このPRで追加したものは、まだColab GPUで速度向上を実証していない。既定の学習設定は変更しない。

| フラグ | 既定値 | 比較する仮説 |
| --- | --- | --- |
| `--train-sampling-strategy` | `random` | `group_by_length`で同じ長さの系列をまとめ、バッチ内のpaddingを減らす |
| `--optim` | `adamw_torch` | `adamw_torch_fused`でoptimizerの処理を短縮する |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | 有効 | 再計算を減らす代わりにGPUメモリ消費が増える |
| `--batch-size` / `--gradient-accumulation-steps` | 32 / 1 | OOM時のフォールバック用。速度改善を再提案するものではない |

今後の新規runでは`train_results.json`にGPU名・ピークCUDAメモリのreserved/allocated・学習データの系列長合計を追加して記録する。CUDAメモリ測定は学習開始直前にリセットし、**学習の終了直後、モデル最終保存前**に確定させる。

### 追加実験の実行例

**32×1の既存計測（98.8秒）をベースライン**にして、各設定を単独で試す。過去runの速度と比べる際はGPU種類・依存パッケージのバージョン・学習データハッシュ・周回数・保存間隔などの違いに注意する。最新環境で厳密な比較が必要な場合だけ、32×1の既定設定を再測定する。

Colabで5,000件・1 epochを新規runに保存する例（上から全てを実施する必要はない）。

```bash
# A: paddingを削減する。その他は32×1の既定設定。
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --train-sampling-strategy group_by_length \
  --output /content/drive/MyDrive/generic-zenzai-202611

# B: fused AdamWだけを変更する。
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --optim adamw_torch_fused \
  --output /content/drive/MyDrive/generic-zenzai-202611

# C: gradient checkpointingだけを無効化する（T4ではOOMの可能性）。
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --no-gradient-checkpointing \
  --output /content/drive/MyDrive/generic-zenzai-202611
```

すべて元のモデルから学習を始め、既存runを上書きしない。各実験は同一データの`data_sha256`、同じ実効バッチ32・seed・学習率で比較する。学習時に系列の並びが変わったり、optimizerの演算が変わったりするため、完全に同一の更新を保証する比較ではない。高速化できてもAJIMEEの完全一致率とMinCERを確認する。

`--no-gradient-checkpointing`でOOMになる場合は無理に32×1で続けず、この条件は不採用にする。バッチ16×累積2への変更は別条件になり、速度の単独要因比較には使えない。

Driveへのcheckpoint保存がボトルネックなら、**別実験**で`--save-steps 200`などを試す。書き込み回数を減らせる反面、Colab切断時に再実行する更新数が増える。作成したrunを`/content`に保存して終了後にDriveへコピーする方法もあるが、コピー前の切断でデータを失うリスクがあるため、実験中のDrive保存とのトレードオフを判断する。

## 保存先

`--output`の下に毎回新しい`run-日時`を作り、開始時にパスを表示する。
通常実行では既存のrunを上書きしない。

- `model/`: 学習済みモデルとtokenizer。`from_pretrained`で読み込める。
- `checkpoint-N/`: optimizer・学習進捗などを含む再開用の状態。
- `run.json`: モデルrevision、データのSHA-256、学習設定。
- `train_results.json`: 完了時の学習損失・速度、更新回数、計算精度、実効バッチなど。

学習コマンドの実行ディレクトリには`model-path.txt`を自動で作成する。
完了時に学習済みモデルの絶対パスを1行で書き出す。
[評価セル](evaluation.md)で読み込めるため、モデルパスのコピーは不要。
次の学習開始時には以前のパスを空にし、完了したときだけ新しいモデルパスに更新する。
同じディレクトリでの学習は順番に実行する。
`model-path.txt`は実行ディレクトリに置くため、Colabランタイムを破棄すると消える。

## 切断後の再開

Colabでclone・環境準備・Drive接続をやり直し、元と同じ`--output`に`--resume`を付ける。
最新run内の、再開用ファイルが揃った最後のチェックポイントを選ぶ。
保存後に進んだ計算はやり直す。

```python
!uv run --locked train \
    --wikipedia-count 7500 \
    --corpus-count 7500 \
    --output /content/drive/MyDrive/generic-zenzai-202611 \
    --resume
```

最新runに有効なチェックポイントがない場合や、完了済みの場合はエラーになる。
古いrunや新規学習へ自動では切り替えない。
件数を変更して学習したrunを再開するときは、元と同じ件数オプションを再指定する。
また、`run.json`に記録したモデル・データ・設定が元と異なる場合も再開できない。
旧設定（バッチ2件・勾配累積16回、8件・4回、16件・2回）のrunは、現在の設定（32件・1回）では再開できない。
新設定を試す場合は`--resume`なしで新しいrunを開始する。

特定のチェックポイントを指定する場合は、`--resume`の代わりに`--resume-from`を使う。
`--output`には、そのrunまたはrunの親ディレクトリを指定する。
次の`run-実際の日時/checkpoint-N`を実在する名前に置き換える。

```python
checkpoint = "/content/drive/MyDrive/generic-zenzai-202611/run-実際の日時/checkpoint-N"
!uv run --locked train \
    --wikipedia-count 7500 \
    --corpus-count 7500 \
    --output /content/drive/MyDrive/generic-zenzai-202611 \
    --resume-from "{checkpoint}"
```

チェックポイントを保存する前に失敗した場合は、原因を解消して`--resume`なしでやり直す。
失敗したrunは残り、新しいrunで最初から学習する。

## 学習データの出典と制限

元データは[Miwa-Keita/zenz-v2.5-dataset](https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset/tree/7d9c9ea36347b638a627c1d264da48ebc7eb38aa)の
revision `7d9c9ea36347b638a627c1d264da48ebc7eb38aa`をDriveに取得する。
学習用ファイルはリポジトリに同梱しない。生成した`train.jsonl`のSHA-256は`run.json`に記録する。
元データ全体からの一様な標本ではなく、読みや意味の人手検証もしていない。

元データの作成者はMiwa-Keita。Wikipedia由来の`train_wikipedia.jsonl`には
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)が適用され、
2024年2月取得の日本語Wikipediaアーカイブに由来する。
`train_llm-jp-corpus-v3.jsonl`は[llm-jp-corpus-v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)の
Common Crawl由来データに基づき、[ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/)と
[Common Crawlの利用条件](https://commoncrawl.org/terms-of-use)が適用される。
詳しい謝辞と品質上の注意は[元データのデータカード](https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset/raw/7d9c9ea36347b638a627c1d264da48ebc7eb38aa/README.md)を参照。
