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

## 学習速度・VRAMのベンチマーク

学習の基本設定は変更していない。速度を比較するため、次のCLIフラグを追加した。

| フラグ | 既定値 | 意味 |
| --- | --- | --- |
| `--batch-size` | 32 | GPU上の1回のミニバッチ件数 |
| `--gradient-accumulation-steps` | 1 | 勾配累積回数 |
| `--train-sampling-strategy` | `random` | `group_by_length`で系列長の近い例をまとめ、パディングを削減 |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | 有効 | 活性の再計算とVRAM使用量のトレードオフ |
| `--optim` | `adamw_torch` | `adamw_torch_fused`でCUDA用fused AdamWを試す |
| `--save-steps` | 50 | チェックポイント間隔。Drive書き込み頻度に影響 |

**先に現在の既定設定でベースラインを測定する。** まず同じGPU、モデル、学習例、epoch数で学習する。学習結果の`train_results.json`にある`train_runtime`、`train_samples_per_second`、`peak_cuda_memory_gib`を記録する。新規実験は新しいrunに保存し、既存runを変更しない。

例：ColabのT4で学習が始められる場合、次の設定を別々の新規runとして試す。いずれも**学習データ・周回数・学習率・実効バッチ32件は維持**する。短い計測では初回の環境構築やモデル取得時間を含めない。

```bash
# 1: ベースライン
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --output /content/drive/MyDrive/generic-zenzai-202611

# 2: パディング削減
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --train-sampling-strategy group_by_length \
  --output /content/drive/MyDrive/generic-zenzai-202611

# 3: パディング削減 + optimizerのfused kernel
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --train-sampling-strategy group_by_length --optim adamw_torch_fused \
  --output /content/drive/MyDrive/generic-zenzai-202611

# 4: VRAMに収まればcheckpointingを無効化（T4ではOOMの場合あり）
uv run --locked train --wikipedia-count 2500 --corpus-count 2500 --epochs 1 \
  --train-sampling-strategy group_by_length --optim adamw_torch_fused \
  --no-gradient-checkpointing \
  --output /content/drive/MyDrive/generic-zenzai-202611
```

`--no-gradient-checkpointing`でメモリ不足になる場合は、`--batch-size 16 --gradient-accumulation-steps 2`または`--batch-size 8 --gradient-accumulation-steps 4`を試す。ただしミニバッチサイズ変更やデータ順序変更は、数値精度・更新方向・学習結果を厳密には同一にしない。AJIMEEの同一条件で精度を再評価する。

チェックポイント保存に時間がかかる場合は`--save-steps 200`なども試せる。ただし停止時にやり直す更新数が増える。学習時間とColab切断リスクの双方で判断する。

学習データの元ファイルはDriveから先頭部分を読み込む。run自体を`/content`に作り、終了後にDriveへ移すと保存I/Oを切り分けられるが、中断時のチェックポイントはColab側にしか残らないため、移動前のランタイム切断には注意する。

開始時に計算精度を表示する。T4では`計算精度: fp16`になる。
最初の更新時と以降10更新ごとに、`step 現在/全体 | loss: … | …`の形式で指標を表示する。
終了時の学習損失・速度なども同じ形式で表示する。
学習中の`loss`は予測誤差であり、変換精度ではない。
学習完了後、別コマンドで[AJIMEE評価](evaluation.md)を実行する。

既定モデルのrevisionは固定し、使用した値を`run.json`に記録する。
モデルの出典は[LLM-jpのモデルカード](https://huggingface.co/llm-jp/llm-jp-3-150m)、
学習データの出典・ライセンス・品質上の制限は[以下の説明](#学習データの出典と制限)を参照。
その他のCLIオプションは`uv run --locked train --help`で確認できる。

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
