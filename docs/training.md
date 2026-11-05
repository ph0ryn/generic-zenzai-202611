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
| モデル | `LiquidAI/LFM2.5-350M-Base`、全パラメータを更新 |
| 周回数 | 1 epoch（`--epochs`で変更） |
| 学習率 | `5e-5`。最初の5%で増やし、その後は線形に下げる |
| バッチ | 32件 × 勾配累積1回（累積なし）。実効32件 |
| 長さ上限 | 入力・正解・終了記号を合わせて512トークン。超過時は切らずに停止 |
| 計算精度 | bf16ハードウェア対応GPUではbf16、それ以外のCUDA GPUではfp16 |
| optimizer | 新規CUDA学習はfused AdamW、CPUは通常のAdamW。再開時は保存済みの方式を維持 |
| 勾配チェックポイント | 有効。`--no-gradient-checkpointing`で再計算を省く代わりにGPUメモリ使用量が増える |
| チェックポイント | 1,000更新ごと（`--save-steps`で変更）。学習中は最新1個を保持し、正常完了後に削除 |

バッチ件数は`--batch-size`、勾配累積は`--gradient-accumulation`で変更できる。
どちらも正の整数が必要で、積がGPU 1台での実効バッチ件数になる。
モデルID・固定revision・保存先は[共通設定](../src/generic_zenzai_202611/config.py)を使い、
準備・学習CLIにはモデルを切り替える引数を設けない。
候補と具体的なセルは[モデル比較](models.md)を参照。

開始時に計算精度を表示する。T4では`計算精度: fp16`になる。
最初の更新時と以降10更新ごとに、`step 現在/全体 | loss: … | …`の形式で指標を表示する。
終了時の学習損失・速度なども同じ形式で表示し、累計学習時間を別の行に表示する。
累計学習時間（`train_runtime_total`）は、再開元チェックポイントまでの学習時間と
今回の学習実行時間（`train_runtime`）の合計。サマリーには両方を表示する。
中断中の待ち時間、データ・モデルの準備、終了後の最終モデル保存は含めない。
チェックポイント保存後に進み、中断によって失われた計算時間も含めない。
学習中の`loss`は予測誤差であり、変換精度ではない。
学習完了後、別コマンドで[AJIMEE評価](evaluation.md)を実行する。

`uv run --locked task prepare`で元データとベースモデルをDriveへ事前保存する。
保存済みのものは再取得せず、学習ではDrive上のベースモデルを読み込む。
学習時には既定モデルの取得・保存を行わない。未準備の場合はエラーになり、先に準備セルを実行する。
モデル・データセットの共有設定は[config.py](../src/generic_zenzai_202611/config.py)で管理する。
既定モデルのrevisionは固定し、Drive上のパスではなくモデルIDとrevisionを`run.json`に記録する。
モデルの出典は[LFM2.5のモデルカード](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)、
学習データの出典・ライセンス・品質上の制限は[以下の説明](#学習データの出典と制限)を参照。
その他のCLIオプションは`uv run --locked train --help`で確認できる。

## T4向けの実行設定

新規CUDA学習では`adamw_torch_fused`を使う。同じAdamW更新をまとめて実行し、
optimizerのカーネル起動や中間メモリの負担を減らすための変更である。
モデルの保存パラメータはfloat32のまま、T4の演算にはFP16とGradScalerを使う。
学習データ・正解部分のマスク・実効バッチ32件・学習率・更新回数は変えない。
実際のT4での高速化率とCCU削減量は未計測。

`--optimizer adamw_torch`で従来の実装、`--optimizer adamw_torch_fused`でfusedを指定できる。
既定の`--optimizer auto`は新規CUDA学習ならfused、CPUなら通常のAdamWを選ぶ。
再開時は`optimizer.pt`に保存された方式を読み取り、その方式を維持する。
旧runを自動でfusedに変換したり、optimizerの状態を初期化したりはしない。
再開元と異なる方式を明示するとエラーになる。方式を比較するときは新規runを使う。

勾配チェックポイントは、活性値の保持を減らす代わりに計算をやり直す仕組みである。
`--no-gradient-checkpointing`を付けると再計算を省けるが、GPUメモリ使用量は増える。
バッチ32件のままT4に収まるかは未計測のため、既定では有効のままとする。
短い試走が成功しても、後半の長いバッチや上限512トークンで収まる保証はない。
メモリ不足時にバッチや精度を自動変更することはない。
保存済みチェックポイントがあれば、勾配チェックポイントを有効に戻して再開できる。

### 短い比較実行

[Colabの環境・データ・モデルの準備](../README.md#usage)後、同じT4ランタイムで次を実行する。
各条件で同じ135,000件から同じ順序の約211更新を行い、途中の定期保存が発生しないよう保存間隔を大きくする。
どの条件も元モデルから始める。通常の3周学習やAJIMEE評価は、この速度比較には不要。
実際の学習件数が異なる場合は、両ソースの件数を全条件で同じ値に変更する。

```sh
%%bash
set -euo pipefail
for variant in baseline fused no-recompute; do
    optimizer=adamw_torch_fused
    checkpointing=--gradient-checkpointing
    case "$variant" in
        baseline) optimizer=adamw_torch ;;
        no-recompute) checkpointing=--no-gradient-checkpointing ;;
    esac
    time uv run --locked train \
        --wikipedia-count 67500 \
        --corpus-count 67500 \
        --epochs 0.05 \
        --batch-size 32 \
        --gradient-accumulation 1 \
        --save-steps 1000000 \
        --optimizer "$optimizer" \
        "$checkpointing" \
        --output "/content/t4-benchmark-$variant"
done
```

- `baseline`: 従来のAdamW＋勾配チェックポイント有効。
- `fused`: fused AdamW＋勾配チェックポイント有効。新規GPU学習の既定値。
- `no-recompute`: fused AdamW＋勾配チェックポイント無効。メモリが足りる場合の候補。

比較には各runの`train_results.json`にある`train_runtime`、`train_steps_per_second`、
`train_peak_memory_allocated`、`train_peak_memory_reserved`を使う。
メモリ値はバイト単位で、optimizer初回更新を含む今回の学習区間のPyTorch allocatorのピーク。
CUDAコンテキストや拡張がPyTorch外で確保した分を含むGPU全体の使用量ではない。
`device_name`、`optimizer`、`gradient_checkpointing`も保存する。
optimizerと勾配チェックポイントの設定は最終モデルの`training_args.bin`にも残る。

保存間隔を大きくしても、Transformersは学習終端でチェックポイントを保存する。
この保存時間は`train_runtime`と`train_steps_per_second`に含まれる。
純粋な計算カーネルの速度ではなく、終端保存を含む短い試走の比較として扱う。
学習完了後の最終モデル保存は`train_runtime`に含まれない。
同じT4・同じ更新数なら、学習区間のCCUは`CCU/時 × train_runtime / 3600`で比較できる。
`time`の`real`にはデータ準備・最終保存も含まれるため、全体の所要時間も併せて見る。
初期ウォームアップを含む短い試走なので、差が小さければ実行順を変えて再確認する。
再開前後を混ぜた`train_runtime_total`は、この比較には使わない。
エラーが出た場合はそこで停止し、原因を確認する。

比較用モデルはColabの一時ディスクに保存する。速度確認用であり、性能評価用のモデルではない。
比較実行で`model-path.txt`も更新されるため、終了後は通常の学習セルで本学習を開始し、
その学習が完了してから評価セルを実行する。

## 高速畳み込みカーネル

Linuxでは`uv sync --locked`により、`causal-conv1d`のビルド済みCUDA wheelを導入する。
[依存設定](../pyproject.toml)で、現在のPython 3.14・PyTorch 2.14.1・CUDA 12.6に対応する
[Astralの配布物](https://github.com/astral-sh-build/build-causal-conv1d#supported-versions)を指定している。
Colabでのソースビルドや、別途の`pip install`は不要。macOSにはこのCUDA依存を導入しない。
Transformersは導入済みのカーネルを自動で使う。警告を非表示にする変更ではない。
この拡張を導入したLinux環境でのLFM2実行はCUDA専用になる。
`--cpu`を使う場合は、拡張を除外する[ローカル開発の手順](development.md)で環境を作る。
実際の学習全体の高速化率は未計測。

Colabでは、以下を独立したセルで実行すると、拡張の読み込みとFP16のCUDA順伝播・逆伝播を確認できる。
失敗した場合は学習を再開せず、表示されたエラーを確認する。

```sh
%%bash
uv run --locked python - <<'PY'
import torch
import torch.nn.functional as F
import causal_conv1d_cuda
from causal_conv1d import causal_conv1d_fn

torch.manual_seed(0)
x = torch.randn(2, 16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
w = torch.randn(16, 3, device="cuda", requires_grad=True)
y = causal_conv1d_fn(x, w)
reference = F.conv1d(x.float(), w[:, None], padding=2, groups=16)[..., :32].half()
torch.testing.assert_close(y, reference, rtol=2e-3, atol=2e-3)
y.float().square().mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert w.grad is not None and torch.isfinite(w.grad).all()
print("causal-conv1d CUDA forward/backward: OK")
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
```

### 進行中の学習へ適用する

再開可能なチェックポイントの保存完了を確認してから、学習セルを停止する。
Colabのリポジトリ直下で次のセルを実行し、上記のカーネル動作確認を行う。

```sh
%%bash
set -euo pipefail
git pull --ff-only
uv sync --locked
```

その後、元の学習コマンドに`--resume`を付けて再実行する。
モデル・データ件数の内訳・周回数・バッチ・勾配累積・出力先は元と同じにする。
今回変えるのは計算カーネルだけで、チェックポイントや`run.json`の形式は変えない。
最後のチェックポイントより後の未保存分は再計算する。
再開時に`causal_conv1d_fn`の参照実装へのフォールバック警告が出る場合は、
カーネルが使われていないため、そのまま長時間学習を続けず環境を確認する。

## 保存先

`--output`の下に毎回新しい`run-日時`を作り、開始時にパスを表示する。
通常実行では既存のrunを上書きしない。

- `model/`: 学習済みモデルとtokenizer。`from_pretrained`で読み込める。
- `checkpoint-N/`: optimizer・学習進捗などを含む再開用の状態。学習中・中断時のみ保持し、正常完了後は削除する。
- `run.json`: モデルrevision、データのSHA-256、学習設定。
- `train_results.json`: 完了時の学習損失・速度、累計と今回の学習時間、更新回数、計算精度、実効バッチなど。

最終モデル・tokenizer・学習結果・モデルパスの保存がすべて成功した後に、
そのrun直下の`checkpoint-数字/`だけを削除する。
学習の中断や保存の失敗時には削除せず、再開用の状態を残す。
最終モデル、学習データ・設定・指標、AJIMEE評価結果は残し、他のrunには触れない。
過去に完了したrunのチェックポイントは、この変更では一括削除しない。

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
    --wikipedia-count 67500 \
    --corpus-count 67500 \
    --epochs 3 \
    --output /content/drive/MyDrive/generic-zenzai-202611/lfm2.5-350m \
    --resume
```

最新runに有効なチェックポイントがない場合や、完了済みの場合はエラーになる。
古いrunや新規学習へ自動では切り替えない。
件数を変更して学習したrunを再開するときは、元と同じ件数オプションを再指定する。
また、`run.json`に記録したモデル・データ・設定が元と異なる場合も再開できない。
旧設定（バッチ2件・勾配累積16回、8件・4回、16件・2回）のrunは、既定設定（32件・1回）では再開できない。
再開するrunと同じ`--batch-size`・`--gradient-accumulation`を指定する。
新設定を試す場合は`--resume`なしで新しいrunを開始する。

特定のチェックポイントを指定する場合は、`--resume`の代わりに`--resume-from`を使う。
`--output`には、そのrunまたはrunの親ディレクトリを指定する。
次の`run-実際の日時/checkpoint-N`を実在する名前に置き換える。

```python
checkpoint = "/content/drive/MyDrive/generic-zenzai-202611/lfm2.5-350m/run-実際の日時/checkpoint-N"
!uv run --locked train \
    --wikipedia-count 67500 \
    --corpus-count 67500 \
    --epochs 3 \
    --output /content/drive/MyDrive/generic-zenzai-202611/lfm2.5-350m \
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
