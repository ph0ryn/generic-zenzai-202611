# generic-zenzai-202611

LLM-jp-3-150mを、文脈と読みからかな漢字変換するモデルに追加学習する。
uvで環境を固定し、ColabからCLIで学習とAJIMEE-Bench評価を実行する。

## 学習

Colabで新規ノートブックを作る。
ランタイムをGPUに変更し、以下の3ブロックを別々のセルにコピーして、
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
まず学習と保存を完了し、その後に下記のコマンドでAJIMEE評価を実行する。
評価は学習と別プロセスなので、評価の失敗で学習をやり直す必要はない。

モデルのrevisionは
`b112feef602fff752e4dac4c30af6a2c2fa41c7a`に固定。
出典は[LLM-jpのモデルカード](https://huggingface.co/llm-jp/llm-jp-3-150m)。
データの出典・ライセンス・品質上の制限は[data/README.md](data/README.md)を参照。

## AJIMEE評価

学習の最後に表示された`学習済みモデル:`のパスを、次のセルの`model`に指定する。
`run-実際の日時`は実際のディレクトリ名に置き換える。同じGPUランタイムで実行できる。
モデルやtokenizerはローカル保存済みのものだけを読み、評価中のダウンロードは行わない。

```python
model = "/content/drive/MyDrive/generic-zenzai-202611/run-実際の日時/model"
!uv run --locked generic-zenzai-evaluate --model "{model}"
```

既存のcloneでsubmoduleが未初期化なら、先に次を実行する。
親repoの更新後も、このコマンドで記録されたcommitに合わせる。`--remote`は付けない。

```python
!git submodule update --init
```

### 評価条件と指標

[AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench)の`JWTD_v2/v1`を使う。
評価データはこのrepoへコピーせず、[submodule](<benchmarks/AJIMEE-Bench>)として参照する。
初期の参照commitは`401666cd56d1a570c2021798b64b6da4396bfd45`。
Gitが参照commitを固定し、更新はsubmoduleの参照変更として管理する。
未初期化や評価データの未commit変更はエラー終了する。

- 全200件を、文脈あり100件・なし100件に分けても集計する。
- 学習と同じプロンプトを使い、左文脈は末尾40文字。読みは分割せず全文を入力する。
- 1件ずつ、fp32・greedy（サンプリングなし、beam数1）で生成する。
  生成上限は128トークン。EOSで終了しなかった件数も記録する。
- 入力＋生成上限が512トークンまたはモデルの位置上限を超える場合は停止する。
  データを除外したり、読みを切り詰めたりはしない。`--max-new-tokens`で上限を変更できる。
- 完全一致率は、生成結果が許容解のいずれかと一致した割合。
  MinCERは各許容解への文字誤り率の最小値で、集計値はその項目ごとの単純平均。
  空白除去や表記の正規化は行わない。生成の特殊トークンのみ除去する。

これは**モデル単体のかな漢字変換評価**であり、Zenzaiの辞書・探索・読み制約・量子化を
含むIME全体の評価ではない。ベースモデルを比較するときは、学習データ・文脈条件・
生成上限を揃える。速度の実機ベンチマークは含まない。

### 評価結果

既定では指定モデルの隣の`ajimee/eval-日時/`に保存する。Colabの例ではDrive上になる。
`--output`で保存先の親ディレクトリを変更でき、毎回新規ディレクトリを作る。

- `predictions.jsonl`: 各項目のID・読み・文脈・許容解・予測・採点・生成上限到達の有無。
  1件ごとに保存する。
- `scores.json`: 全体・文脈あり・文脈なしの件数、完全一致率、平均MinCER。
  全件成功時のみ保存し、このファイルが評価完了の目印になる。
- `evaluation.json`: モデルのパス、実際のベンチcommitとデータのSHA-256、
  プロンプト・生成条件・デバイス・ライブラリバージョン。

評価途中で切断した場合も途中結果は残る。評価の再開機能はなく、同じコマンドで全件を
新しいディレクトリへ再評価する。入力データ本体のコピーは保存しない。

ローカルでも同じCLIを使える。GPUなしなら`--cpu`を指定する。
別のAJIMEE checkoutを使う場合は`--benchmark /path/to/AJIMEE-Bench`を指定する。

評価データの出典はazooKeyのAJIMEE-Benchと、京都大学の
[日本語Wikipedia入力誤りデータセット(v2)](https://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット)。
データはCC BY-SA 3.0、公式の採点コードはCC0 1.0 Universal。
詳細は[上流の出典・ライセンス説明](<benchmarks/AJIMEE-Bench/README.md>)を参照。
許容解などを含む評価結果を再配布するときも、データの出典・ライセンスを維持する。

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
評価CLIも、小さなモデルと合成した評価データで実際の生成・採点・保存・異常終了を確認する。
上流のデータやモデルのダウンロードはテストに不要。
通常の学習コマンドにはCUDA GPUが必要。学習の`--cpu`は小規模な動作確認用。

処理は次の責務で分割している。

- [train.py](<src/generic_zenzai_202611/train.py>): 学習と再開。
- [prompting.py](<src/generic_zenzai_202611/prompting.py>): 学習・評価共通のプロンプト。
- [inference.py](<src/generic_zenzai_202611/inference.py>): 保存済みモデルの読み込みと生成。
- [benchmark.py](<src/generic_zenzai_202611/benchmark.py>): ベンチデータの検証と採点。
- [evaluate.py](<src/generic_zenzai_202611/evaluate.py>): 評価CLIと結果保存。
