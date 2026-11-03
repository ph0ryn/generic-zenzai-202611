# 次のモデル比較とColab T4での実行

調査日: 2026年11月3日。対象GPUはT4（16GB）。

## 推薦

次は **LiquidAI/LFM2.5-350M-Base**、次点は **Qwen/Qwen3-0.6B-Base**。
同じ135,000件を3周学習し、LLM-jp-3-150mの完全一致44.50%・平均MinCER 0.0984と比較する。
1周・3周の比較を各モデルで繰り返すことは、次の実験の必須条件にしない。

LFM2.5のBaseは日本語を含む9言語に対応し、公式カードが言語特化の追加学習を用途に挙げている。
総パラメータ数が約354Mなので、全パラメータ学習を維持してT4で試す候補として選ぶ。
この用途の記述は、かな漢字変換での優位性が実証された根拠ではない。
構造・モデル容量・事前学習・トークナイザが同時に変わるため、改善要因の分離はできない。

## 候補の比較

パラメータ数は2026年11月3日のHugging Face APIのSafetensors情報で確認した総数。
モデル名の数値とは区別する。

| 候補・公式資料 | 総パラメータ数 | 判断 |
| --- | ---: | --- |
| [LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base) | 354,483,968 | 第一候補。追加学習向けBase、日本語対応。畳み込みとAttentionの混合構造 |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 596,049,920 | 次点。119言語で事前学習されたBase。通常のTransformerで既存コードとの互換性を確認 |
| [Sarashina2.2-0.5B](https://huggingface.co/sbintuitions/sarashina2.2-0.5b) | 793,048,320 | 日本語中心の比較に有力。ただし「0.5B」は埋め込みとLM headを除いた数。T4では余裕が小さい |
| [Granite-4.0-350M-Base](https://huggingface.co/ibm-granite/granite-4.0-350m-base) | 352,379,904 | 日本語を含む12言語対応のBase。350M版はdense Transformer。予備候補 |
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | 354,483,968 | こちらはpost-trained版。今回は追加学習用のLFM2.5 Baseを優先 |

モデルカード上のライセンスは、LFM系列がLFM Open License、Qwen・GraniteがApache-2.0、
SarashinaがMIT。日本語QAなどの一般指標を、かな漢字変換の成績の代わりには使わない。

## T4でのバッチ設定

従来のLLM-jpでは約3/16GBしか使っていなかったというユーザーの実測を踏まえ、
最初は**バッチ32・勾配累積1**を維持する。
新モデルのVRAM使用量は未測定なので、同じ3GBで済むとは仮定しない。

現在の実装はFP32の重みを保持し、T4ではFP16 autocastで学習する。
重み・勾配・AdamWの2種類の状態を各4 bytesとして見積もると、
常駐部分だけでLFM2.5は約5.28GiB、Qwenは約8.88GiB、Sarashinaは約11.82GiBになる。
これは`総パラメータ数 × 16 / 2^30`による概算で、
活性値、logits、混合精度の一時コピー、optimizerの一時領域、CUDAの予約領域を含まない。
T4に収まるという実測保証ではない。

VRAMが足りなければ、`--batch-size 16 --gradient-accumulation 2`、
さらに必要なら`8`と`4`にして実効バッチ32件を維持する。
累積を増やすだけで精度が上がるわけではなく、実バッチを小さくすると遅くなることがある。
この変更は新規runで行う。再開時にバッチ構成を変更すると設定不一致で停止する。
根拠: [Transformersの勾配累積](https://huggingface.co/docs/transformers/main/en/grad_accumulation)、
[混合精度](https://huggingface.co/docs/transformers/main/en/mixed_precision_training)。

## LFM2.5の実行セル

[README](../README.md#usage)のclone・環境構築・Drive接続後、準備・学習セルを以下に置き換える。
これらのオプションを追加したコードがColabのcheckoutへ反映されていることが必要。

```python
!uv run --locked prepare \
    --model LiquidAI/LFM2.5-350M-Base \
    --revision 9960764e30892e01f29a6dc23df2533fcd8bd5ae \
    --model-dir /content/drive/MyDrive/models/LFM2.5-350M-Base
```

```python
!uv run --locked train \
    --model LiquidAI/LFM2.5-350M-Base \
    --revision 9960764e30892e01f29a6dc23df2533fcd8bd5ae \
    --model-dir /content/drive/MyDrive/models/LFM2.5-350M-Base \
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

学習率5e-5、文脈40文字、上限512トークン、seed 42、全パラメータ学習は既存設定を使う。
評価は同じAJIMEE 200件・greedy・生成上限64トークン・FP32。
モデルごとに出力親ディレクトリを分け、別モデルのrunへの誤った再開を避ける。
再開時は同じ学習コマンドに`--resume`を付ける。

データを再生成したら`run.json`の`data_sha256`が前回の
`ef51b5c08775961802764cf7c42da7f4debc426bc5fdeaca3925f954781cef57`と一致することを確認する。
同じ件数でも入力データの内容が変われば同条件比較にはならない。

Qwenを試す場合は、準備・学習の両セルを次の値に置き換える。

| オプション | Qwen用の値 |
| --- | --- |
| `--model` | `Qwen/Qwen3-0.6B-Base` |
| `--revision` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| `--model-dir` | `/content/drive/MyDrive/models/Qwen3-0.6B-Base` |
| `--output`（学習のみ） | `/content/drive/MyDrive/generic-zenzai-202611/qwen3-0.6b` |

固定revisionは[モデルの履歴](https://huggingface.co/LiquidAI/LFM2.5-350M-Base/commits/main)と
Hugging Face APIから取得したもの。準備完了印と学習時のrevisionが違えば停止する。
既定のLLM-jpの準備・学習コマンドも引き続き使用できる。

## 確認済みの範囲

- LFM2.5・Qwenの実トークナイザで今回の学習135,000件を確認。
  入力＋正解＋EOSは最大206・182トークンで、512超過はともに0件。
- 同じトークナイザでAJIMEEの入力長を確認。入力＋生成上限64は最大191・184トークン。
  学習前の性能評価や生成結果による採点は行っていない。
- Qwen3・LFM2の縮小モデルで、CLIからの勾配累積学習、重み更新、設定保存、
  Converterによる再読み込み・生成を自動テスト。
- LFM2.5-350M-Baseの公開重みそのものでも、CPU・FP32・4例・2更新で有限のloss、
  重みの更新、保存、再読み込み・生成を確認した。確認用入力ではEOSだけを出力した。
  これは互換性の確認であり、かな漢字変換の性能測定ではない。
- Sarashinaは現行依存関係ではトークナイザ読み込み時にSentencePiece不足で停止した。
  今回はLFM2.5とQwenを対象にし、この追加依存は導入していない。

LFM2では`causal_conv1d`が未導入のため、Transformersが明示的に警告し、
標準PyTorch実装を使用する。CPUでの学習は通ったが、専用カーネル使用時より遅くなり得る。
この実験では既存依存関係で動かし、速度の問題が出た場合に別途検討する。
公式のTRL fine-tuning例はLoRAを用いており、今回の全パラメータ学習とは条件が異なる。

T4上でのFP16学習、ピークVRAM、所要時間、135,000件×3周後の精度は未確認。
