# 学習データの出典

`train.jsonl`は、既存のZenzai実験用データからコピーした5,000件。
`input`・`output`・`left_context`の値は変更していない。

- 元データ: [Miwa-Keita/zenz-v2.5-dataset][dataset]
- revision: `7d9c9ea36347b638a627c1d264da48ebc7eb38aa`
- ソース: `train_llm-jp-corpus-v3.jsonl`から2,804件、
  `train_wikipedia.jsonl`から2,196件。
- 文脈: `null`が2,500件、文字列が2,500件。
- SHA-256: `f689d71af23e2613ebdff82740dc27a2ec784b4608be5d31eea7dd40b44f7782`

## ライセンスと帰属

元のデータカードに従い、Wikipedia由来の部分には
[CC BY-SA 4.0][cc]が適用される。
2024年2月取得の日本語Wikipediaアーカイブに由来する。
Common Crawl由来の部分には[ODC-BY 1.0][odc]および
[Common Crawlの利用条件][crawl]が適用される。
こちらは[llm-jp-corpus-v3][corpus]の
`ja/ja_cc/level0/CC-MAIN-2023-40.jsonl.gz`に由来する。

データセット作成者はMiwa-Keita。
データカードはllm-jp-corpus-v3、MeCab、ipadic-NEologd、
jaconv、日本語Wikipediaアーカイブを利用元として挙げ、
IPAの2024年度未踏事業による支援を記載している。
この帰属と出典別の条件を維持する。

## 抽出方法と制限

各ソースの先頭250 MiBだけを取得し、同じ出力文字列から1件を選んだ。
出力文字列のグループごとに学習・検証への割り当てを決めた後、
seed `20261101`で順序を決め、文脈の有無を半数ずつに調整している。
ここに同梱するのは学習分だけであり、元コーパス全体の一様標本ではない。

元のデータカードは読みの不正確さに注意を促している。
構造の確認だけを行っており、読みや意味の人手検証・修正はしていない。
抽出元のハッシュ、各行のソース行番号・バイト位置は
[manifest.json](manifest.json)に記録している。

[dataset]: https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset/tree/7d9c9ea36347b638a627c1d264da48ebc7eb38aa
[cc]: https://creativecommons.org/licenses/by-sa/4.0/deed.ja
[odc]: https://opendatacommons.org/licenses/by/1-0/
[crawl]: https://commoncrawl.org/terms-of-use
[corpus]: https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3
