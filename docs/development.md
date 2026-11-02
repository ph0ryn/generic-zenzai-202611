# ローカル開発

プロジェクトのルートで実行する。

```sh
uv sync --locked
uv run task lint
uv run task test
```

テストは小型モデルと合成データを使い、CPUで学習・保存・再開・評価を検証する。
外部モデルや評価データのダウンロードは不要。

通常の学習にはCUDA GPUが必要。学習CLIの`--cpu`は小規模な動作確認用であり、
同梱データでの学習をCPUで行うための推奨設定ではない。
評価CLIはCUDA GPUまたは`--cpu`で実行できる（MPSには未対応）。
