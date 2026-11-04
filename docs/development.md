# ローカル開発

プロジェクトのルートで実行する。
CPUで検証するため、LinuxでもCUDA専用の`causal-conv1d`を除外し、実行時の再同期を行わない。
ColabのGPU学習では除外せず、[READMEの準備手順](../README.md#usage)を使う。

```sh
uv sync --locked --no-install-package causal-conv1d
uv run --no-sync task lint
uv run --no-sync task test
```

テストは小型モデルと合成データを使い、CPUで学習・保存・再開・評価を検証する。
外部モデルや評価データのダウンロードは不要。

通常の学習にはCUDA GPUが必要。学習CLIの`--cpu`は小規模な動作確認用であり、
Driveから生成した学習データでCPU学習を行うための推奨設定ではない。
評価CLIはCUDA GPUまたは`--cpu`で実行できる（MPSには未対応）。
