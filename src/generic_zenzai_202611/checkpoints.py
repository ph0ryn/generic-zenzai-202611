"""Training run creation and resumable checkpoint validation."""

import json
import shutil
from datetime import datetime


def resolve_run(output, resume=False, resume_from=None):
    if resume_from is not None:
        run = resume_from.parent
        if output.resolve() not in (run.resolve(), run.parent.resolve()):
            raise ValueError("resume-fromにはoutput内のcheckpointを指定してください。")
        if not valid_checkpoint(resume_from):
            raise ValueError("有効な再開用チェックポイントが見つかりません。")
        return run, resume_from
    if resume:
        runs = sorted(path for path in output.glob("run-*") if path.is_dir())
        if not runs:
            raise ValueError("再開するrunが見つかりません。")
        run = runs[-1]
        if (run / "train_results.json").exists():
            raise ValueError(f"最新のrunは完了済みです: {run}")
        checkpoints = [
            path
            for path in run.glob("checkpoint-*")
            if path.name.removeprefix("checkpoint-").isdigit()
            and valid_checkpoint(path)
        ]
        if not checkpoints:
            raise ValueError(f"最新のrunに有効なチェックポイントがありません: {run}")
        return run, max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    run = output / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run.mkdir(parents=True, exist_ok=False)
    return run, None


def remove_checkpoints(run):
    """Remove only this completed run's direct, non-symlink checkpoint directories."""
    run = run.resolve()
    for path in run.glob("checkpoint-*"):
        if (
            path.name.removeprefix("checkpoint-").isdigit()
            and not path.is_symlink()
            and path.is_dir()
            and path.resolve().parent == run
        ):
            shutil.rmtree(path)


def valid_checkpoint(path):
    required = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "config.json")
    if not all((path / name).is_file() for name in required):
        return False
    weights = ("model.safetensors", "pytorch_model.bin")
    indexes = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    try:
        json.loads((path / "trainer_state.json").read_text())
        if any((path / name).is_file() for name in weights):
            return True
        for name in indexes:
            if (path / name).is_file():
                shards = json.loads((path / name).read_text())["weight_map"].values()
                return bool(shards) and all(
                    (path / shard).is_file() for shard in shards
                )
    except OSError, ValueError, KeyError, TypeError:
        return False
    return False
