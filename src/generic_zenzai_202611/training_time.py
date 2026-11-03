"""Accumulate active training time along the saved checkpoint history."""

from time import perf_counter

from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState


class TrainingTimeCallback(TrainerCallback, ExportableState):
    def __init__(self, elapsed=0.0):
        self.elapsed = elapsed
        self.started_at = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = perf_counter()

    def state(self):
        elapsed = self.elapsed
        if self.started_at is not None:
            elapsed += perf_counter() - self.started_at
        # Persist a duration, never the process-local clock origin. Trainer calls
        # this before writing trainer_state.json; on_save would be too late.
        return {"args": {"elapsed": elapsed}, "attributes": {}}
