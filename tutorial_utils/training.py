"""Early-stopping and checkpoint plumbing for the SGD training loop.

The visible loop body (`zero_grad / backward / step`) stays in the notebook;
this module hides the best-state caching and torch.save/load wrangling.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch


class EarlyStopper:
    """Track the best loss seen so far and signal when patience is exceeded.

    Caches `best_state` via `copy.deepcopy(model.state_dict())` whenever a new
    best is reached. Call `update(loss, step)` once per step; returns True
    when the run should stop.
    """

    def __init__(self, patience, min_delta, model):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_step = -1
        self.best_state = copy.deepcopy(model.state_dict())

    def update_best(self, loss, step, model):
        """Cache `model.state_dict()` if `loss` is a new best. Call before backward."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = step
            self.best_state = copy.deepcopy(model.state_dict())

    def should_stop(self, step):
        """True when no improvement for `patience` steps. Call after optimizer.step()."""
        return step - self.best_step >= self.patience

    def restore_best(self, model):
        model.load_state_dict(self.best_state)


def try_load_checkpoint(model, path):
    """Load a saved checkpoint into `model` if the file exists.

    Returns the saved `training_run` dict (or `lstm_run`, whichever was saved)
    or None if the file does not exist. The state_dict is loaded into `model`.
    """
    path = Path(path)
    if not path.exists():
        return None
    saved = torch.load(path, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    return saved


def save_checkpoint(model, path, **meta):
    payload = {"state_dict": model.state_dict()}
    payload.update(meta)
    torch.save(payload, Path(path))
