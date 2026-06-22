"""Pure-ML LSTM baseline: feature prep, model class, training loop.

This is the reference model that has *no physics in the loop*. We pull all
the boilerplate out of the notebook because it's not the educational focus —
the hybrid model is. The LSTM exists only to demonstrate, by contrast, what
the inductive bias of the WOFOST engine buys you.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffwofost.physical_models.config import ComputeConfig


LSTM_SEQ_LEN = 200
LSTM_HIDDEN = 32
DAYS_NORMALIZER = 200.0

LSTM_VAL_FRACTION = 0.2
LSTM_VAL_SEED = 1234
LSTM_MAX_STEPS = 1500
LSTM_PATIENCE = 100
LSTM_MIN_DELTA = 1e-3

PURE_OBS_VARS = ["LeavesDW", "StemDW", "tubersDW", "LAI"]
PURE_OBS_TO_PCSE = ["WLV", "TWST", "TWSO", "LAI"]

# Sowing dates — same for all plots in this dataset.
SOWING_DATE_BY_YEAR = {
    2019: pd.Timestamp("2019-04-20"),
    2020: pd.Timestamp("2020-04-20"),
}


def _output_scales():
    return torch.tensor(
        [3000.0, 2500.0, 20000.0, 8.0],   # WLV, TWST, TWSO, LAI
        dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device(),
    )


def _raw_pure_features(plot, date, weather_features, weather_feature_dim,
                       plot_context_fn):
    norm_date = pd.Timestamp(date).normalize()
    wf = weather_features[plot.Location].get(norm_date)
    if wf is None:
        wf = torch.zeros(weather_feature_dim, dtype=ComputeConfig.get_dtype(),
                         device=ComputeConfig.get_device())
    days = (norm_date - SOWING_DATE_BY_YEAR[plot.Year]).days
    days_t = torch.tensor([days / DAYS_NORMALIZER], dtype=ComputeConfig.get_dtype(),
                          device=ComputeConfig.get_device())
    ctx = plot_context_fn(plot.Cultivar, plot.Nitrogen, plot.Irrigation, plot.Location)
    return torch.cat([wf, days_t, ctx])


def prepare_pure_features(train_plots, obs_df, weather_features, weather_feature_dim,
                          plot_context_fn):
    """Compute (pure_features_for_date, mean, std) from the training plots.

    `mean` / `std` are computed from per-observation feature vectors, then the
    one-hot / binary-flag dimensions are reset to (0, 1) so they stay in their
    raw {0, 1} scale.
    """
    feats = []
    for plot in train_plots:
        rows = obs_df[
            (obs_df["Year"] == plot.Year)
            & (obs_df["Location"] == plot.Location)
            & (obs_df["Plotnumber"] == plot.Plotnumber)
            & obs_df[PURE_OBS_VARS].notna().any(axis=1)
        ]
        for _, r in rows.iterrows():
            feats.append(_raw_pure_features(
                plot, r["Date"], weather_features, weather_feature_dim, plot_context_fn,
            ))
    raw_train = torch.stack(feats)
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0).clamp_min(1e-3)
    # Keep one-hots / binary flags in their raw {0, 1} scale.
    std[weather_feature_dim + 1:] = 1.0
    mean[weather_feature_dim + 1:] = 0.0

    def pure_features_for_date(plot, date):
        raw = _raw_pure_features(plot, date, weather_features, weather_feature_dim,
                                 plot_context_fn)
        return (raw - mean) / std

    return pure_features_for_date, mean, std


class PureLSTM(torch.nn.Module):
    """Sequence-aware pure-ML reference. (seq_len, 14) -> (seq_len, 4)."""

    def __init__(self, n_features=14, hidden=LSTM_HIDDEN, n_outputs=4, output_scales=None):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=1, batch_first=True,
        )
        self.head = torch.nn.Linear(hidden, n_outputs)
        if output_scales is None:
            output_scales = _output_scales()
        self.register_buffer("output_scales", output_scales.clone().to(
            dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device(),
        ))
        self.to(device=ComputeConfig.get_device(), dtype=ComputeConfig.get_dtype())

    def forward(self, seq):
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        h, _ = self.lstm(seq)              # (B, T, hidden)
        raw = self.head(h)                 # (B, T, 4)
        return torch.sigmoid(raw) * self.output_scales


def build_lstm_data(plot, pure_features_for_date, obs_df, n_days=LSTM_SEQ_LEN):
    sowing = SOWING_DATE_BY_YEAR[plot.Year]
    feats = [pure_features_for_date(plot, sowing + pd.Timedelta(days=d))
             for d in range(n_days)]
    seq = torch.stack(feats)

    rows = obs_df[
        (obs_df["Year"] == plot.Year)
        & (obs_df["Location"] == plot.Location)
        & (obs_df["Plotnumber"] == plot.Plotnumber)
        & obs_df[PURE_OBS_VARS].notna().any(axis=1)
    ].sort_values("Date").reset_index(drop=True)
    if rows.empty:
        return seq, None, None
    indices, targets = [], []
    for _, r in rows.iterrows():
        d = (pd.Timestamp(r["Date"]).normalize() - sowing).days
        d = max(0, min(d, n_days - 1))
        indices.append(d)
        targets.append([r[v] for v in PURE_OBS_VARS])
    idx_t = torch.tensor(indices, dtype=torch.long, device=ComputeConfig.get_device())
    tgt_t = torch.tensor(targets, dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device())
    return seq, idx_t, tgt_t


def build_lstm_datasets(train_plots, test_plots, pure_features_for_date, obs_df,
                        n_days=LSTM_SEQ_LEN):
    train = {p: build_lstm_data(p, pure_features_for_date, obs_df, n_days) for p in train_plots}
    test  = {p: build_lstm_data(p, pure_features_for_date, obs_df, n_days) for p in test_plots}
    return train, test


def lstm_pooled_loss(lstm, plot_data_dict, weights_tensor):
    var_total = torch.zeros(4, dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device())
    var_count = torch.zeros(4, dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device())
    plot_preds = {}
    for plot, (seq, idx, tgt) in plot_data_dict.items():
        if idx is None or len(idx) == 0:
            continue
        pred_full = lstm(seq)
        pred_at_obs = pred_full[0, idx, :]
        plot_preds[plot] = pred_full[0]
        for i in range(4):
            valid = torch.isfinite(tgt[:, i])
            if not valid.any():
                continue
            pv = pred_at_obs[valid, i]
            tv = tgt[valid, i]
            scale = tv.mean().abs().clamp_min(1e-6)
            rmse = torch.sqrt(torch.mean(((pv - tv) / scale) ** 2))
            var_total[i] = var_total[i] + weights_tensor[i] * rmse
            var_count[i] = var_count[i] + 1
    n_plots = sum(1 for _, (_, idx, _) in plot_data_dict.items() if idx is not None)
    if n_plots == 0:
        return torch.zeros((), dtype=ComputeConfig.get_dtype()), {}, {}
    total = var_total.sum() / n_plots
    diag = {
        name: (var_total[i] / var_count[i].clamp_min(1)).item() / weights_tensor[i].item()
        for i, name in enumerate(PURE_OBS_TO_PCSE)
    }
    return total, diag, plot_preds


def train_pure_lstm(pure_lstm, lstm_data_train, lstm_data_test, pure_variable_weights,
                    model_path, force_retrain=False, pure_n_features=14):
    """Load or train the LSTM. Returns `lstm_run` dict."""
    model_path = Path(model_path)
    if model_path.exists() and not force_retrain:
        saved = torch.load(model_path, weights_only=False)
        pure_lstm.load_state_dict(saved["state_dict"])
        lstm_run = saved["lstm_run"]
        print(f"Loaded saved LSTM from {model_path.name}")
        print(f"  Saved train loss: {lstm_run['train_history'][-1]:.4f}")
        print(f"  Saved test  loss: {lstm_run['test_history'][-1]:.4f}")
        return lstm_run

    val_rng = np.random.default_rng(LSTM_VAL_SEED)
    all_train_plots_for_lstm = list(lstm_data_train.keys())
    shuffled = list(all_train_plots_for_lstm)
    val_rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * LSTM_VAL_FRACTION)))
    val_plots = shuffled[:n_val]
    train_subset_plots = shuffled[n_val:]
    lstm_data_train_subset = {p: lstm_data_train[p] for p in train_subset_plots}
    lstm_data_val = {p: lstm_data_train[p] for p in val_plots}
    print(f"  LSTM split: train_subset={len(lstm_data_train_subset)}  "
          f"val={len(lstm_data_val)}  test={len(lstm_data_test)}")

    optimizer = torch.optim.Adam(pure_lstm.parameters(), lr=0.005, weight_decay=1e-3)
    lstm_run = {"train_history": [], "val_history": [], "test_history": [], "diag_history": []}

    best_val_loss, best_step = float("inf"), -1
    best_state = copy.deepcopy(pure_lstm.state_dict())

    for step in range(LSTM_MAX_STEPS):
        optimizer.zero_grad()
        train_loss, train_diag, _ = lstm_pooled_loss(pure_lstm, lstm_data_train_subset, pure_variable_weights)
        with torch.no_grad():
            val_loss, _, _ = lstm_pooled_loss(pure_lstm, lstm_data_val, pure_variable_weights)
            test_loss, _, _ = lstm_pooled_loss(pure_lstm, lstm_data_test, pure_variable_weights)
        lstm_run["train_history"].append(train_loss.item())
        lstm_run["val_history"].append(val_loss.item())
        lstm_run["test_history"].append(test_loss.item())
        lstm_run["diag_history"].append(train_diag)

        if step % 50 == 0:
            marker = " *" if val_loss.item() < best_val_loss - LSTM_MIN_DELTA else ""
            print(f"  pure_lstm step {step:04d} | train={train_loss.item():.4f} "
                  f"val={val_loss.item():.4f}{marker} test={test_loss.item():.4f}")

        if val_loss.item() < best_val_loss - LSTM_MIN_DELTA:
            best_val_loss = val_loss.item(); best_step = step
            best_state = copy.deepcopy(pure_lstm.state_dict())

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(pure_lstm.parameters(), max_norm=2.0)
        optimizer.step()

        if step - best_step >= LSTM_PATIENCE:
            print(f"  pure_lstm early stopping at step {step} (best val {best_val_loss:.4f})")
            break

    pure_lstm.load_state_dict(best_state)
    torch.save({
        "state_dict": pure_lstm.state_dict(),
        "lstm_run": lstm_run,
        "n_features": pure_n_features,
        "seq_len": LSTM_SEQ_LEN,
        "hidden": LSTM_HIDDEN,
    }, model_path)
    print(f"Saved trained LSTM to {model_path}")
    return lstm_run
