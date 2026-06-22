"""Observation-date alignment and per-plot RMSE accumulator for the pooled loss.

The notebook keeps `pooled_loss` visible so students see the outer math; the
two helpers here handle the bookkeeping (matching observation dates to
simulated days, building target tensors, computing weighted RMSE per plot).
"""
from __future__ import annotations

import pandas as pd
import torch

from diffwofost.physical_models.config import ComputeConfig


def get_plot_observations(plot, simulation_days, obs_df, bio_columns, root_column,
                          obs_to_pcse, root_pcse):
    """Return (indices, targets) aligning observation rows to simulated days.

    `indices` is a LongTensor into `simulation_days`; `targets` is a dict of
    PCSE-name -> tensor of observations (one per matched row). Observations
    falling outside the simulation window are clipped to the nearest endpoint.
    Returns (None, None) if the plot has no usable observations.
    """
    rows = obs_df[
        (obs_df["Year"] == plot.Year)
        & (obs_df["Location"] == plot.Location)
        & (obs_df["Plotnumber"] == plot.Plotnumber)
    ].copy()
    keep = rows[bio_columns + [root_column]].notna().any(axis=1)
    rows = rows.loc[keep].sort_values("Date").reset_index(drop=True)

    lookup = {pd.Timestamp(d).normalize(): i for i, d in enumerate(simulation_days)}
    last_idx = len(simulation_days) - 1
    last_day = pd.Timestamp(simulation_days[-1]).normalize()
    first_day = pd.Timestamp(simulation_days[0]).normalize()
    matched = []
    for _, r in rows.iterrows():
        key = pd.Timestamp(r["Date"]).normalize()
        if key in lookup:
            matched.append((lookup[key], r))
        elif key > last_day:
            matched.append((last_idx, r))
        elif key < first_day:
            matched.append((0, r))
    if not matched:
        return None, None
    indices = torch.tensor([m[0] for m in matched], dtype=torch.long)
    df_m = pd.DataFrame([m[1] for m in matched])
    targets = {p: torch.tensor(df_m[obs_col].to_numpy(),
                               dtype=ComputeConfig.get_dtype(),
                               device=ComputeConfig.get_device())
               for obs_col, p in obs_to_pcse.items()}
    if df_m[root_column].notna().any():
        targets[root_pcse] = torch.tensor(df_m[root_column].to_numpy(),
                                          dtype=ComputeConfig.get_dtype(),
                                          device=ComputeConfig.get_device())
    return indices, targets


def per_plot_loss(results, targets, indices, weights):
    """Weighted normalised-RMSE accumulator for one plot.

    Returns (loss_tensor, diag_dict) where diag_dict[name] is the unweighted
    normalised RMSE per variable.
    """
    tl = torch.zeros((), dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device())
    diag = {}
    for name, target in targets.items():
        pred = results[name].index_select(0, indices)
        valid = torch.isfinite(target)
        if not torch.any(valid):
            continue
        pred = pred[valid]; target_valid = target[valid]
        scale = torch.mean(target_valid).abs().clamp_min(1e-6)
        rmse = torch.sqrt(torch.mean(((pred - target_valid) / scale) ** 2))
        tl = tl + weights[name] * rmse
        diag[name] = rmse.detach().cpu().item()
    return tl, diag
