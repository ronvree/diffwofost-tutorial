"""Post-training diagnostics: per-plot RMSE breakdown and DVS-binned RFTRA aggregation."""
from __future__ import annotations

import numpy as np
import torch

from diffwofost.physical_models.config import ComputeConfig

from .losses import get_plot_observations


DVS_BINS = np.linspace(0.0, 2.0, 21)
DVS_CENTERS = 0.5 * (DVS_BINS[:-1] + DVS_BINS[1:])


def per_plot_diag(plots, results_dict, weights_keys, obs_df, bio_columns, root_column,
                  obs_to_pcse, root_pcse):
    """Per-plot normalised RMSE per variable. Returns {var: [rmse, ...]}."""
    out = {k: [] for k in weights_keys}
    for plot in plots:
        key = (plot.Year, plot.Location, plot.Plotnumber)
        r = results_dict.get(key)
        if r is None:
            continue
        idx, tgt = get_plot_observations(
            plot, r["day"], obs_df, bio_columns, root_column, obs_to_pcse, root_pcse,
        )
        if idx is None:
            continue
        for name, target in tgt.items():
            pred = r[name].index_select(0, idx)
            valid = torch.isfinite(target)
            if not torch.any(valid):
                continue
            pred = pred[valid]; target_valid = target[valid]
            scale = torch.mean(target_valid).abs().clamp_min(1e-6)
            rmse = torch.sqrt(torch.mean(((pred - target_valid) / scale) ** 2))
            out[name].append(rmse.item())
    return out


def aggregate_rftra_by_dvs(plots, results_lookups, group_key):
    """Bin RFTRA by DVS, averaging across plots that share `group_key(plot)`.

    `results_lookups` is an iterable of dicts to try in order (e.g. test_results
    then train_results) — the first one containing the plot wins.
    Returns {group_value: rftra_means_array_of_len_DVS_CENTERS}.
    """
    groups = sorted({group_key(p) for p in plots})
    sums = {g: np.zeros(len(DVS_CENTERS)) for g in groups}
    counts = {g: np.zeros(len(DVS_CENTERS)) for g in groups}
    for plot in plots:
        key = (plot.Year, plot.Location, plot.Plotnumber)
        r = None
        for d in results_lookups:
            if key in d:
                r = d[key]; break
        if r is None:
            continue
        dvs = r["DVS"].detach().cpu().numpy()
        rftra = r["RFTRA"].detach().cpu().numpy()
        active = (rftra > 1e-6) & (dvs > 0)
        if not active.any():
            continue
        bin_idx = np.clip(np.digitize(dvs[active], DVS_BINS) - 1, 0, len(DVS_CENTERS) - 1)
        g = group_key(plot)
        for b, v in zip(bin_idx, rftra[active]):
            sums[g][b] += v
            counts[g][b] += 1
    out = {}
    for g in groups:
        means = np.full(len(DVS_CENTERS), np.nan)
        nz = counts[g] > 0
        means[nz] = sums[g][nz] / counts[g][nz]
        out[g] = means
    return out
