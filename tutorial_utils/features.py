"""Per-plot context encodings and per-day weather features for the stress NN."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from diffwofost.physical_models.config import ComputeConfig


SITE_INDEX = {"L": 0, "V": 1}
N_LEVEL_NUMERIC = {"N0": 0.0, "N1": 0.3, "N2": 1.3}
W_LEVEL_INDEX = {"W1": 0, "W2": 1}
CULTIVARS = ["C1", "C2", "C3", "C4", "C5", "C6"]
CULTIVAR_INDEX = {c: i for i, c in enumerate(CULTIVARS)}

PLOT_CONTEXT_DIM = 3 + len(CULTIVARS)
WEATHER_FEATURE_DIM = 4
N_FEATURES = WEATHER_FEATURE_DIM + 1 + PLOT_CONTEXT_DIM


def make_plot_context_tensor(cultivar, nitrogen, irrigation, location):
    site_bit = float(SITE_INDEX[location])
    n_num = float(N_LEVEL_NUMERIC[nitrogen])
    w_bit = float(W_LEVEL_INDEX[irrigation])
    cultivar_oh = [0.0] * len(CULTIVARS)
    cultivar_oh[CULTIVAR_INDEX[cultivar]] = 1.0
    return torch.tensor(
        [site_bit, n_num, w_bit] + cultivar_oh,
        dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device(),
    )


def saturation_vp(temp_c):
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def precompute_weather_features(weather_path):
    df = pd.read_excel(weather_path, header=10).iloc[1:].reset_index(drop=True)
    df["DAY"] = pd.to_datetime(df["DAY"])
    for c in ["IRRAD", "TMIN", "TMAX", "VAP", "WIND", "RAIN", "SNOWDEPTH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("DAY").reset_index(drop=True)
    for c in ["RAIN", "TMAX", "VAP", "IRRAD"]:
        df.loc[df[c] <= -990, c] = np.nan
    df["RAIN"] = df["RAIN"].fillna(0.0)
    df["VPD"] = (saturation_vp(df["TMAX"]) - df["VAP"]).clip(lower=0.0)
    df["RAIN_ROLL_7D"] = df["RAIN"].rolling(7, min_periods=1).sum()
    df["IRRAD_NORM"] = df["IRRAD"] / 1e4

    features = {}
    for _, row in df.iterrows():
        t = pd.Timestamp(row["DAY"]).normalize()
        features[t] = torch.tensor(
            [row["VPD"], row["TMAX"], row["RAIN_ROLL_7D"], row["IRRAD_NORM"]],
            dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device(),
        )
    return features


def build_weather_features(weather_paths):
    return {site: precompute_weather_features(p) for site, p in weather_paths.items()}


class PlotFeatureBuilder:
    """Daily feature assembler for a specific plot.

    Pre-computed weather features and the plot context tensor are baked in at
    construction; only DVS is read fresh from the kiosk on each call.
    """

    def __init__(self, weather_features_by_date, plot_context):
        self.weather_features = weather_features_by_date
        self.plot_context = plot_context

    def __call__(self, day, drv, kiosk):
        wf = self.weather_features.get(pd.Timestamp(day).normalize())
        if wf is None:
            wf = torch.zeros(
                WEATHER_FEATURE_DIM,
                dtype=ComputeConfig.get_dtype(),
                device=ComputeConfig.get_device(),
            )
        dvs = kiosk["DVS"] if "DVS" in kiosk else torch.tensor(0.0)
        if not isinstance(dvs, torch.Tensor):
            dvs = torch.tensor(dvs, dtype=ComputeConfig.get_dtype())
        crop_state = dvs.flatten()[:1]
        return torch.cat([wf, crop_state, self.plot_context])
