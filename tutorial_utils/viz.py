"""All matplotlib plotting used in the tutorial.

Each function takes its data explicitly (no notebook globals) and renders one
figure. Kept terse — the visual style mirrors the original notebook output so
post-refactor figures match pixel-for-pixel.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_VARIABLES = [
    ("LeavesDW", "Leaf dry matter (g / m²)"),
    ("StemDW", "Stem dry matter (g / m²)"),
    ("tubersDW", "Tuber dry matter (g / m²)"),
    ("LAI", "Leaf area index"),
]


# ---- §4. Data exploration ------------------------------------------------

def plot_cultivar_trajectories(obs_df):
    cultivar_colors = {
        c: plt.cm.tab10(i)
        for i, c in enumerate(sorted(obs_df["Cultivar"].dropna().unique()))
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, (col, title) in zip(axes.ravel(), PLOT_VARIABLES):
        for cultivar in sorted(obs_df["Cultivar"].dropna().unique()):
            subset = obs_df[(obs_df["Cultivar"] == cultivar) & obs_df[col].notna()]
            if subset.empty:
                continue
            ax.scatter(subset["Date"], subset[col],
                       s=16, alpha=0.5,
                       color=cultivar_colors[cultivar], label=cultivar)
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(title="Cultivar", loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()


def plot_site_comparison(obs_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    loc_colors = {"L": "tab:blue", "V": "tab:orange"}
    for ax, (col, title) in zip(axes.ravel(), PLOT_VARIABLES):
        for loc, sub in obs_df[obs_df[col].notna()].groupby("Location"):
            ax.scatter(sub["Date"], sub[col],
                       s=14, alpha=0.5,
                       color=loc_colors.get(str(loc), "gray"), label=str(loc))
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(title="Location")
    plt.tight_layout(); plt.show()


def plot_treatment_effects(obs_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_values = sorted(obs_df["Nitrogen"].dropna().unique())
    for n in n_values:
        sub = obs_df[(obs_df["Nitrogen"] == n) & obs_df["tubersDW"].notna()]
        axes[0].scatter(sub["Date"], sub["tubersDW"], s=14, alpha=0.5,
                        label=f"N = {n}")
    axes[0].set_title("tubersDW vs Date, by Nitrogen level")
    axes[0].set_ylabel("tubersDW (g / m²)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="x", rotation=30)

    w_values = sorted(obs_df["Irrigation"].dropna().unique())
    for w in w_values:
        sub = obs_df[(obs_df["Irrigation"] == w) & obs_df["LAI"].notna()]
        axes[1].scatter(sub["Date"], sub["LAI"], s=14, alpha=0.5,
                        label=f"W = {w}")
    axes[1].set_title("LAI vs Date, by Irrigation level")
    axes[1].set_ylabel("LAI")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()


# ---- §11. Inspect model performance --------------------------------------

def plot_loss_curves(training_run):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(training_run["train_history"], label="train", linewidth=2)
    ax.plot(training_run["test_history"], label="test", linewidth=2, linestyle="--")
    ax.set_title("Pooled normalised loss (random 80/20 split)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()


def select_representative_plots(plots, cultivars, n_per_cultivar=1):
    by_c = {}
    for p in plots:
        by_c.setdefault(p.Cultivar, []).append(p)
    selected = []
    for c in cultivars:
        if c in by_c:
            selected.extend(by_c[c][:n_per_cultivar])
    return selected


def _plot_observed(ax, plot, obs_col, obs_df):
    sub = obs_df[
        (obs_df["Year"] == plot.Year)
        & (obs_df["Location"] == plot.Location)
        & (obs_df["Plotnumber"] == plot.Plotnumber)
    ]
    ax.scatter(sub["Date"], sub[obs_col], s=28, color="black", zorder=5, label="Observed")


PER_PLOT_FIT_VARS = [
    ("WLV", "LeavesDW", "Leaf DM"),
    ("TWST", "StemDW", "Stem DM"),
    ("TWSO", "tubersDW", "Tuber DM"),
    ("LAI", "LAI", "LAI"),
]


def plot_per_plot_fits(display_plots, train_results, test_results, reference_results, obs_df):
    fig, axes = plt.subplots(len(display_plots), 4, figsize=(20, 3 * len(display_plots)),
                              squeeze=False)
    for row, plot in enumerate(display_plots):
        key = (plot.Year, plot.Location, plot.Plotnumber)
        fitted = test_results.get(key) or train_results.get(key)
        ref = reference_results.get(key)
        label = f"{plot.Cultivar}@{plot.Location}, {plot.Nitrogen}, {plot.Irrigation}, {plot.Year}"
        for col, (var, obs_col, title) in enumerate(PER_PLOT_FIT_VARS):
            ax = axes[row, col]
            if ref is not None:
                ax.plot(ref["day"], ref[var].detach().cpu().numpy(),
                        label="Default WOFOST72 (uncalibrated)", linewidth=1.6)
            if fitted is not None:
                ax.plot(fitted["day"], fitted[var].detach().cpu().numpy(),
                        label="Hybrid (NN stress)", linewidth=1.6, linestyle="--")
            _plot_observed(ax, plot, obs_col, obs_df)
            ax.set_title(f"{title} — {label}", fontsize=9)
            ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30, labelsize=7)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()


def plot_rftra_per_plot(display_plots, train_results, test_results):
    fig, axes = plt.subplots(1, len(display_plots), figsize=(4 * len(display_plots), 4),
                              squeeze=False)
    for col, plot in enumerate(display_plots):
        ax = axes[0, col]
        key = (plot.Year, plot.Location, plot.Plotnumber)
        fitted = test_results.get(key) or train_results.get(key)
        if fitted is not None:
            ax.plot(fitted["day"], fitted["RFTRA"].detach().cpu().numpy(),
                    color="tab:orange", linewidth=2)
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="no stress")
        ax.set_title(f"{plot.Cultivar}@{plot.Location} {plot.Nitrogen}{plot.Irrigation} {plot.Year}",
                     fontsize=9)
        ax.set_ylabel("RFTRA (learned stress factor)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30, labelsize=7)
    plt.tight_layout(); plt.show()


def plot_rftra_by_cultivar(cultivar_rftra_by_dvs, cultivars, dvs_centers):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, c in enumerate(cultivars):
        curve = cultivar_rftra_by_dvs.get(c)
        if curve is not None and not np.isnan(curve).all():
            ax.plot(dvs_centers, curve, marker="o", linewidth=2,
                    label=c, color=plt.cm.tab10(i))
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.6)
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.6, label="DVS=1 (anthesis)")
    ax.set_xlabel("DVS (development stage)")
    ax.set_ylabel("mean learned RFTRA")
    ax.set_title("Seasonal stress profile per cultivar")
    ax.set_ylim(0.0, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.show()


def plot_rftra_by_nitrogen(n_rftra_by_dvs, dvs_centers):
    n_levels = ["N0", "N1", "N2"]
    n_colors = {"N0": "tab:red", "N1": "tab:orange", "N2": "tab:green"}
    n_labels = {
        "N0": "N0 (no fertiliser)",
        "N1": "N1 (30% advised)",
        "N2": "N2 (130% advised)",
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    for n in n_levels:
        curve = n_rftra_by_dvs.get(n)
        if curve is not None and not np.isnan(curve).all():
            ax.plot(dvs_centers, curve, marker="o", linewidth=2,
                    color=n_colors[n], label=n_labels[n])
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.6)
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.6, label="DVS=1 (anthesis)")
    ax.set_xlabel("DVS (development stage)")
    ax.set_ylabel("mean learned RFTRA")
    ax.set_title("Seasonal stress profile per nitrogen treatment")
    ax.set_ylim(0.0, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.show()

    print("Mean RFTRA across the active season (DVS > 0) per N treatment:")
    for n in n_levels:
        finite = n_rftra_by_dvs[n][~np.isnan(n_rftra_by_dvs[n])]
        if len(finite):
            print(f"  {n}: mean = {finite.mean():.3f}  min = {finite.min():.3f}  max = {finite.max():.3f}")


# ---- §12. Reference comparison -------------------------------------------

def plot_per_variable_rmse_bars(variables, train_diag_per_plot, test_diag_per_plot, ref_diag_per_plot):
    n_vars = len(variables)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_vars)
    width = 0.27
    train_means = [np.mean(train_diag_per_plot[v]) if train_diag_per_plot[v] else 0 for v in variables]
    test_means  = [np.mean(test_diag_per_plot[v]) if test_diag_per_plot[v] else 0 for v in variables]
    ref_means   = [np.mean(ref_diag_per_plot[v]) if ref_diag_per_plot[v] else 0 for v in variables]

    ax.bar(x - width, ref_means, width, label="Default WOFOST72 (uncalibrated)",
           color="lightgray", edgecolor="black")
    ax.bar(x, train_means, width, label="Hybrid, on TRAIN", color="steelblue", edgecolor="black")
    ax.bar(x + width, test_means, width, label="Hybrid, on TEST", color="orange", edgecolor="black")

    ax.set_xticks(x); ax.set_xticklabels(variables)
    ax.set_ylabel("normalised RMSE (averaged over plots)")
    ax.set_title("Per-variable RMSE: uncalibrated WOFOST reference vs. trained hybrid")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()

    print()
    print("Per-variable summary:")
    for v, ref, tr, te in zip(variables, ref_means, train_means, test_means):
        print(f"  {v:<6} default={ref:.3f}  train={tr:.3f}  test={te:.3f}  "
              f"(test/train: {te / max(tr, 1e-6):.2f})")


# ---- §13. Three-way comparison -------------------------------------------

def plot_three_way_rmse_bars(pure_obs_to_pcse, ref_diag, final_test_diag, final_lstm_test_diag):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pure_obs_to_pcse))
    width = 0.27
    ax.bar(x - width, [ref_diag[v] for v in pure_obs_to_pcse], width,
           label="Default WOFOST72 (uncalibrated)", color="lightgray", edgecolor="black")
    ax.bar(x, [final_test_diag.get(v, np.nan) for v in pure_obs_to_pcse], width,
           label="Hybrid (PP + NN stress)", color="tab:orange", edgecolor="black")
    ax.bar(x + width, [final_lstm_test_diag.get(v, np.nan) for v in pure_obs_to_pcse], width,
           label="Pure LSTM (no physics)", color="tab:purple", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(pure_obs_to_pcse)
    ax.set_ylabel("normalised RMSE on TEST set")
    ax.set_title("Per-variable test RMSE: three models, random 80/20 split")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()


def plot_three_way_trajectories(sample_plot, lstm_traj, lstm_dates,
                                 reference_results, hybrid_results, obs_df, lstm_seq_len):
    ref_traj_sample = reference_results.get(
        (sample_plot.Year, sample_plot.Location, sample_plot.Plotnumber)
    )
    hybrid_traj_sample = hybrid_results.get(
        (sample_plot.Year, sample_plot.Location, sample_plot.Plotnumber)
    )
    obs_for_lstm = obs_df[
        (obs_df["Year"] == sample_plot.Year)
        & (obs_df["Location"] == sample_plot.Location)
        & (obs_df["Plotnumber"] == sample_plot.Plotnumber)
    ]
    organ_list = [
        ("WLV",  "LeavesDW",  "Leaves (WLV)", 0),
        ("TWST", "StemDW",    "Stems (TWST)", 1),
        ("TWSO", "tubersDW",  "Storage organs (TWSO)", 2),
        ("LAI",  "LAI",       "LAI", 3),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (var, obs_col, title, lstm_idx) in zip(axes.ravel(), organ_list):
        if ref_traj_sample is not None and var in ref_traj_sample:
            ax.plot(ref_traj_sample["day"], ref_traj_sample[var].detach().cpu().numpy(),
                    label="Default WOFOST72 (uncalibrated)", linewidth=2, color="tab:blue")
        if hybrid_traj_sample is not None and var in hybrid_traj_sample:
            ax.plot(hybrid_traj_sample["day"], hybrid_traj_sample[var].detach().cpu().numpy(),
                    label="Hybrid (PP + NN stress)", linewidth=2, linestyle="--", color="tab:orange")
        ax.plot(lstm_dates, lstm_traj[:, lstm_idx],
                label="Pure LSTM (no physics)", linewidth=2, linestyle=":", color="tab:purple")
        if obs_col in obs_for_lstm.columns:
            finite_obs = obs_for_lstm[obs_for_lstm[obs_col].notna()]
            if len(finite_obs):
                ax.scatter(finite_obs["Date"], finite_obs[obs_col],
                           s=44, color="black", zorder=5, label="Observed")
        ax.set_title(title); ax.set_ylabel("kg/ha (LAI: m²/m²)")
        ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(loc="upper left", fontsize=9)
    plot_label = (f"WOFOST + hybrid + LSTM — {sample_plot.Cultivar}@{sample_plot.Location} "
                  f"{sample_plot.Nitrogen}{sample_plot.Irrigation} {sample_plot.Year}")
    fig.suptitle(plot_label, y=1.02, fontsize=13)
    plt.tight_layout(); plt.show()


def plot_train_test_gap(hybrid_train, hybrid_test, lstm_train, lstm_test):
    print(f"Generalisation gap (test/train) on random 80/20 split:")
    print(f"  Hybrid    : train={hybrid_train:.4f}  test={hybrid_test:.4f}  "
          f"ratio={hybrid_test/max(hybrid_train,1e-9):.2f}")
    print(f"  Pure LSTM : train={lstm_train:.4f}  test={lstm_test:.4f}  "
          f"ratio={lstm_test/max(lstm_train,1e-9):.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["Hybrid\n(PP + NN stress)", "Pure LSTM\n(no physics)"]
    trains = [hybrid_train, lstm_train]
    tests  = [hybrid_test, lstm_test]
    xpos = np.arange(len(models))
    ax.bar(xpos - 0.2, trains, 0.4, label="train loss", color="tab:blue", edgecolor="black")
    ax.bar(xpos + 0.2, tests,  0.4, label="test loss",  color="tab:orange", edgecolor="black")
    ax.set_xticks(xpos); ax.set_xticklabels(models)
    ax.set_ylabel("pooled normalised RMSE")
    ax.set_title("Train vs. test loss — physics keeps the hybrid honest")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()


# ---- §14. Sensitivity ----------------------------------------------------

def plot_sensitivity_bars(sensitivities_pct, typical_perturb, twso_baseline):
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(sensitivities_pct.keys())[::-1]
    values = [sensitivities_pct[n] for n in names]
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="black", height=0.6)

    for bar, n, v in zip(bars, names, values):
        label = f"{v:+.1f}%   (perturbing {n} by {typical_perturb[n]:g} units)"
        if v >= 0:
            ax.text(max(v, 0.05) + 0.05, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha="left", fontsize=10)
        else:
            ax.text(min(v, -0.05) - 0.05, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha="right", fontsize=10)

    ax.axvline(0, color="black", linewidth=0.7)
    xlim = max(abs(v) for v in values) * 1.6 + 0.5
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("% change in final TWSO per typical-magnitude perturbation")
    ax.set_title(f"Yield sensitivity (baseline TWSO = {twso_baseline:.0f} kg/ha)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.show()
