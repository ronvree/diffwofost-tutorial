"""Generate hybrid_stress_correction.ipynb from the upstream tutorial.

Reads cell sources dumped from
  ../diffWOFOST/docs/notebooks/hybrid_stress_correction_tutorial.ipynb
(via the temporary _cells_src.json in the repo root) and assembles the Colab
variant by:

  * replacing the title cell with an "Open in Colab" preamble,
  * replacing the §2 Setup block (venv-flavoured) with a Colab boot section:
        - pip install diffwofost from a pinned commit on main
        - imports without the two NN classes
        - inline definitions of StressNN and NNStressFactor (not yet on PyPI)
        - download field-trial data + pre-trained weights + PCSE stock files
        - download the tutorial_utils/*.py helper modules from this repo
          so the rest of the notebook can `from tutorial_utils import ...`,
  * walking every upstream cell from §3 onwards through a TRANSFORMS table
    that either passes the cell through, replaces it with a thin call into
    `tutorial_utils`, or prepends a "🔑 Key idea" callout above it.

Run from the repo root:
    python scripts/build_notebook.py

Writes hybrid_stress_correction.ipynb to the repo root.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_JSON  = REPO_ROOT / "_cells_src.json"
OUT       = REPO_ROOT / "hybrid_stress_correction.ipynb"

# Source for the two embedded classes.
DIFFWOFOST_SRC = REPO_ROOT.parent / "diffWOFOST" / "src" / "diffwofost"
STRESS_NN_PY = DIFFWOFOST_SRC / "ml_models" / "stress.py"
NN_STRESS_FACTOR_PY = DIFFWOFOST_SRC / "ml_models" / "crop" / "evapotranspiration.py"

# When the upstream NN integration is merged + released, this is the URL
# students will be told to switch to:
UPSTREAM_PR_URL = "https://github.com/WUR-AI/diffWOFOST"

# Where this tutorial repo is hosted. Used by the Colab badge, the
# release-asset URL, and the tutorial_utils download URL.
# Update this one line if the repo is renamed or transferred and re-run.
TUTORIAL_REPO    = "ronvree/diffwofost-tutorial"
TUTORIAL_BRANCH  = "master"
TUTORIAL_VERSION = "v1.0"

TUTORIAL_RAW_BASE = (
    f"https://raw.githubusercontent.com/{TUTORIAL_REPO}/{TUTORIAL_BRANCH}"
)
TUTORIAL_DATA_URL  = f"{TUTORIAL_RAW_BASE}/data"
TUTORIAL_UTILS_URL = f"{TUTORIAL_RAW_BASE}/tutorial_utils"
COLAB_BADGE_URL = (
    f"https://colab.research.google.com/github/{TUTORIAL_REPO}/"
    f"blob/{TUTORIAL_BRANCH}/hybrid_stress_correction.ipynb"
)

# Field-trial data: Ten Den et al. (2024), CC BY-NC-SA 4.0.
DATAVERSE_DOI = "10.7910/DVN/1LC6W7"
FIELD_DATA_FILES = [
    "Plotspecific_processed.csv",
    "Weatherfile_lelystad.xlsx",
    "Weatherfile_vredepeel.xlsx",
]

# Helper modules pulled into /content/tutorial_utils at Colab boot.
TUTORIAL_UTILS_MODULES = [
    "__init__.py",
    "data.py",
    "features.py",
    "losses.py",
    "training.py",
    "evaluation.py",
    "viz.py",
    "lstm.py",
]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def extract_class_source(path: Path, class_name: str) -> str:
    """Return the source of one top-level class plus its docstring header.

    We strip the module docstring (replaced by the surrounding markdown cell)
    but keep the imports the class needs.
    """
    text = path.read_text(encoding="utf-8")
    if text.startswith('"""'):
        end = text.find('"""', 3)
        text = text[end + 3:].lstrip("\n")
    return text


def load_upstream_cells():
    return json.loads(SRC_JSON.read_text(encoding="utf-8"))


# ============================================================================
# Replacement cell bodies — these substitute for verbose upstream cells with
# thin shims that import from tutorial_utils. Each one is a complete cell.
# ============================================================================

LOAD_OBSERVATIONS_CELL = '''\
from tutorial_utils.data import load_observations, BIO_COLUMNS, ROOT_COLUMN

obs_df, PLOT_KEYS, plot_keys = load_observations(data_temp_dir / "Plotspecific_processed.csv")

print(f"Total plot-years (after Shadow drop): {len(PLOT_KEYS)}")
print()
print("Plot-years per (Year, Location, Cultivar):")
print(plot_keys.groupby(["Year", "Location", "Cultivar"]).size().unstack(fill_value=0))
print()
print("Per-variable observation counts:")
for col in BIO_COLUMNS + [ROOT_COLUMN]:
    print(f"  {col:<10} non-null: {obs_df[col].notna().sum()}")
'''

CULTIVAR_TRAJ_CELL = '''\
from tutorial_utils.viz import plot_cultivar_trajectories
plot_cultivar_trajectories(obs_df)
'''

SITE_COMPARISON_CELL = '''\
from tutorial_utils.viz import plot_site_comparison
plot_site_comparison(obs_df)
'''

TREATMENT_EFFECTS_CELL = '''\
from tutorial_utils.viz import plot_treatment_effects
plot_treatment_effects(obs_df)
'''

ENCODINGS_CELL = '''\
from tutorial_utils.features import (
    SITE_INDEX, N_LEVEL_NUMERIC, W_LEVEL_INDEX, CULTIVARS, CULTIVAR_INDEX,
    PLOT_CONTEXT_DIM, make_plot_context_tensor,
)

print(f"Plot context dim: {PLOT_CONTEXT_DIM}")
print(f"Example (C2, N2, W2, Lelystad): {make_plot_context_tensor('C2', 'N2', 'W2', 'L')}")
'''

WEATHER_FEATURES_CELL = '''\
from tutorial_utils.features import (
    saturation_vp, build_weather_features, WEATHER_FEATURE_DIM,
)

WEATHER_FEATURES = build_weather_features(weather_paths)
print(f"Weather feature dim: {WEATHER_FEATURE_DIM}")
print(f"Lelystad features pre-computed for {len(WEATHER_FEATURES['L'])} days")
print(f"Vredepeel features pre-computed for {len(WEATHER_FEATURES['V'])} days")
'''

LOSS_CELL = '''\
from tutorial_utils.losses import get_plot_observations, per_plot_loss

OBS_TO_PCSE = {"LeavesDW": "WLV", "StemDW": "TWST", "tubersDW": "TWSO", "LAI": "LAI"}
ROOT_PCSE = "TWRT"
VAR_BASE_WEIGHTS = {"WLV": 1.0, "TWST": 1.0, "TWSO": 1.0, "LAI": 1.0, "TWRT": 0.5}


def pooled_loss(plots, runner, weights):
    """Pooled normalised RMSE over plots and observed variables.

    For each plot: run `runner(plot)`, align predictions to observation dates,
    compute per-variable normalised RMSE (handled by `per_plot_loss`),
    weight-sum into a scalar.
    """
    total = torch.zeros((), dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device())
    n_used = 0
    all_diag = {k: [] for k in weights}
    per_plot = {}
    for p in plots:
        try:
            r = runner(p)
        except Exception as exc:
            print(f"  WARN: failed on plot {p}: {exc}")
            continue
        idx, tgt = get_plot_observations(
            p, r["day"], obs_df, BIO_COLUMNS, ROOT_COLUMN, OBS_TO_PCSE, ROOT_PCSE,
        )
        if idx is None:
            continue
        pl, diag = per_plot_loss(r, tgt, idx, weights)
        total = total + pl
        n_used += 1
        per_plot[(p.Year, p.Location, p.Plotnumber)] = r
        for k, v in diag.items():
            all_diag[k].append(v)
    avg_diag = {k: float(np.mean(v)) if v else float("nan") for k, v in all_diag.items()}
    return total / max(n_used, 1), avg_diag, per_plot, n_used


print("Loss machinery ready.")
'''

SPLIT_CELL = '''\
from tutorial_utils.data import split_random

train_plots, test_plots = split_random(PLOT_KEYS)
print(f"  train plot-years: {len(train_plots)}")
print(f"  test  plot-years: {len(test_plots)}")
'''

TRAINING_CELL = '''\
from tutorial_utils.training import EarlyStopper, try_load_checkpoint, save_checkpoint

FORCE_RETRAIN = False
MODEL_DIR = data_temp_dir / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)
model_path = MODEL_DIR / "stress_nn_random.pt"

training_config = {"lr": 0.02, "max_steps": 60, "patience": 15, "min_delta": 5e-4}

saved = try_load_checkpoint(stress_nn, model_path) if not FORCE_RETRAIN else None
if saved is not None:
    training_run = saved["training_run"]
    print(f"Loading saved model from {model_path.name}")
    print(f"  Previously trained for {len(training_run['train_history'])} steps")
    print(f"  Saved train loss: {training_run['train_history'][-1]:.4f}")
    print(f"  Saved test  loss: {training_run['test_history'][-1]:.4f}")
    print(f"  (Set FORCE_RETRAIN = True above to retrain.)")
else:
    print("Training from scratch — this will take a while...")
    optimizer = torch.optim.Adam(stress_nn.parameters(), lr=training_config["lr"])
    stopper = EarlyStopper(training_config["patience"], training_config["min_delta"], stress_nn)
    train_history, test_history, diag_history = [], [], []

    def runner(plot):
        return run_plot_with_nn(plot, stress_nn)

    # The SGD loop: forward through the engine, backward through the engine,
    # step. Autograd takes care of propagating gradients across the whole
    # simulation back into the NN parameters.
    for step in range(training_config["max_steps"]):
        optimizer.zero_grad()
        train_loss, train_diag, _, _ = pooled_loss(train_plots, runner, NORMALIZED_WEIGHTS)
        train_history.append(train_loss.detach().cpu().item())
        diag_history.append(train_diag)

        with torch.no_grad():
            test_loss, _, _, _ = pooled_loss(test_plots, runner, NORMALIZED_WEIGHTS)
        test_history.append(test_loss.detach().cpu().item())

        if step % 2 == 0:
            print(f"  step {step:03d} | train={train_history[-1]:.4f} test={test_history[-1]:.4f}")

        stopper.update_best(train_history[-1], step, stress_nn)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(stress_nn.parameters(), max_norm=1.0)
        optimizer.step()

        if stopper.should_stop(step):
            print(f"  early stopping at step {step}")
            break

    stopper.restore_best(stress_nn)
    training_run = {
        "train_history": train_history,
        "test_history":  test_history,
        "diag_history":  diag_history,
    }
    save_checkpoint(stress_nn, model_path, training_run=training_run, n_features=N_FEATURES)
    print(f"Saved trained model to {model_path}")

# Final evaluation — collect simulation outputs for every plot (train + test)
with torch.no_grad():
    final_train_loss, final_train_diag, train_results, _ = pooled_loss(
        train_plots, lambda p: run_plot_with_nn(p, stress_nn), NORMALIZED_WEIGHTS,
    )
    final_test_loss, final_test_diag, test_results, _ = pooled_loss(
        test_plots, lambda p: run_plot_with_nn(p, stress_nn), NORMALIZED_WEIGHTS,
    )

print()
print(f"FINAL: train loss={final_train_loss.item():.4f}  test loss={final_test_loss.item():.4f}")
'''

LOSS_CURVES_CELL = '''\
from tutorial_utils.viz import plot_loss_curves
plot_loss_curves(training_run)
'''

PER_PLOT_FITS_CELL = '''\
from tutorial_utils.viz import select_representative_plots, plot_per_plot_fits

display_plots = select_representative_plots(test_plots, CULTIVARS, n_per_cultivar=1)
if not display_plots:
    display_plots = select_representative_plots(train_plots, CULTIVARS, n_per_cultivar=1)
print(f"Showing fits for {len(display_plots)} plots (one per cultivar in the test set)")

plot_per_plot_fits(display_plots, train_results, test_results, REFERENCE_PLOT_RESULTS, obs_df)
'''

RFTRA_PER_PLOT_CELL = '''\
from tutorial_utils.viz import plot_rftra_per_plot
plot_rftra_per_plot(display_plots, train_results, test_results)
'''

RFTRA_BY_CULTIVAR_CELL = '''\
from tutorial_utils.evaluation import aggregate_rftra_by_dvs, DVS_CENTERS
from tutorial_utils.viz import plot_rftra_by_cultivar

cultivar_rftra_by_dvs = aggregate_rftra_by_dvs(
    train_plots + test_plots,
    [train_results, test_results],
    lambda p: p.Cultivar,
)
plot_rftra_by_cultivar(cultivar_rftra_by_dvs, CULTIVARS, DVS_CENTERS)
'''

RFTRA_BY_N_CELL = '''\
from tutorial_utils.evaluation import aggregate_rftra_by_dvs
from tutorial_utils.viz import plot_rftra_by_nitrogen

n_rftra_by_dvs = aggregate_rftra_by_dvs(
    train_plots + test_plots,
    [train_results, test_results],
    lambda p: p.Nitrogen,
)
plot_rftra_by_nitrogen(n_rftra_by_dvs, DVS_CENTERS)
'''

PER_VAR_BARS_CELL = '''\
from tutorial_utils.evaluation import per_plot_diag
from tutorial_utils.viz import plot_per_variable_rmse_bars

train_diag_per_plot = per_plot_diag(
    train_plots, train_results, list(NORMALIZED_WEIGHTS.keys()),
    obs_df, BIO_COLUMNS, ROOT_COLUMN, OBS_TO_PCSE, ROOT_PCSE,
)
test_diag_per_plot = per_plot_diag(
    test_plots, test_results, list(NORMALIZED_WEIGHTS.keys()),
    obs_df, BIO_COLUMNS, ROOT_COLUMN, OBS_TO_PCSE, ROOT_PCSE,
)
ref_diag_per_plot = per_plot_diag(
    train_plots + test_plots, REFERENCE_PLOT_RESULTS, list(NORMALIZED_WEIGHTS.keys()),
    obs_df, BIO_COLUMNS, ROOT_COLUMN, OBS_TO_PCSE, ROOT_PCSE,
)

variables = list(NORMALIZED_WEIGHTS.keys())
plot_per_variable_rmse_bars(variables, train_diag_per_plot, test_diag_per_plot, ref_diag_per_plot)
'''

LSTM_DATA_PREP_CELL = '''\
from tutorial_utils.lstm import (
    prepare_pure_features, build_lstm_datasets, PureLSTM,
    LSTM_SEQ_LEN, LSTM_HIDDEN, PURE_OBS_VARS, PURE_OBS_TO_PCSE, SOWING_DATE_BY_YEAR,
)

PURE_N_FEATURES = WEATHER_FEATURE_DIM + 1 + PLOT_CONTEXT_DIM

pure_features_for_date, PURE_FEATURES_MEAN, PURE_FEATURES_STD = prepare_pure_features(
    train_plots, obs_df, WEATHER_FEATURES, WEATHER_FEATURE_DIM, make_plot_context_tensor,
)

# Re-use the same per-variable loss weights as the hybrid model so the two
# losses are directly comparable.
PURE_VARIABLE_WEIGHTS = torch.tensor(
    [NORMALIZED_WEIGHTS[v] for v in PURE_OBS_TO_PCSE],
    dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device(),
)

print("Pre-building LSTM sequences for every plot...")
lstm_data_train, lstm_data_test = build_lstm_datasets(
    train_plots, test_plots, pure_features_for_date, obs_df,
)
print(f"  train sequences: {len(lstm_data_train)} plots, each ({LSTM_SEQ_LEN}, {PURE_N_FEATURES})")
print(f"  test  sequences: {len(lstm_data_test)} plots, each ({LSTM_SEQ_LEN}, {PURE_N_FEATURES})")
'''

LSTM_TRAIN_CELL = '''\
from tutorial_utils.lstm import train_pure_lstm, lstm_pooled_loss

lstm_model_path = MODEL_DIR / "pure_lstm_random.pt"

torch.manual_seed(23)
pure_lstm = PureLSTM(n_features=PURE_N_FEATURES, hidden=LSTM_HIDDEN)
print(f"PureLSTM: {sum(p.numel() for p in pure_lstm.parameters())} parameters")

lstm_run = train_pure_lstm(
    pure_lstm, lstm_data_train, lstm_data_test, PURE_VARIABLE_WEIGHTS,
    lstm_model_path, force_retrain=FORCE_RETRAIN, pure_n_features=PURE_N_FEATURES,
)

with torch.no_grad():
    final_lstm_train_loss, final_lstm_train_diag, lstm_train_preds = lstm_pooled_loss(
        pure_lstm, lstm_data_train, PURE_VARIABLE_WEIGHTS,
    )
    final_lstm_test_loss, final_lstm_test_diag, lstm_test_preds = lstm_pooled_loss(
        pure_lstm, lstm_data_test, PURE_VARIABLE_WEIGHTS,
    )

print()
print(f"Pure LSTM | final train loss = {final_lstm_train_loss.item():.4f}")
print(f"Pure LSTM | final test  loss = {final_lstm_test_loss.item():.4f}")
'''

THREE_WAY_BARS_CELL = '''\
from tutorial_utils.viz import plot_three_way_rmse_bars
plot_three_way_rmse_bars(PURE_OBS_TO_PCSE, ref_diag, final_test_diag, final_lstm_test_diag)
'''

THREE_WAY_TRAJ_CELL = '''\
from tutorial_utils.lstm import build_lstm_data
from tutorial_utils.viz import plot_three_way_trajectories

sample_plot_for_lstm = test_plots[0] if test_plots else train_plots[0]
sample_data = lstm_data_test.get(sample_plot_for_lstm) or lstm_data_train.get(sample_plot_for_lstm)
if sample_data is None:
    seq_sample, _, _ = build_lstm_data(sample_plot_for_lstm, pure_features_for_date, obs_df)
else:
    seq_sample = sample_data[0]

with torch.no_grad():
    lstm_traj = pure_lstm(seq_sample)[0].cpu().numpy()
sowing_for_sample = SOWING_DATE_BY_YEAR[sample_plot_for_lstm.Year]
lstm_dates = [sowing_for_sample + pd.Timedelta(days=d) for d in range(LSTM_SEQ_LEN)]

hybrid_results = {**train_results, **test_results}
plot_three_way_trajectories(
    sample_plot_for_lstm, lstm_traj, lstm_dates,
    REFERENCE_PLOT_RESULTS, hybrid_results, obs_df, LSTM_SEQ_LEN,
)
'''

TRAIN_TEST_GAP_CELL = '''\
from tutorial_utils.viz import plot_train_test_gap
plot_train_test_gap(
    final_train_loss.item(), final_test_loss.item(),
    final_lstm_train_loss.item(), final_lstm_test_loss.item(),
)
'''

SENSITIVITY_BARS_CELL = '''\
from tutorial_utils.viz import plot_sensitivity_bars

TYPICAL_PERTURB = {"SPAN": 5.0, "TSUM1": 50.0, "TSUM2": 100.0}
gradients = {
    "SPAN":  span_t.grad.item(),
    "TSUM1": tsum1_t.grad.item(),
    "TSUM2": tsum2_t.grad.item(),
}
twso_baseline = twso_final.item()
sensitivities_pct = {
    name: (gradients[name] * TYPICAL_PERTURB[name]) / twso_baseline * 100
    for name in gradients
}

plot_sensitivity_bars(sensitivities_pct, TYPICAL_PERTURB, twso_baseline)
'''

# ============================================================================
# LOCAL EDITS — cells that were hand-edited in this repo's notebook and never
# made it back into _cells_src.json (which is a faithful dump from upstream).
# Keeping the prose here means it survives the next re-dump from upstream.
# Wired into TRANSFORMS (as REPLACE_MD) and INSERT_AFTER_MD below.
# ============================================================================

LOCAL_SECTION_1_BIG_PICTURE = '''\
## 1. The big picture: why hybrid modelling?

Mechanistic crop models like **WOFOST** encode decades of agronomic knowledge as
ODEs (Ordinary Differential Equation): photosynthesis, partitioning, water balance, phenology. They are
*interpretable*, *physically grounded*, and *generalise* to weather and
management regimes outside their calibration data — but they have to make
simplifying assumptions, and the assumptions sometimes break.

Pure **data-driven** models (an LSTM that maps weather + soil → yield) can
absorb whatever structure is in the data, but they have **no inductive bias**:
nothing tells them carbon is conserved, that leaves grow before tubers, or that
photosynthesis caps at light saturation. They overfit small datasets badly and
extrapolate poorly outside the training distribution.

**Hybrid models** keep the physics where it's well-understood and let a neural
network fill in the parts that are poorly modelled. Concretely, this tutorial:

| Component | Source |
|-----------|--------|
| Phenology (DVS, TSUM) | physical (WOFOST) |
| Carbon partitioning to leaves/stems/tubers | physical (WOFOST) |
| Soil water balance (Potetial Production variant) | physical (WOFOST) |
| **Stress reduction factor (`RFTRA`)** | **learned NN** |

In standard WOFOST, `RFTRA` represents a transpiration reduction factor:
a scalar that reduces gross assimilation under water-limited conditions.
Strictly speaking, it is intended to capture water stress only.

In this tutorial, however, we deliberately use `RFTRA` more broadly as a
generic stress gate on assimilation. The `_PP` (potential production) variant
of WOFOST does not model nutrient stress, and therefore systematically
overpredicts growth on nitrogen-limited plots. Instead of implementing the full
water- and nitrogen-limited production variants, we train a small neural network
to infer an effective daily stress factor directly from weather and treatment
context.

The physics still governs phenology, carbon allocation, and crop growth
dynamics; the neural network only modulates how much carbon is fixed each day.

**Disclaimer** This is admittedly a modelling shortcut — we are “misusing” `RFTRA` beyond its
original physiological interpretation — but it is defensible because RFTRA
acts multiplicatively on gross carbon assimilation. From the model’s perspective, both drought stress and nitrogen stress ultimately reduce canopy assimilation, even if the underlying physiological mechanisms differ.

**Activity (discuss with a neighbour).** What other parts of a crop model might
be good candidates for a hybrid replacement? What are the trade-offs of
replacing more physics with NN vs. less?
'''

LOCAL_SECTION_5_HYBRID = '''\
## 5. The hybrid model

The hybrid we will build looks like this (simplified):

```
                +-------------------+      +-------------------+
   weather ---> | Phenology (DVS)   | ---> | Partitioning      |
   crop YAML    | (WOFOST physics)  |      | (WOFOST physics)  |
                +-------------------+      +-------------------+
                                                    |
                                                    v
                  +------------------+      +---------------------+
   weather ---->  | Stress NN        | -->  | Gross assimilation  |
   DVS, treatment | (learned RFTRA)  |      | (WOFOST × RFTRA)    |
                  +------------------+      +---------------------+
                                                    |
                                                    v
                                            organ biomass per day
                                                    |
                                            compare to observations
                                                    |
                                            loss --> backprop --> NN
```
**How it works.** In standard WOFOST, the daily stress reduction factor `RFTRA` is computed inside the evapotranspiration module. This factor represents how much crop assimilation should be reduced due to limited transpiration under water stress.
Under the `_PP` (potential production) configuration, however, no water stress
is simulated, and the model simply returns `RFTRA = 1.0` (no stress).

In this tutorial, we replace that evapotranspiration stress component with a
small neural-network module called `NNStressFactor`.

Each simulated day, the WOFOST engine calls this module to ask:
> “By how much should today’s gross assimilation be reduced?”


The replacement module:

1. assembles the day’s feature vector
(weather variables, DVS, and plot/treatment context),
2. feeds those inputs through a small multilayer perceptron (StressNN),
3. returns a sigmoid output as `RFTRA ∈ [0, 1]`.


Importantly, we use this learned stress factor not only for water stress, but
also implicitly for nutrient limitations such as nitrogen stress. This is a
modelling shortcut: physiologically, RFTRA was originally intended only for
transpiration-related stress. However, because it acts as a multiplicative
reduction on gross assimilation, it can also serve as an effective proxy for
other stressors that reduce canopy carbon fixation.

Because the entire simulation engine is implemented in PyTorch, every operation
remains differentiable. The seasonal loss (for example, yield prediction error)
therefore stays connected to the neural-network parameters through the complete
computational graph of the crop model.

Calling `loss.backward()` propagates gradients backward through the entire growing season and into the weights of `StressNN` — which is the central idea behind diffWOFOST.
'''

LOCAL_SECTION_6_FEATURES = '''\
## 6. Building input features for the NN

The NN gets three groups of inputs:

**Per-plot context (constant over the season).**

| Field | Encoding | Dim |
|-------|----------|-----|
| Site | binary (Lelystad=0, Vredepeel=1) | 1 |
| Nitrogen level | numeric (N0=0.0, N1=0.3, N2=1.3) | 1 |
| Irrigation level | binary (W1=0, W2=1) | 1 |
| Cultivar | one-hot of [C1, C2, C3, C4, C5, C6] | 6 |

**Per-day weather features (computed once per site).**

| Field | Computed from | Dim |
|-------|---------------|-----|
| VPD (vapor pressure deficit) | `SVP(TMAX) − VAP` | 1 |
| TMAX (max temp) | weather file | 1 |
| 7-day rolling rainfall | weather file | 1 |
| IRRAD (radiation) | weather file (÷ 1e4) | 1 |

**Per-day crop-state features (read from the engine kiosk daily).**

| Field | Dim |
|-------|-----|
| DVS | 1 |

**Total NN input dim: 4 (weather) + 1 (DVS) + 9 (plot context) = 14.**

'''

LOCAL_DISCUSSION_QUESTION = '''\
**Discussion question**

The neural network predicts a daily stress factor using information available at
the current timestep.

1. Do you think this is enough information to model crop stress realistically?
Why or why not?

2. If not, how could the model be modified to account for stress developing over
time?'''

LOCAL_SECTION_8_3_SPLIT = '''\
### 8.3 Train/test split

We use a simple random 80/20 split of the plot-years. This tests
*interpolation within the design space*: training and test see the same
cultivars, sites, and treatments — just different plots.'''

LOCAL_ACTIVITY_CULTIVAR_SPLIT = '''\
**Activity (discuss with a neighbour).** Look at the three views above.
On *training* data, which model wins? On *test* data? And if we would do spits by cultivar which would you expect to degrade more? Why?
'''

LOCAL_SECTION_15_RECAP = '''\
## 15. Recap and what's next

We just walked through the full pipeline of a hybrid crop model:

1. **Loaded a real field-trial dataset** (174 plot-years, two sites, six
   cultivars, three N levels, two W levels).
2. **Diagnosed the gap**: default (uncalibrated) WOFOST72_PP overpredicts
   biomass on water/N-stressed plots because it has no stress module.
3. **Replaced the stress component** with a tiny NN producing daily `RFTRA`.
4. **Trained the NN end-to-end** with gradient descent through the WOFOST
   engine.
5. **Inspected what the NN learned**: per-plot trajectories, per-cultivar and
   per-N-treatment stress profiles — checking that the NN actually uses its
   inputs.
6. **Compared against two reference points**: an uncalibrated WOFOST baseline
   (to see what the NN is correcting) and a pure-ML LSTM (to see what the
   engine's inductive bias contributes — especially the smaller train/test
   gap).
7. **Saw the bonus**: free gradients with respect to any parameter.

### Mental model: where does each model belong?

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Calibrated WOFOST** | Interpretable, transferable, mechanistic | Labour-intensive to fit; assumptions break |
| **Uncalibrated WOFOST (this notebook's reference)** | None — just a starting point | Systematic biases, especially under stress |
| **Pure LSTM** | High fit on training data | Unstable, brittle on held-out conditions |
| **Hybrid (this tutorial)** | Reliable shape from physics, NN closes the gap | NN is opaque; needs careful feature design |

### Where to go from here

- **Partitioning hybrid**:
  https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/hybrid_partitioning_wofost72.ipynb
  applies the same idea to a *different* WOFOST component (carbon
  partitioning), useful for understanding what hybrid replacement can and
  can't do.
- **Pure calibration**: see https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_phenology.ipynb (and friends) for parameter-only
  calibration (no NN) using the same differentiable engine — that's the
  apples-to-apples WOFOST comparison this notebook deliberately does not do.

### Caveats

- The trained NN is specific to this sowing schedule and weather window.
  Predictions outside the training distribution are unreliable.
- Cultivar-specific genetics (`SPAN`, `TSUM2`) are *not* calibrated here.
  Residuals related to leaf longevity and time-to-maturity remain visible.

That's it — happy hybrid modelling!
'''


# Extra markdown cells inserted *after* a given upstream idx. Used for cells
# that are entirely new (not present upstream), e.g. local pedagogical inserts.
INSERT_AFTER_MD = {
    20: LOCAL_DISCUSSION_QUESTION,    # after PlotFeatureBuilder code
}


# ----- Markdown callouts above key teaching cells --------------------------

CALLOUT_PLOT_FEATURE_BUILDER = (
    "> 🔑 **Key idea**: `PlotFeatureBuilder` is the daily NN-input assembler. "
    "Its `__call__(day, drv, kiosk)` signature is exactly what the engine "
    "invokes once per simulated day to build the input vector handed to the NN. "
    "Read it carefully — it's the bridge between the simulation state and the "
    "neural network.\n"
)

CALLOUT_ENGINE_WIRING = (
    "> 🔑 **Key idea**: this is where the NN slots *into* the WOFOST engine. "
    "Look for the line in `build_config` that overrides "
    "`cfg.CROP_COMPONENTS[\"evapotranspiration\"]` with `NNStressFactor` — that "
    "single line is what turns a pure physics simulator into a hybrid model. "
    "Every other component (phenology, partitioning, leaf dynamics) is left "
    "untouched.\n"
)

CALLOUT_LOSS_HIGH_LEVEL = (
    "### 8.1 Loss function\n"
    "\n"
    "**The loss in plain English.** For every training plot we (1) run the "
    "hybrid simulation, (2) line up the simulated trajectory against the "
    "field-measured observation dates, (3) compute a normalised RMSE per "
    "observed variable (`WLV`, `TWST`, `TWSO`, `LAI`, plus a half-weighted "
    "`TWRT`), and (4) sum these into a single scalar — weighted so each "
    "variable contributes on a comparable scale.\n"
    "\n"
    "*\"Normalised\"* means dividing by the mean of the observations, so the "
    "loss is in units of *fractional error* rather than the raw `kg/ha` (which "
    "would make `TWSO` dominate everything else).\n"
    "\n"
    "The boring bookkeeping — matching simulated days to observation dates, "
    "building target tensors — lives in `tutorial_utils.losses`. What stays "
    "below is the pooled-loss math itself.\n"
)

CALLOUT_SGD_LOOP = (
    "> 🔑 **Key idea**: every training step is *standard* SGD — "
    "`optimizer.zero_grad()` → forward (which runs a full season of WOFOST per "
    "plot) → `loss.backward()` → `optimizer.step()`. The novelty is that "
    "`backward()` propagates gradients **through the entire simulation** "
    "back into the NN parameters. From PyTorch's point of view the engine is "
    "just another differentiable module.\n"
)

CALLOUT_AUTOGRAD_BONUS = (
    "> 🔑 **Key idea**: setting `requires_grad=True` on a parameter tensor and "
    "calling `.backward()` once on any simulated quantity gives you "
    "∂(quantity)/∂(param) for **every** parameter in one pass. The engine being "
    "differentiable end-to-end means autograd tracks the dependency through "
    "the whole season — no finite-differencing, no manual derivation.\n"
)


# upstream_idx → action
#   "PASS_THROUGH"
#   ("REPLACE", code_src)
#   ("PREPEND_MD", md_text)
TRANSFORMS = {
    7:  ("REPLACE", LOAD_OBSERVATIONS_CELL),
    10: ("REPLACE", CULTIVAR_TRAJ_CELL),
    12: ("REPLACE", SITE_COMPARISON_CELL),
    14: ("REPLACE", TREATMENT_EFFECTS_CELL),
    15: ("REPLACE_MD", LOCAL_SECTION_5_HYBRID),       # local edit
    16: ("REPLACE_MD", LOCAL_SECTION_6_FEATURES),     # local edit
    17: ("REPLACE", ENCODINGS_CELL),
    18: ("REPLACE", WEATHER_FEATURES_CELL),
    20: ("PREPEND_MD", CALLOUT_PLOT_FEATURE_BUILDER),
    22: ("PREPEND_MD", CALLOUT_ENGINE_WIRING),
    24: ("REPLACE_MD", CALLOUT_LOSS_HIGH_LEVEL),      # replace the §8.1 markdown
    25: ("REPLACE", LOSS_CELL),
    28: ("REPLACE_MD", LOCAL_SECTION_8_3_SPLIT),      # local edit
    29: ("REPLACE", SPLIT_CELL),
    32: ("PREPEND_MD", CALLOUT_SGD_LOOP),
    33: ("REPLACE", TRAINING_CELL),
    36: ("REPLACE", LOSS_CURVES_CELL),
    38: ("REPLACE", PER_PLOT_FITS_CELL),
    40: ("REPLACE", RFTRA_PER_PLOT_CELL),
    42: ("REPLACE", RFTRA_BY_CULTIVAR_CELL),
    44: ("REPLACE", RFTRA_BY_N_CELL),
    47: ("REPLACE", PER_VAR_BARS_CELL),
    51: ("REPLACE", LSTM_DATA_PREP_CELL),
    53: ("REPLACE", LSTM_TRAIN_CELL),
    55: ("REPLACE", THREE_WAY_BARS_CELL),
    57: ("REPLACE", THREE_WAY_TRAJ_CELL),
    59: ("REPLACE", TRAIN_TEST_GAP_CELL),
    60: ("REPLACE_MD", LOCAL_ACTIVITY_CULTIVAR_SPLIT),  # local edit
    61: ("PREPEND_MD", CALLOUT_AUTOGRAD_BONUS),
    64: ("REPLACE", SENSITIVITY_BARS_CELL),
    66: ("REPLACE_MD", LOCAL_SECTION_15_RECAP),       # local edit
}


def emit_upstream_cell(cell):
    return md(cell["src"]) if cell["type"] == "markdown" else code(cell["src"])


def build():
    upstream = load_upstream_cells()
    cells = []

    # ---- Cell 0: Colab preamble (replaces upstream title block) -----------
    cells.append(md(
        '<div style="display: flex; align-items: center;">\n'
        '    <h1>Tutorial on hybrid crop modelling with diffWOFOST</h1>\n'
        '    <img src="https://raw.githubusercontent.com/WUR-AI/diffWOFOST/refs/heads/main/docs/logo/diffwofost.png" width="150" style="margin-left: 20px;">\n'
        '</div>\n'
        '\n'
        f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BADGE_URL})\n'
        '\n'
        'In this tutorial we will build a **hybrid crop model** by\n'
        'plugging a small neural network into WOFOST72 and training\n'
        'it end-to-end. The notebook\n'
        f'uses the [diffwofost]({UPSTREAM_PR_URL}) Python package and a public\n'
        'field-trial dataset.\n'
        '\n'
        'By the end you will have:\n'
        '\n'
        '- explored a real field-trial dataset (potato, two Dutch sites, 174 plot-years);\n'
        "- replaced WOFOST's standard evapotranspiration block with a learnable stress\n"
        '  factor (`RFTRA`) produced by a neural network;\n'
        '- compared the hybrid against an uncalibrated WOFOST72_PP reference *and*\n'
        '  against a pure-ML LSTM with no physics in the loop;\n'
        '- seen a teaser of what else differentiability buys you (parameter sensitivities\n'
        '  for free).\n'
        '\n'
        '> **Citations.** The field data is from Ten Den et al. (2024), Harvard\n'
        f'> Dataverse [doi:{DATAVERSE_DOI}](https://doi.org/{DATAVERSE_DOI}),\n'
        '> CC BY-NC-SA 4.0. The diffwofost engine and this tutorial are\n'
        '> EUPL-1.1. Pretrained models in this notebook are derivative works of\n'
        '> the field data and are therefore also released under CC BY-NC-SA 4.0.\n'
    ))

    # ---- Cell 1: §1 "big picture" (local edit overrides upstream) ---------
    cells.append(md(LOCAL_SECTION_1_BIG_PICTURE))

    # ---- Cell 2: §2 markdown intro to Colab boot --------------------------
    cells.append(md(
        '## 2. Run me first (Colab setup)\n'
        '\n'
        'The next four cells install diffwofost, import everything, define two auxillary\n'
        'classes, and download the data + pre-trained weights.\n'
    ))

    # ---- Cell 3: pip install ----------------------------------------------
    cells.append(code(
        '!pip install -q "diffwofost @ git+https://github.com/WUR-AI/diffWOFOST@0a4d4a3b6682"\n'
    ))

    # ---- Cell 4: imports (modified — no NN classes, no asserts) -----------
    cells.append(code(
        'import copy\n'
        'import warnings\n'
        'import sys\n'
        'import zipfile\n'
        'from pathlib import Path\n'
        'from urllib.request import urlretrieve\n'
        '\n'
        'import matplotlib.pyplot as plt\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import torch\n'
        'import openpyxl\n'
        'from openpyxl import Workbook\n'
        'from pcse.base import ParameterProvider\n'
        'from pcse.input import (\n'
        '    ExcelWeatherDataProvider,\n'
        '    YAMLAgroManagementReader,\n'
        '    YAMLCropDataProvider,\n'
        '    WOFOST72SiteDataProvider,\n'
        ')\n'
        'from pcse.util import DummySoilDataProvider\n'
        'from pcse.traitlets import Instance\n'
        '\n'
        '# diffwofost @ main (commit 0a4d4a3) — physics engine, Configuration\n'
        '#   with CROP_COMPONENTS support (post-v0.4.0), classic_waterbalance.\n'
        '# StressNN and NNStressFactor are NOT on PyPI yet; we embed them below.\n'
        'from diffwofost.physical_models.config import ComputeConfig, Configuration\n'
        'from diffwofost.physical_models.crop.wofost72 import Wofost72\n'
        'from diffwofost.physical_models.engine import Engine\n'
        'from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP\n'
        'from diffwofost.physical_models.crop.evapotranspiration import Evapotranspiration\n'
        '\n'
        'warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")\n'
        'ComputeConfig.set_device("cpu")\n'
        'ComputeConfig.set_dtype(torch.float64)\n'
        '\n'
        '# Colab puts everything under /content. These paths line up with what the\n'
        '# data-fetch cell below produces, and what the rest of the notebook expects.\n'
        'data_dir      = Path("/content/data")\n'
        'data_temp_dir = Path("/content/data_temp")\n'
        'print(f"data_dir      = {data_dir}")\n'
        'print(f"data_temp_dir = {data_temp_dir}")\n'
    ))

    # ---- Cell 5: StressNN callout (markdown) ------------------------------
    cells.append(md(
        '### 2.1 Embedded NN classes\n'
        'The two cells below define `StressNN` and `NNStressFactor`.\n'
    ))

    # ---- Cell 6: callout + embedded StressNN ------------------------------
    cells.append(md(
        "> 🔑 **Key idea**: `StressNN` is a tiny MLP (14 → 16 → 1, sigmoid). "
        "It's initialised so the *untrained* output is ~1.0 (no stress) — that "
        "way training only learns *deviations* from the WOFOST baseline, "
        "rather than having to re-discover \"no stress\" from a random init.\n"
    ))
    cells.append(code(extract_class_source(STRESS_NN_PY, "StressNN")))

    # ---- Cell 7: callout + embedded NNStressFactor ------------------------
    cells.append(md(
        "> 🔑 **Key idea**: `NNStressFactor` is the PCSE-side shim that holds "
        "the NN and returns its output as `RFTRA` every simulation day. **This "
        "is the seam where ML meets physics**: it has the same interface as "
        "WOFOST's stock `Evapotranspiration` class, so the engine can swap one "
        "for the other with zero changes elsewhere.\n"
    ))
    cells.append(code(extract_class_source(NN_STRESS_FACTOR_PY, "NNStressFactor")))

    # ---- Cell 8: data-fetch markdown --------------------------------------
    cells.append(md(
        '### 2.2 Download data and pre-trained weights\n'
        '\n'
        'Three sources:\n'
        '\n'
        "1. **This tutorial repo's `data/` directory** — three field-trial files\n"
        '   (~900 KB) plus `models_bundle.zip` with the pre-trained `stress_nn_random.pt`\n'
        '   (the hybrid) and `pure_lstm_random.pt` (the pure-ML reference).\n'
        '\n'
        '   The field-trial files are mirrored verbatim from the\n'
        f'   [Harvard Dataverse dataset by Ten Den et al. (2024)](https://doi.org/{DATAVERSE_DOI}),\n'
        '   licensed CC BY-NC-SA 4.0 (see [DATA_LICENSE.md](https://github.com/' + TUTORIAL_REPO + '/blob/' + TUTORIAL_BRANCH + '/DATA_LICENSE.md)).\n'
        '   The model weights are derivative works of the same data and ship\n'
        '   under the same licence.\n'
        '2. **PCSE stock files** — config, crop YAML, and an agromanagement\n'
        '   template, pulled from `ajwdewit/pcse` and `ajwdewit/pcse_notebooks`\n'
        '   (Apache-2.0).\n'
        "3. **Tutorial helper modules** — this repo's `tutorial_utils/*.py`,\n"
        '   pulled into `/content/tutorial_utils/` so the rest of the notebook\n'
        '   can `from tutorial_utils import ...`.\n'
    ))

    # ---- Cell 9: data-fetch code ------------------------------------------
    files_block = "\n".join(
        f'    "{name}",' for name in FIELD_DATA_FILES
    )
    utils_files_block = "\n".join(
        f'    "{name}",' for name in TUTORIAL_UTILS_MODULES
    )
    cells.append(code(
        'for d in [\n'
        '    data_temp_dir,\n'
        '    data_temp_dir / "trained_models",\n'
        '    data_dir / "conf",\n'
        '    data_dir / "crop",\n'
        '    data_dir / "agro",\n'
        ']:\n'
        '    d.mkdir(parents=True, exist_ok=True)\n'
        '\n'
        "# 1. Field-trial data + pre-trained models — this tutorial repo's data/.\n"
        f'#    Field data: Ten Den et al. (2024), CC BY-NC-SA 4.0, mirrored from\n'
        f'#    Harvard Dataverse (doi:{DATAVERSE_DOI}). Models: derivative works,\n'
        '#    same licence. See DATA_LICENSE.md.\n'
        f'TUTORIAL_DATA = "{TUTORIAL_DATA_URL}"\n'
        'for name in [\n'
        f'{files_block}\n'
        ']:\n'
        '    dest = data_temp_dir / name\n'
        '    if not dest.exists():\n'
        '        print(f"Downloading {name}...")\n'
        '        urlretrieve(f"{TUTORIAL_DATA}/{name}", dest)\n'
        '\n'
        'models_dir = data_temp_dir / "trained_models"\n'
        'zip_path = models_dir / "models_bundle.zip"\n'
        'if not (models_dir / "stress_nn_random.pt").exists():\n'
        '    print("Downloading pre-trained models...")\n'
        '    urlretrieve(f"{TUTORIAL_DATA}/models_bundle.zip", zip_path)\n'
        '    with zipfile.ZipFile(zip_path) as z:\n'
        '        z.extractall(models_dir)\n'
        '    zip_path.unlink()\n'
        '\n'
        '# 2. PCSE stock files (Apache-2.0)\n'
        'PCSE_URL = "https://raw.githubusercontent.com/ajwdewit/pcse/master/pcse/conf"\n'
        'NB_URL   = "https://raw.githubusercontent.com/ajwdewit/pcse_notebooks/master/data"\n'
        'for url, dest in [\n'
        '    (f"{PCSE_URL}/Wofost72_PP.conf",     data_dir / "conf" / "Wofost72_PP.conf"),\n'
        '    (f"{NB_URL}/crop/crops.yaml",        data_dir / "crop" / "crops.yaml"),\n'
        '    (f"{NB_URL}/crop/potato.yaml",       data_dir / "crop" / "potato.yaml"),\n'
        '    (f"{NB_URL}/agro/AGMT_C2_2020.agro", data_dir / "agro" / "AGMT_C2_2020.agro"),\n'
        ']:\n'
        '    if not dest.exists():\n'
        '        print(f"Downloading {dest.name}...")\n'
        '        urlretrieve(url, dest)\n'
        '\n'
        '# 3. The 2019 agromanagement file is a one-line variant of the 2020 template\n'
        '#    (different year). Generate it inline rather than maintaining a copy.\n'
        'agro_2019 = data_dir / "agro" / "AGMT_C2_2019.agro"\n'
        'if not agro_2019.exists():\n'
        '    agro_2019.write_text(\n'
        '        "Version: 1.0\\n"\n'
        '        "AgroManagement:\\n"\n'
        "        \"- 2019-04-20:\\n\"\n"
        '        "    CropCalendar:\\n"\n'
        "        \"        crop_name: \'potato\'\\n\"\n"
        "        \"        variety_name: \'Potato_C2_C5\'\\n\"\n"
        '        "        crop_start_date: 2019-04-20\\n"\n'
        "        \"        crop_start_type: \'sowing\'\\n\"\n"
        '        "        crop_end_date: 2019-10-31\\n"\n'
        "        \"        crop_end_type: \'maturity\'\\n\"\n"
        '        "        max_duration: 300\\n"\n'
        '        "    TimedEvents:\\n"\n'
        '        "    StateEvents:\\n"\n'
        '    )\n'
        '\n'
        '# 4. tutorial_utils helper modules — this repo, fetched as raw .py files.\n'
        f'TUTORIAL_UTILS_URL = "{TUTORIAL_UTILS_URL}"\n'
        'utils_dir = Path("/content/tutorial_utils")\n'
        'utils_dir.mkdir(parents=True, exist_ok=True)\n'
        'for name in [\n'
        f'{utils_files_block}\n'
        ']:\n'
        '    dest = utils_dir / name\n'
        '    if not dest.exists():\n'
        '        print(f"Downloading tutorial_utils/{name}...")\n'
        '        urlretrieve(f"{TUTORIAL_UTILS_URL}/{name}", dest)\n'
        'if "/content" not in sys.path:\n'
        '    sys.path.insert(0, "/content")\n'
        '\n'
        '# Convenient handles used by the rest of the notebook\n'
        'conf_path = data_dir / "conf" / "Wofost72_PP.conf"\n'
        'crop_path = data_dir / "crop"\n'
        '\n'
        'print("\\nAll data ready.")\n'
    ))

    # ---- Cell 10: weather conversion via tutorial_utils.data --------------
    cells.append(code(
        'from tutorial_utils.data import convert_weather_files\n'
        '\n'
        'weather_paths = convert_weather_files(\n'
        '    {\n'
        '        "L": data_temp_dir / "Weatherfile_lelystad.xlsx",\n'
        '        "V": data_temp_dir / "Weatherfile_vredepeel.xlsx",\n'
        '    },\n'
        '    data_temp_dir,\n'
        ')\n'
        'print(f"Weather files ready: {list(weather_paths.values())}")\n'
    ))

    # ---- Cells 11+: §3 onwards, transformed via TRANSFORMS table ----------
    for idx, cell in enumerate(upstream):
        if idx < 6:
            continue  # handled by the boot section above
        action = TRANSFORMS.get(idx, "PASS_THROUGH")

        if action == "PASS_THROUGH":
            cells.append(emit_upstream_cell(cell))
        else:
            kind = action[0]
            if kind == "REPLACE":
                cells.append(code(action[1]))
            elif kind == "REPLACE_MD":
                cells.append(md(action[1]))
            elif kind == "PREPEND_MD":
                cells.append(md(action[1]))
                cells.append(emit_upstream_cell(cell))
            else:
                raise ValueError(f"Unknown action {action} for idx {idx}")

        # Optional: extra markdown cell inserted *after* this upstream cell.
        # Used for content with no upstream counterpart (e.g. discussion
        # questions added by the maintainer of this Colab fork).
        if idx in INSERT_AFTER_MD:
            cells.append(md(INSERT_AFTER_MD[idx]))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python",
                "name": "python", "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3", "version": "3.11",
            },
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(cells)} cells)")


if __name__ == "__main__":
    build()
