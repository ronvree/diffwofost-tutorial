#!/usr/bin/env python3
"""One-off merge: master tutorial_utils + slim UX/pedagogy into the notebook."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "hybrid_stress_correction.ipynb"
UPSTREAM_PR = "https://github.com/WUR-AI/diffWOFOST"

SECTION_2 = """\
## 2. Run me first (Colab setup)

Run the collapsed setup cells below (click **Show code** on Colab to expand any of
them). They install diffwofost, download field data + pre-trained weights +
`tutorial_utils` helpers, and convert the weather files.

The hybrid NN classes (`StressNN`, `NNStressFactor`) are defined in **§5.1** once
we have introduced the architecture — not here.

After setup finishes, use **Runtime → Run all** (about 8–10 min on a free Colab
CPU runtime with pretrained weights).
"""

SECTION_5_1 = f"""\
### 5.1 The two plug-in components

`StressNN` is the tiny MLP that maps daily features → `RFTRA ∈ [0, 1]`.
`NNStressFactor` is the WOFOST evapotranspiration swap that calls it every
simulated day.

The implementations are included below so you can see exactly how the hybrid hook
works — skim the docstrings if you like, or jump to §6 for the input features.
They are embedded here because the diffwofost PyPI release does not yet ship this
NN integration (we install from a pinned commit in §2). Reference source:

- [`stress.py`]({UPSTREAM_PR}/blob/add-partitioning-sigmoid/src/diffwofost/ml_models/stress.py)
- [`evapotranspiration.py`]({UPSTREAM_PR}/blob/add-partitioning-sigmoid/src/diffwofost/ml_models/crop/evapotranspiration.py)
"""

KEY_STRESSNN = """\
> 🔑 **Key idea**: `StressNN` is a tiny MLP (14 → 16 → 1, sigmoid). It is initialised so the *untrained* output is ~0.90 (mild stress) — that way training can learn deviations from the WOFOST baseline without starting from a random init that kills all growth.
"""

KEY_NNSTRESS = """\
> 🔑 **Key idea**: `NNStressFactor` is the PCSE-side shim that calls the NN each day and returns its output as `RFTRA`. **This is the seam where ML meets physics**: same interface as WOFOST's stock evapotranspiration class, so the engine swaps one for the other with no changes elsewhere.
"""

FEATURE_BUILDER = """\
Each simulated day, a **feature builder** gathers that day's weather, the current
development stage (`DVS`), and the plot's treatment info into one input vector for
`StressNN`. The engine calls it automatically at every time step — you don't
invoke it yourself (`tutorial_utils.features.PlotFeatureBuilder`).
"""

DISCUSSION = """\
**Discussion question**

Crop stress is often **cumulative**: a week of low rain matters, not just today's
VPD. The hybrid stress NN is fed mostly same-day inputs (weather, `DVS`,
treatment), except for one explicit history term — **7-day rolling rainfall**.

1. Does that mix feel sufficient, or too memoryless for drought and N stress?
2. If stress should depend on what happened earlier in the season, what would you
   add? (Rolling windows? soil moisture from WOFOST? A recurrent network like the
   LSTM in §13?)
"""

SECTION_8_2 = """\
### 8.2 Reference run: default WOFOST (no NN)

Before training the hybrid, we run **stock WOFOST72_PP** on every plot — no neural
network, no stress correction. That gives a **reference curve** for the fit plots
in §11 and the comparison in §12.

> ☕ **This cell takes a few minutes** — it simulates all plot-years on CPU.

> **Important caveat.** This WOFOST run is *uncalibrated* for this dataset (stock
> potato parameters, generic sowing schedule). It shows the **gap the hybrid is
> targeting**, not proof that hybrid beats a properly calibrated WOFOST. See the
> diffWOFOST `optimization_*.ipynb` notebooks for parameter-only calibration.
>
> The same run also **balances the training loss** across variables (leaves, stems,
> tubers, LAI, roots) so tuber biomass does not dominate — handled automatically
> in the code below.
"""

FORM_RULES: list[tuple[str, str, str]] = [
    ("!pip install", "Install diffwofost", "§2 — pinned git install."),
    ("ComputeConfig.set_dtype", "Imports and global settings", "§2 — libraries, paths, float64 on CPU."),
    ("TUTORIAL_UTILS_URL", "Download data and helpers", "§2 — field data, models, PCSE configs, tutorial_utils."),
    ("convert_weather_files", "Convert weather to PCSE format", "§2 — weather Excel → PCSE format."),
    ("load_observations", "Load field-trial data", "§3 — parse CSV, build plot index."),
    ("crop_data_provider = YAMLCropDataProvider", "Engine setup", "§7 — WOFOST72_PP + NNStressFactor plug-in."),
    ("make_plot_context_tensor", "Feature encodings: plot context", "§6 — site, N, W, cultivar encodings."),
    ("build_weather_features", "Feature encodings: weather", "§6 — VPD, 7-day rolling rain, etc."),
    ("PlotFeatureBuilder, N_FEATURES", "Confirm NN input dimension", "§6 — 14 features total."),
    ("def pooled_loss", "Loss function: pooled RMSE", "§8.1 — normalised RMSE summed over variables."),
    ("Running reference WOFOST72_PP", "Reference WOFOST run", "§8.2 — all plot-years, ~3–4 min on CPU."),
    ("split_random", "Train/test split", "§8.3 — random 80/20 of plot-years."),
    ("EarlyStopper", "Train (or load) hybrid NN", "§10 — pretrained by default; FORCE_RETRAIN = True to retrain."),
    ("plot_cultivar_trajectories", "Plot: biomass and LAI by cultivar", "§4.1"),
    ("plot_site_comparison", "Plot: site comparison", "§4.2"),
    ("plot_treatment_effects", "Plot: treatment-effect preview", "§4.3"),
    ("plot_loss_curves", "Plot: training loss curves", "§11.1"),
    ("plot_per_plot_fits", "Plot: per-plot model fits", "§11.2"),
    ("plot_rftra_per_plot", "Plot: learned RFTRA trajectories", "§11.3"),
    ("plot_rftra_by_cultivar", "Plot: RFTRA profile by cultivar", "§11.4"),
    ("plot_rftra_by_nitrogen", "Plot: RFTRA profile by N treatment", "§11.5"),
    ("plot_per_variable_rmse_bars", "Plot: hybrid vs WOFOST RMSE", "§12"),
    ("prepare_pure_features", "LSTM data preparation", "§13.1 — feature sequences for pure-ML baseline."),
    ("train_pure_lstm", "Train (or load) LSTM", "§13.2 — pretrained by default."),
    ("plot_three_way_rmse_bars", "Plot: three-model test RMSE", "§13.3"),
    ("plot_three_way_trajectories", "Plot: three-model trajectories", "§13.3"),
    ("plot_train_test_gap", "Plot: hybrid vs LSTM generalisation", "§13.3"),
    ("plot_sensitivity_bars", "Plot: yield parameter sensitivities", "§14"),
]

KEEP_VISIBLE = (
    "class StressNN",
    "class NNStressFactor",
    "stress_nn = StressNN",
    "def run_with_param_overrides",
)


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_md(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")


def wrap_form(cell: dict, title: str, markdown: str) -> None:
    src = cell["source"]
    cell["source"] = [
        f'#@title {title} {{ display-mode: "form" }}\n',
        f"#@markdown {markdown}\n",
        "\n",
    ] + src


def maybe_form(cell: dict) -> None:
    if cell["cell_type"] != "code":
        return
    src = cell_text(cell)
    if src.startswith("#@title"):
        return
    if any(k in src for k in KEEP_VISIBLE):
        return
    for needle, title, md in FORM_RULES:
        if needle in src:
            wrap_form(cell, title, md)
            return


def extract_nn_block(cells: list) -> tuple[list, None]:
    """Remove §2.1 NN cells; return ordered cells to insert at §5.1."""
    ordered: list[dict] = []
    remove_idx: list[int] = []
    for i, c in enumerate(cells):
        t = cell_text(c)
        if "### 2.1 Embedded NN classes" in t:
            remove_idx.append(i)
        elif "🔑 **Key idea**: `StressNN`" in t:
            ordered.append(deepcopy(c))
            remove_idx.append(i)
        elif "class StressNN(torch.nn.Module)" in t:
            ordered.append(deepcopy(c))
            remove_idx.append(i)
        elif "🔑 **Key idea**: `NNStressFactor`" in t:
            ordered.append(deepcopy(c))
            remove_idx.append(i)
        elif "class NNStressFactor(Evapotranspiration)" in t:
            ordered.append(deepcopy(c))
            remove_idx.append(i)
    for i in sorted(remove_idx, reverse=True):
        del cells[i]
    for c in ordered:
        if "Key idea**: `StressNN`" in cell_text(c):
            set_md(c, KEY_STRESSNN.strip())
        elif "Key idea**: `NNStressFactor`" in cell_text(c):
            set_md(c, KEY_NNSTRESS.strip())
    return ordered


def replace_plot_feature_builder(cells: list) -> None:
    """Replace inline PlotFeatureBuilder + key-idea cells with short prose + import."""
    remove_idx: set[int] = set()
    insert_after: int | None = None
    for i, c in enumerate(cells):
        t = cell_text(c)
        if t.startswith("Finally a small callable"):
            set_md(c, FEATURE_BUILDER)
            insert_after = i
        elif "🔑 **Key idea**: `PlotFeatureBuilder`" in t:
            remove_idx.add(i)
        elif "class PlotFeatureBuilder:" in t:
            remove_idx.add(i)

    import_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from tutorial_utils.features import PlotFeatureBuilder, N_FEATURES\n",
            "\n",
            "print(f\"Total NN input dim: {N_FEATURES}\")\n",
        ],
    }

    new_cells: list[dict] = []
    for i, c in enumerate(cells):
        if i in remove_idx:
            continue
        new_cells.append(c)
        if i == insert_after:
            new_cells.append(import_cell)
    cells[:] = new_cells


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # §2 intro
    for c in cells:
        if cell_text(c).startswith("## 2. Run me first"):
            set_md(c, SECTION_2)
        if "### 2.2 Download" in cell_text(c):
            set_md(c, cell_text(c).replace("### 2.2", "### 2.1"))
        if cell_text(c).startswith("## 1. The big picture"):
            set_md(c, cell_text(c).replace("Potetial", "Potential"))
        if "embed them below" in cell_text(c):
            c["source"] = [
                line.replace("embed them below", "defined in §5.1")
                for line in c["source"]
            ]

    nn_block = extract_nn_block(cells)
    nb["cells"] = cells

    # Insert §5.1 before §6
    section_cells = [{"cell_type": "markdown", "metadata": {}, "source": []}]
    set_md(section_cells[0], SECTION_5_1)

    idx6 = next(i for i, c in enumerate(cells) if cell_text(c).startswith("## 6. Building"))
    if idx6 is None:
        raise SystemExit("§6 not found")

    for j, c in enumerate([section_cells[0]] + nn_block):
        cells.insert(idx6 + j, c)

    replace_plot_feature_builder(cells)

    for c in cells:
        if cell_text(c).startswith("**Discussion question**") and "current timestep" in cell_text(c):
            set_md(c, DISCUSSION)
        if cell_text(c).startswith("### 8.2 Reference"):
            set_md(c, SECTION_8_2)

    for c in cells:
        maybe_form(c)

    nb["cells"] = cells
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
