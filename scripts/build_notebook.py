"""Generate hybrid_stress_correction.ipynb from the upstream tutorial.

Reads cell sources dumped from
  ../diffWOFOST/docs/notebooks/hybrid_stress_correction_tutorial.ipynb
(via the temporary _cells_src.json in the repo root) and assembles the Colab
variant by:

  * replacing the title cell with a "Open in Colab" preamble,
  * replacing the §2 Setup block (venv-flavoured) with a Colab boot section:
        - pip install diffwofost==0.4.0
        - imports without the two NN classes
        - inline definitions of StressNN and NNStressFactor (not yet on PyPI)
        - download field-trial data from Harvard Dataverse, models from this
          repo's release asset, PCSE stock files from raw.githubusercontent.com,
          and write AGMT_C2_2019.agro inline,
  * keeping every cell from §3 onwards unchanged.

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

# Where this tutorial repo is hosted. Used by both the Colab badge and the
# release-asset URL below. Update this one line if the repo is renamed or
# transferred (e.g. ronvree/diffwofost-tutorial -> WUR-AI/diffwofost-tutorial)
# and re-run this script to regenerate the notebook.
TUTORIAL_REPO    = "ronvree/diffwofost-tutorial"
TUTORIAL_BRANCH  = "master"    # flip to "main" after `git branch -M main`
TUTORIAL_VERSION = "v1.0"

TUTORIAL_DATA_URL = (
    f"https://raw.githubusercontent.com/{TUTORIAL_REPO}/{TUTORIAL_BRANCH}/data"
)
COLAB_BADGE_URL = (
    f"https://colab.research.google.com/github/{TUTORIAL_REPO}/"
    f"blob/{TUTORIAL_BRANCH}/hybrid_stress_correction.ipynb"
)

# Field-trial data: Ten Den et al. (2024), CC BY-NC-SA 4.0.
# Originally from Harvard Dataverse (doi:10.7910/DVN/1LC6W7); mirrored verbatim
# in this repo's data/ directory for reproducibility (CC BY-NC-SA 4.0 permits
# this provided attribution is preserved — see DATA_LICENSE.md).
DATAVERSE_DOI = "10.7910/DVN/1LC6W7"
FIELD_DATA_FILES = [
    "Plotspecific_processed.csv",
    "Weatherfile_lelystad.xlsx",
    "Weatherfile_vredepeel.xlsx",
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
    # Drop the module-level docstring (first triple-quoted block).
    if text.startswith('"""'):
        end = text.find('"""', 3)
        text = text[end + 3:].lstrip("\n")
    return text


def load_upstream_cells():
    return json.loads(SRC_JSON.read_text(encoding="utf-8"))


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
        'A self-contained Colab tutorial: build a **hybrid crop model** by\n'
        'plugging a small neural network into the WOFOST72 simulator and training\n'
        'it end-to-end with gradient descent *through* the engine. The notebook\n'
        f'uses the [diffwofost]({UPSTREAM_PR_URL}) Python package and a public\n'
        'field-trial dataset.\n'
        '\n'
        'By the end you will have:\n'
        '\n'
        '- explored a real field-trial dataset (potato, two Dutch sites, 174 plot-years);\n'
        "- replaced WOFOST's standard evapotranspiration block with a learnable stress\n"
        '  factor (`RFTRA`) produced by a neural network;\n'
        '- loaded a pre-trained checkpoint to skip ~15 min of training on Colab CPU;\n'
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

    # ---- Cell 1: §1 "big picture" (unchanged from upstream) ----------------
    cells.append(md(upstream[1]["src"]))

    # ---- Cell 2: §2 markdown intro to Colab boot --------------------------
    cells.append(md(
        '## 2. Run me first (Colab setup)\n'
        '\n'
        'The next four cells install diffwofost, import everything, define two\n'
        'classes that are not yet on PyPI (see callout below), and download the\n'
        'data + pre-trained weights. After they finish you can `Run All` and\n'
        'watch the rest of the notebook execute end-to-end (about 8–10 min on a\n'
        'free Colab CPU runtime).\n'
        '\n'
        '> 💡 On Colab go to **Runtime → Run all** once the install cell has\n'
        '> finished.\n'
    ))

    # ---- Cell 3: pip install ----------------------------------------------
    # Pin to a specific main commit (not the v0.4.0 PyPI release).
    # v0.4.0 predates the introduction of CROP_COMPONENTS on Configuration
    # (commit 0a4d4a3, PR #108) which the NN-integration code below depends on.
    # When a newer release is cut on PyPI containing both CROP_COMPONENTS and
    # the embedded NN classes, switch back to a simple PyPI pin and delete
    # the §2.1 embed cells.
    cells.append(code(
        '!pip install -q "diffwofost @ git+https://github.com/WUR-AI/diffWOFOST@0a4d4a3b6682"\n'
    ))

    # ---- Cell 4: imports (modified — no NN classes, no asserts) -----------
    cells.append(code(
        'import copy\n'
        'import warnings\n'
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
        '### 2.1 Embedded NN classes (temporary)\n'
        '\n'
        f'The two cells below define `StressNN` and `NNStressFactor`. They are\n'
        f'embedded here directly because **the diffwofost release on PyPI\n'
        f'(`v0.4.0`) does not yet ship the NN integration**. We install from\n'
        f'`main` above (pinned to a specific commit so the tutorial stays\n'
        f'reproducible), which gives us the recent `CROP_COMPONENTS` change to\n'
        f'`Configuration` needed to plug in the NN. The two embedded classes\n'
        f'themselves live on a development branch — reference source:\n'
        f'\n'
        f'- [`src/diffwofost/ml_models/stress.py`]({UPSTREAM_PR_URL}/blob/add-partitioning-sigmoid/src/diffwofost/ml_models/stress.py)\n'
        f'- [`src/diffwofost/ml_models/crop/evapotranspiration.py`]({UPSTREAM_PR_URL}/blob/add-partitioning-sigmoid/src/diffwofost/ml_models/crop/evapotranspiration.py)\n'
        f'\n'
        f"Once they're merged to main and a new PyPI release cuts, this whole\n"
        f'section can be replaced with two imports and the install line above\n'
        f'switched back to a simple `diffwofost==0.5.0`.\n'
    ))

    # ---- Cell 6: embedded StressNN ----------------------------------------
    cells.append(code(extract_class_source(STRESS_NN_PY, "StressNN")))

    # ---- Cell 7: embedded NNStressFactor ----------------------------------
    cells.append(code(extract_class_source(NN_STRESS_FACTOR_PY, "NNStressFactor")))

    # ---- Cell 8: data-fetch markdown --------------------------------------
    cells.append(md(
        '### 2.2 Download data and pre-trained weights\n'
        '\n'
        'Two sources:\n'
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
        '\n'
        'All downloads are cached: re-running the cell is a no-op.\n'
    ))

    # ---- Cell 9: data-fetch code ------------------------------------------
    files_block = "\n".join(
        f'    "{name}",' for name in FIELD_DATA_FILES
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
        '# Convenient handles used by the rest of the notebook\n'
        'conf_path = data_dir / "conf" / "Wofost72_PP.conf"\n'
        'crop_path = data_dir / "crop"\n'
        '\n'
        'print("\\nAll data ready.")\n'
    ))

    # ---- Cell 10: weather conversion (lightly adapted from upstream) ------
    # Reuse the upstream conversion function verbatim. weather_paths is set
    # exactly as before; data_temp_dir was set in the imports cell.
    cells.append(code(upstream[5]["src"]))

    # ---- Cells 11+: keep §3 onwards verbatim ------------------------------
    for cell in upstream[6:]:
        if cell["type"] == "markdown":
            cells.append(md(cell["src"]))
        else:
            cells.append(code(cell["src"]))

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
