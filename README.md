# diffwofost-tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronvree/diffwofost-tutorial/blob/master/hybrid_stress_correction.ipynb)

A self-contained Google Colab tutorial showing how to build a **hybrid crop
model** by plugging a small neural network into the
[WOFOST72](https://github.com/ajwdewit/pcse) simulator and training it
end-to-end with gradient descent *through* the engine, using the
[diffwofost](https://github.com/WUR-AI/diffWOFOST) Python package.

Click the badge above to open the notebook in Colab. `Runtime → Run all` walks
through the whole pipeline in 8–10 min on a free CPU runtime.

## What the notebook covers

1. **Data exploration** — a real potato field trial (two Dutch sites, six
   cultivars, three N-levels, two W-levels; 174 plot-years).
2. **The hybrid architecture** — replacing WOFOST's stock evapotranspiration
   block with a learnable `RFTRA` stress factor produced by a tiny MLP.
3. **End-to-end training** — gradient descent through the differentiable
   WOFOST engine. (Pre-trained checkpoint loaded by default so Colab runtime
   stays short.)
4. **Inspection** — loss curves, per-plot fits, learned stress profiles per
   cultivar and per N-treatment.
5. **Comparison against two references** — uncalibrated WOFOST72_PP (to see
   the gap the NN is closing) and a pure-ML LSTM (to see what physics-as-
   inductive-bias buys you).
6. **Bonus** — parameter sensitivities via autograd through the engine.

## Repo layout

```
.
├── hybrid_stress_correction.ipynb   ← the tutorial; open this in Colab
├── README.md                        ← this file
├── LICENSE                          ← EUPL-1.1 (code + notebook)
├── DATA_LICENSE.md                  ← CC BY-NC-SA 4.0 (field data + models)
└── scripts/
    ├── build_notebook.py            ← regenerates the .ipynb from a local diffwofost checkout
    └── build_models_bundle.py       ← packages the pre-trained weights for release
```

## What gets downloaded at runtime

The notebook's setup cells fetch everything from three public sources — there
is nothing to install or copy locally:

| Source | Files | Size | Licence |
|--------|-------|------|---------|
| [Harvard Dataverse: Ten Den et al. (2024)](https://doi.org/10.7910/DVN/1LC6W7) | `Plotspecific_processed.csv`, `Weatherfile_lelystad.xlsx`, `Weatherfile_vredepeel.xlsx` | ~900 KB | CC BY-NC-SA 4.0 |
| This repo's `v1.0` release | `models_bundle.zip` (pre-trained `stress_nn_random.pt` + `pure_lstm_random.pt`) | ~80 KB | CC BY-NC-SA 4.0 (derivative work) |
| [`ajwdewit/pcse`](https://github.com/ajwdewit/pcse) + [`ajwdewit/pcse_notebooks`](https://github.com/ajwdewit/pcse_notebooks) | `Wofost72_PP.conf`, `crops.yaml`, `potato.yaml`, `AGMT_C2_2020.agro` | ~50 KB | Apache-2.0 |

PyPI install: `diffwofost==0.4.0` (pulls `pcse`, `torch`, etc. transitively).

## Citing this work

If you use the notebook or the pre-trained models, please cite the field-trial
dataset:

> Ten Den, T., van de Wiel, I., van Evert, F., van Ittersum, M., de Wit, A.,
> & Reidsma, P. (2024). *Agronomic dataset on potato growth and yield in the
> Netherlands.* Harvard Dataverse. https://doi.org/10.7910/DVN/1LC6W7

…and the diffwofost package: https://github.com/WUR-AI/diffWOFOST

## Maintainer notes

This repo deliberately holds a *snapshot-style* fork of one notebook from the
upstream [diffwofost](https://github.com/WUR-AI/diffWOFOST) repo. Once the NN
integration code (`StressNN` and `NNStressFactor`, currently embedded in the
notebook's §2.1) is merged to diffwofost main and released to PyPI, the embed
section can be replaced with two simple imports and a `v2.0` release cut.

### Regenerate the notebook from a local diffwofost checkout

```bash
# 1. Dump cell sources from the upstream tutorial
cd ../diffWOFOST/docs/notebooks
python -c "import json; nb = json.load(open('hybrid_stress_correction_tutorial.ipynb', encoding='utf-8')); \
  json.dump([{'idx':i,'type':c['cell_type'],'src':''.join(c['source'])} for i,c in enumerate(nb['cells'])], \
            open('../../../diffwofost-tutorial/_cells_src.json','w',encoding='utf-8'), indent=2)"

# 2. Regenerate the Colab notebook + models bundle
cd ../../../diffwofost-tutorial
python scripts/build_notebook.py
python scripts/build_models_bundle.py
```
