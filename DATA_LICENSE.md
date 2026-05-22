# Data and model-weight licensing

## Field-trial data — CC BY-NC-SA 4.0

The notebook downloads three files from Harvard Dataverse:

- `Plotspecific_processed.csv`
- `Weatherfile_lelystad.xlsx`
- `Weatherfile_vredepeel.xlsx`

These are part of the dataset:

> Ten Den, T., van de Wiel, I., van Evert, F., van Ittersum, M., de Wit, A.,
> & Reidsma, P. (2024). *Agronomic dataset on potato growth and yield in the
> Netherlands.* Harvard Dataverse.
> [https://doi.org/10.7910/DVN/1LC6W7](https://doi.org/10.7910/DVN/1LC6W7)

Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**This tutorial does not redistribute these files** — they are fetched at
runtime directly from Dataverse. Any further use must comply with the upstream
licence (attribute the authors, no commercial use, derivative works share
alike).

## Pre-trained model weights — CC BY-NC-SA 4.0

`stress_nn_random.pt` and `pure_lstm_random.pt` (shipped as `models_bundle.zip`
in this repo's GitHub releases) were trained on the field-trial dataset above
and are therefore derivative works. They are released under the same
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence
as the source data.

## PCSE stock files — Apache-2.0

`Wofost72_PP.conf`, `crops.yaml`, `potato.yaml`, and `AGMT_C2_2020.agro` are
fetched at runtime from the [`ajwdewit/pcse`](https://github.com/ajwdewit/pcse)
and [`ajwdewit/pcse_notebooks`](https://github.com/ajwdewit/pcse_notebooks)
repositories. Both are licensed under Apache License 2.0.

## Code (this repo + the notebook itself) — EUPL-1.1

See [LICENSE](LICENSE). Matches the upstream
[diffwofost](https://github.com/WUR-AI/diffWOFOST) package licence.
