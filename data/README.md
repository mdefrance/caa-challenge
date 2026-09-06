# Data

The challenge data is **not** stored in this repository. It is freely available
from the official ENS *Challenge Data* page (free registration with an email
address is required):

> **Crédit Agricole Assurances — Claim frequency & severity prediction**
> https://challengedata.ens.fr/challenges/161

Download the files below and place them in this `data/` folder. The notebooks
read from `../data/` (`data_path = "../data/"`).

## Required raw files

| File | Used by | Notes |
|------|---------|-------|
| `train_input_Z61KlZo.csv`  | both notebooks | training features |
| `train_output_DzPxaPY.csv` | both notebooks | training target  |
| `test_input_5qJzHrr.csv`   | both notebooks | out-of-sample features for the submission |

## Generated intermediates (created by running the notebooks)

These are produced by `src/frequency_model.ipynb` and consumed by
`src/amount_model.ipynb` — you do not need to download them:

- `frequency_<version>.csv`, `oos_frequency_<version>.csv`
- `src/predictions/<version>_pred.csv`

## Auxiliary data — required, not optional

`Incendies.csv` is public fire-incident reference data, joined in as extra
features by `src/data_toolkit.py`. **The notebooks will not run without
it**, and the article's feature matrix is not reproducible from an equivalent
source — the joined columns depend on this extract's geography and period.

> **Source:** *Base de Données sur les Incendies de Forêts en France* (BDIFF),
> Ministère de l'Agriculture —
> https://bdiff.agriculture.gouv.fr/indicateurs/cartes
>
> Export the indicator maps to CSV and save the result as `data/Incendies.csv`.
> BDIFF is updated continuously, so an export taken today will not be
> byte-identical to the one behind the article's numbers.

## Kaggle mirror — the exact extract

Every file above, including the `Incendies.csv` extract behind the article's numbers,
is mirrored as a Kaggle dataset. **Prefer it for reproduction** — a fresh BDIFF export
will not be byte-identical.

> https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025

The two 2026 notebooks also run there directly, attached to that dataset:
[frequency](https://www.kaggle.com/code/mariodefrance/caa-frequency-model) ·
[severity](https://www.kaggle.com/code/mariodefrance/caa-amount-model).

## Licence and attribution

**Challenge files** — Study Data from ENS *Challenge Data* #161, published by Crédit
Agricole Assurances for the public challenge and anonymised/sanitised by them
(`surface*`, `capital*`, `prev*`). Article 4.6.1 of the platform's
[terms of use](https://challengedata.ens.fr/terms_of_use) makes Study Data open under the
**Etalab Licence Ouverte** by default unless the provider states otherwise in the challenge
description; challenge #161 states nothing, so **Licence Ouverte 2.0** applies —
redistribution is permitted with attribution to Crédit Agricole Assurances via ENS
Challenge Data.

**`Incendies.csv`** — BDIFF, produced by the **Institut national de l'information
géographique et forestière (IGN)** for the Ministère de l'Agriculture et de la Souveraineté
Alimentaire, also under **Licence Ouverte 2.0**. IGN records that the base "peut contenir
des imprécisions et manquer d'exhaustivité", with particular caution for fires before 2020
outside the Mediterranean regions — so the burnt-area features carry a geographic reporting
bias, not just noise.
