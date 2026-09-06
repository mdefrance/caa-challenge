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
features by `src/utils/data_toolkit.py`. **The notebooks will not run without
it**, and the article's feature matrix is not reproducible from an equivalent
source — the joined columns depend on this extract's geography and period.

> **Source:** *Base de Données sur les Incendies de Forêts en France* (BDIFF),
> Ministère de l'Agriculture —
> https://bdiff.agriculture.gouv.fr/indicateurs/cartes
>
> Export the indicator maps to CSV and save the result as `data/Incendies.csv`.
> BDIFF is updated continuously, so an export taken today will not be
> byte-identical to the one behind the article's numbers.

If a Kaggle mirror of this repository's dataset exists, it carries the exact
extract that was used — prefer it for reproduction.

The dataset is published by Crédit Agricole Assurances for the public challenge
and has been anonymised/sanitised by them; please respect the challenge's terms
of use.
