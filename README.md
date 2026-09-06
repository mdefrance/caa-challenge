# Crédit Agricole Assurances — Claim Frequency & Severity Challenge (1st place)

Winning solution to the public [ENS *Challenge Data* #161](https://challengedata.ens.fr/challenges/161),
hosted by **Crédit Agricole Assurances**, on predicting insurance **claim
frequency** and **claim severity (amount)** — plus a full re-run of the same
pipeline a year later on the current [AutoCarver](https://github.com/mdefrance/AutoCarver),
with every number measured on one machine.

**Authors:** Mario Defrance & Zacharie Buisson — finished **1st** on the
challenge. The challenge was used as a teaching vehicle in the **Data Science**
course we taught to the final-year **Fintech** students at **CY Tech**.

The approach is a classic actuarial **frequency–severity decomposition**, with
all feature engineering and supervised binning handled by the open-source
[**AutoCarver**](https://github.com/mdefrance/AutoCarver) library and the final
models built with XGBoost tuned via Optuna.

> 📝 **The article lives here too — [`ARTICLE.md`](ARTICLE.md).** *"Stop losing
> accuracy to manual binning"* walks through the method and revisits it with the
> current library. Every figure it quotes was measured on the notebooks in this
> repository, on one machine, with nothing else running, and every measurement
> has a script here that regenerates it.

## Method at a glance

**Frequency model**
- Target: number of claims per policy (multiclass `0` / `1` / `2+`,
  derived from `FREQ × ANNEE_ASSURANCE`).
- Stratified train/dev split; **inverse-frequency class weighting** to handle
  the rarity of claims (0.69 % positive rate).
- Supervised binning of quantitative, qualitative and ordinal features with
  **AutoCarver** — 2025 used `MulticlassCarver` (one-vs-rest), the 2026 re-run
  uses `OrdinalCarver`, since `0 < 1 < 2+` is ordered.
- Feature pre-selection by target association and inter-feature redundancy.
- XGBoost + **Optuna** Bayesian tuning, with the number of pruned features
  treated as a tunable hyper-parameter.
- Expected frequency recovered via the **law of total expectation** over the
  predicted class probabilities.

**Severity / amount model**
- Models claim amount on policies with claims, carved with `ContinuousCarver`.
- Train/dev split stratified on a **discretized** claim amount — itself a
  carving, fit in the frequency notebook and saved as a small artifact.
- **Error-weighted**: each observation is weighted by the frequency model's
  absolute error, focusing the severity model on the frequency model's misses.
- End-to-end metric: `CHARGE = expected claim count × predicted claim amount`.

## Repository layout

```
src/
  frequency_model.ipynb              # 2025 — the winning claim-frequency model
  amount_model.ipynb                 # 2025 — the winning claim-severity model
  frequency_model_2026.ipynb         # 2026 re-run on the current AutoCarver
  amount_model_2026.ipynb            # 2026 severity re-run (consumes the frequency hand-off)
  frequency_model_2025_timed.ipynb   # AutoCarver 7.0.5 carving-only timing baseline
  amount_model_2025_timed.ipynb      # same, severity side
  ab_carver_arms_2026.ipynb          # A/B: carver geometry arms (one-vs-rest / multiclass /
                                     #   ordinal, plus a Wilson-CI-off control)
  frequency_model_2026_ordinal_selector.ipynb
                                     # negative result: the frequency arm re-selected with
                                     #   OrdinalSelector instead of ClassificationSelector
  utils/
    data_toolkit.py                  # feature engineering / processing helpers
    objectives.py                    # Optuna objectives, IsolationForest outlier handling
  model/                             # generated artifacts (gitignored, bar two ~2 kB carvers)
data/
  README.md                          # how to download the challenge data (not tracked here)
  ab_arms/                           # measurement outputs quoted by the article
tools/
  scrub_notebook_paths.py            # strips local filesystem paths out of saved notebook outputs
  make_hero_chart.py                 # renders docs/hero_*.svg from one carved feature
  seed_variance.py                   # same features, same 300 trials, four TPE seeds
  cmp_selectors.py                   # Classification vs Ordinal selector on one carved frame
  measure_grid.py                    # the same, varying each selector's association measures
docs/                                # figures used by the article, generated from this data
  hero_<F>.svg                       #   transparent and theme-aware (CSS vars + a
                                     #   prefers-color-scheme override injected into the file)
  hero_<F>.png / _dark.png           #   opaque fallbacks for platforms that refuse SVG
ARTICLE.md                           # the article itself; its figure paths are repo-relative
requirements-705.txt                 # frozen AutoCarver 7.0.5 environment for the 2025 baselines
```

Each CSV under `data/ab_arms/` is written by one of the above and nothing else:
`per_column_tau_c.csv` and `summary.csv` by `ab_carver_arms_2026.ipynb`,
`seed_variance.csv` by `tools/seed_variance.py`, `selector_measure_grid.csv` by
`tools/measure_grid.py`. Every measurement the article quotes has a script in
this repository that regenerates it.

## Getting started

```bash
uv sync
```

1. Download the challenge data into `data/` — see [`data/README.md`](data/README.md).
   Four files are needed, including `Incendies.csv`.
2. **Run the notebooks with `src/` as the working directory** — they resolve data
   as `../data/` and import `utils.*` relative to themselves. From a shell:
   `cd src && uv run jupyter lab`, or point your editor's kernel at `src/`.
   `CAA_DATA_DIR` overrides the data location.
3. Run the **frequency** notebook end to end — it writes the intermediates the
   severity model needs (`data/frequency_*.csv`, `data/oos_frequency_*.csv`) and
   fits the target carver that defines the severity split.
4. Run the matching **severity** notebook to produce the amount model and the
   final submission.

### Hardware

The measured runs used an NVIDIA GPU, and XGBoost trains on `device="cuda"`
whenever one is available. `src/utils/objectives.py` probes for it once at
import and falls back to `"cpu"` if there is none, so the notebooks run either
way — CPU is slower and XGBoost's histogram construction is not bit-identical
across devices, so a CPU run reproduces the pipeline but not the last digits.
Force either with `CAA_XGB_DEVICE=cpu` / `=cuda`.

Keep the eras separate: `frequency_model_2026.ipynb` → `amount_model_2026.ipynb`,
and the 2025 pair together. Crossing a 2026 severity run onto a 2025 frequency
hand-off makes the comparison meaningless.

The notebooks are stored **with their outputs** — they are the source of the
numbers the article quotes. The Optuna searches are **seeded**
(`TPESampler(seed=42)`), the train/dev splits and every XGBoost estimator use
`random_state=42`, and carving is deterministic, so a clean re-run reproduces
them. Re-running with `nbconvert --inplace` overwrites the stored outputs.

The 7.0.5 timing baselines need their own interpreter — 7.0.5 and the current
line are not API-compatible. `requirements-705.txt` is a full freeze of the
environment those wall-clocks were measured in, not a floor, because installing
`autocarver==7.0.5` today resolves a different scikit-learn and numpy than the
ones the article reports:

```bash
uv venv --python 3.11 .venv-705
uv pip install --python .venv-705 -r requirements-705.txt
```

`frequency_model_2025_timed.ipynb` and `amount_model_2025_timed.ipynb` are
carving-only trims of the 2025 notebooks that exist to measure wall-clock; run
them against that interpreter.

## Reproducing the article's numbers

The dependency floor is deliberate: `pyproject.toml` pins
`autocarver[jupyter]>=7.7.2`. Older releases still import and run, but they
apportion the selection budget differently and rank ordinal candidate groupings
differently, so they quietly produce a different feature mix and different
numbers rather than raising anything. Pin the floor and the notebooks reproduce
what the article reports.

Reproducible is not the same as representative, and the repository says so with a
measurement rather than a caveat: `tools/seed_variance.py` re-runs the identical
300-trial frequency search under four TPE seeds on one carved and selected feature
set. The spread it measures is why the article calls the two frequency arms a draw
instead of reading their difference as a result. Seed 42 is the control — it must
reproduce `frequency_model_2026.ipynb` exactly, and the script says so when it does
not.

```bash
uv run python tools/seed_variance.py      # ~3 h, writes data/ab_arms/seed_variance.csv
uv run python tools/cmp_selectors.py      # ~5 min
uv run python tools/measure_grid.py       # ~20 min, writes selector_measure_grid.csv
```

## Licence

MIT — see [`LICENSE`](LICENSE). The challenge dataset is not covered by it; it
remains subject to the competition's own terms of use.

## Acknowledgements

Thanks to **Crédit Agricole Assurances** and **ENS Challenge Data** for
organising the competition and releasing the dataset publicly, and to the
**Fintech** students at **CY Tech** who tackled it alongside us.

Built with [AutoCarver](https://github.com/mdefrance/AutoCarver) ·
[docs](https://autocarver.readthedocs.io) · `pip install autocarver`
