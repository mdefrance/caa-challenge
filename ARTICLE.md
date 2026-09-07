# We re-ran our winning insurance model a year later

*Carving got 27× faster. The accuracy gap turned out to be a coin flip — and the way
we found that out is the part worth stealing.*

> By **Mario Defrance**, with **Zacharie Buisson**, who co-built the winning solution.
> Every number below is measured, and the notebooks in this repository are stored with
> the outputs that produced them.

> *Disclosure: I am the author and maintainer of AutoCarver. Read the library sections
> with that in mind; every number is reproducible from the repository linked at the end.*

---

## The finish line, first

![SURFACE4, floor area in sixteen raw bands merged into two buckets at 1000 m²; lower
panel shows each band's share of the portfolio against the 2 % minimum-frequency
line](docs/hero_SURFACE4.svg)

*Tschuprow's T between the feature and the target, on held-out data: 0.0240 across the
sixteen raw levels, 0.0358 across the two buckets — a 1.49× rise.*

That is one real feature from the challenge data. Floor area comes in sixteen bands;
claim frequency climbs roughly tenfold across them. Supervised binning cuts them **in
two, at 1000 m²** — and not because of the frequency floor: ten of the sixteen bands
clear the 2 % `min_freq`, so finer groupings were admissible and the carver's association
measure ranked this one above all of them, with or without the dev check
(`tools/make_hero_chart.py --explain`). What the lower panel shows is why the tail is not
to be trusted: every band from 4500 m² to 7000 m² covers less than 2 % of the portfolio,
so the most dramatic-looking rates rest on a handful of policies. Nobody picked that cut,
and nobody had to defend it in a meeting.

Do that for four hundred features and you have the quiet half of a winning model — the
half most pipelines settle with a default `qcut`.

It took us to **first place** in the Crédit Agricole Assurances *Data Science Academy*
hackathon [6], out of more than 500 participants. The hackathon ran on the public [ENS
*Challenge Data* #161](https://challengedata.ens.fr/challenges/161) [1] and was ranked on
that platform's private leaderboard, as it stood when the hackathon closed in spring 2025;
the results were presented that June. One thing to keep in mind if you go and look: the
challenge is still open to submissions, so the live leaderboard is a later snapshot than
the one we won, and nothing in this article is a leaderboard score. The challenge doubled
as the running example in the Data Science course we taught to final-year Fintech students
at CY Tech.

Re-running the pipeline a year later on today's tooling, **multiprocessing and a new
search algorithm turned 133 minutes of carving into 4.3**. The rest of this article is
how — and, just as honestly, which shiny new features changed nothing at all.

## The recipe, in three decisions

No giant model, no ensemble of ensembles. Three decisions did it:

- **Factor the problem** into frequency × severity, like actuaries do.
- **Bin every feature against the target** with the open-source
  [AutoCarver](https://github.com/mdefrance/AutoCarver) library (mine — see the
  disclosure above) [2] — this is where we believe the bulk of the lift came from; we
  did not run an ablation without it, and say so in §3.4.
- **Weight the rare signal**: class weights in the loss, error-weighted
  observations across the two stages.

A year later, AutoCarver has grown a lot, so we re-ran the challenge on the
current version and measured what each new feature is actually worth — including
the ones that turned out to be worth nothing.

---

## 1. The problem

Insurers don't predict "will this policy have a claim and how much" in one shot. They
factor it, the way actuaries have for decades:

```
expected cost  =  E[ number of claims ]  ×  E[ amount per claim ]
                  └──── frequency ────┘     └──── severity ────┘
```

The CAA challenge asked exactly this: predict **claim frequency** and **claim
severity** from anonymised policy and property features. Two things make it
harder than it looks:

1. **Claims are rare.** Almost every policy has zero. A model that predicts "zero"
   for everyone looks accurate and is useless — the signal lives in a thin tail.
2. **The targets behave nothing alike.** Frequency is a small count; severity a
   heavy-tailed amount. One model for both blurs each.

Before reading on, worth asking yourself: **how would *you* handle a signal this
rare?** Oversample it? Reweight it? Trust the tree to find it? Hold your answer against
what follows.

So: two models.

## 2. The winning recipe (2025)

### 2.1 The quiet workhorse: carve the features first

The least glamorous part, and the step we would defend first.

Raw insurance features are a mess: high-cardinality categoricals, ordinal levels with
tiny populations, skewed numerics, and `NaN`s that *mean something*. Throw those at a
tree model and it spends its depth rediscovering structure you could have handed it.

**Supervised binning** hands it that structure. AutoCarver searches the
admissible groupings of each feature's values and keeps the one that maximises
statistical association with the target — under a minimum bin frequency, a cap
on bins per feature, and (crucially) **validation on a held-out dev set**, so
a grouping that only works on train is rejected outright. Declare your feature
types once, carve everything in one `fit`:

```python
# AutoCarver 7.0.5, as written in 2025 — see §3 for the current equivalent.
from AutoCarver import Features, MulticlassCarver

features = Features(
    categoricals=["property_type", "region", ...],
    numericals=["insured_value", "surface", ...],
    ordinals={"quality_band": ["low", "mid", "high", "premium"]},
)

carver = MulticlassCarver(features=features, min_freq=0.02, max_n_mod=5)
train_carved = carver.fit_transform(train, train["n_claims"], X_dev=dev, y_dev=dev["n_claims"])

carver.summary  # every bucket: frequency, target rate, association
```

⚠️ **That snippet still runs today, and does something else.** In 7.0.5,
`MulticlassCarver` carved **one-vs-rest** — one binary carving per target class, so every
raw feature came back as several carved columns. That behaviour is `OneVsRestCarver` now;
`MulticlassCarver` means *one* carving per feature. Same import, different output. Every
snippet in §2 is 2025 code, kept as written; §3 is the current API.

**A word on [optbinning](https://github.com/guillermo-navas-palencia/optbinning).** The
obvious alternative [5], and a proven one on tabular insurance data: it solves binning to
*proven* optimality with a CP/MILP formulation rather than by heuristic search.
We went a different way for reasons about how we framed the problem, not about the
library: we wanted every grouping tested on a held-out sample, and `fit(x, y)` has no
notion of one; our target is ordered, where `MulticlassOptimalBinning` does not treat it
so (§3.2); and its multiclass path is numerical-only, while most of our 435 features are
categorical or declared ordinals.

What that bought us here:

- ordinal features merged **in their natural order**, never collapsed by raw frequency;
- missing values **never silently imputed**: `dropna=True` carves `NaN` as its own
  association-scored bucket. We ran `dropna=False` and let XGBoost route them natively —
  the point is that it is a declared choice, not a default;
- buckets we could **read, sanity-check, and defend**, for every feature;
- one uniform pipeline over numeric, categorical and ordinal features.

We then pre-selected features by association with the target and pruned redundant ones
(`ClassificationSelector` does both in one pass).

### 2.2 Frequency: a weighted multiclass model

Almost no policy has more than two claims, so frequency became a **multiclass**
problem over `0 / 1 / 2+` claims. Inverse-frequency class weights stopped the
model from collapsing onto the majority class. To turn class probabilities
back into an expected frequency, the law of total expectation:

```
E[Y] = P(0)·E[Y|0] + P(1)·E[Y|1] + P(2+)·E[Y|2+]
```

Every log loss in this article is the **class-weighted** one, using those same
inverse-frequency weights — it is what the search minimised. Unweighted, a constant
base-rate model already scores 0.041; weighted, it scores 4.47, and a uniform ⅓ guess
scores 1.099. Read 0.92 against those.

### 2.3 Severity: spend capacity on the frequency model's mistakes

The severity model only sees policies with claims. The trick we credit most, unmeasured in isolation:
**weight each observation by the frequency model's absolute error**, so it concentrates
exactly where the first stage was wrong — a cheap, boosting-flavoured correction across
the two stages.

### 2.4 XGBoost, tuned honestly

XGBoost on top, tuned with Optuna. One idea worth stealing: the **number of pruned
features was itself a hyper-parameter** — Optuna decided how aggressively to trim,
alongside the usual ones.

Factor the problem, carve the features, weight the rare signal, tune the pruning.
**1st place in the hackathon.**

## 3. Re-running it with today's AutoCarver (2026)

We won with AutoCarver `7.0.5`. Several releases later we re-ran both pipelines —
same data, same train/dev split, same XGBoost search budget — on the current version
(`src/frequency_model_2026.ipynb`, `src/amount_model_2026.ipynb`). The 2026 search is
**seeded**, and so are the split and every estimator, so a clean checkout reproduces
the numbers below exactly.

**Three caveats up front, because they bound everything that follows.**

*The carving step is not the only thing that differs* — selection measures and the
severity model's `min_freq` moved too (details in Setup).

*2025 fed both models a wider candidate set,* so 2026 is the same pipeline on a narrower
one, not a like-for-like re-run (details in Setup).

*Only one side of each metric comparison is seeded* — the 2025 baseline was tuned by an
unseeded search, so speed comparisons are clean and metric comparisons are not (details
in Setup).

### 3.1 The slow step got fast: multiprocessing

Carving a multi-gigabyte dataset was the slowest step of our 2025 loop: every re-run cost
real competition time.

```python
# AutoCarver 7.7.3
from AutoCarver import OrdinalCarver
from AutoCarver.discretizers import ProcessingConfig

carver = OrdinalCarver(features=features, min_freq=0.02, max_n_mod=5, config=ProcessingConfig(n_jobs=6))
```

Carving wall-clock on the full dataset — 383,610 rows, ~435 qualitative features, same
machine, runs serialised:

| Pipeline | Carver | 7.0.5 (1 process) | today (6 workers) | speed-up |
|---|---|---|---|---|
| Frequency, qualitative | `MulticlassCarver` → `OrdinalCarver` | **2842 s** (47.4 min) | **120 s** (2.0 min) | **23.7×** |
| Frequency, quantitative | `MulticlassCarver` | 383 s (6.4 min) | *not carved — left to the selector* | — |
| Frequency, total | | **3226 s** (53.8 min) | **120 s** (2.0 min) | **26.9×** |
| Severity | `ContinuousCarver` | **4725 s** (78.7 min) | **140 s** (2.3 min) | **33.8×** |

**Where that factor comes from, honestly.** 7.0.5 was single-process and we ran today's
carve on 6 workers, so perfect scaling alone would give 6×. The remaining ~4× on frequency
and ~5× on severity is the dynamic-programming top-k search that replaced brute-force
enumeration — half the headline is your cores, the other half a better algorithm.

The severity carve is the heavier one, and in 2026 it ran at a *lower* `min_freq`
(0.02 vs 0.03) — thinner buckets, more work — so 33.8× is if anything conservative.
Unlike the accuracy numbers later on, these are not close calls: repeated carves on this
machine vary by ~16 %, and a 27× gap sits nowhere near that.

### 3.2 The right target geometry: `OrdinalCarver`

Our claim-count target isn't just multiclass — `0 < 1 < 2+` is **ordered**, and
`MulticlassCarver` ignores that ordering. AutoCarver now ships `OrdinalCarver`, which
optimises bins against an ordinal target using Kendall's tau-c (tau-b and Somers' D are
available too). The change is the one import above, in place of 2025's `MulticlassCarver`.

Two things changed at once, and they are worth separating. `OrdinalCarver` uses the
target's **order**, which `MulticlassCarver` cannot. It also produces **one carved column
per feature**, where 2025's one-vs-rest produced several — a narrower matrix for the same
information, which matters when a selection budget decides how many columns the model
ever sees (§3.4).

So we measured it. Three arms, same library, machine and split, one variable each: the
2025 `OneVsRestCarver` geometry, `MulticlassCarver` (target unordered), and
`OrdinalCarver` (target ordered). Scored on a **common yardstick** — Kendall's tau-c on
the **dev** set, computed after the fact for every carved column. Each carver optimises
its own measure, so those are not comparable; tau-c is defined for any ordered binning.

Carve time and column count below are over all 435 inputs; buckets and association are
matched on the **364** features every arm carved into two or more buckets:

| Arm | Carve (435) | Columns (435) | Buckets / feature (364) | Best \|tau-c\| per feature (364) |
|---|---|---|---|---|
| `OneVsRestCarver` *(2025)* | 482 s | **823** | 4.32 | 0.00132 |
| `MulticlassCarver` | 139 s | 420 | 2.20 | 0.00123 |
| `OrdinalCarver` *(2026)* | **127 s** | **397** | **2.13** | **0.00133** |

**The ordinal carver retains the same association per feature as the 2025 geometry and
charges a fraction of the price: 2.07× fewer columns, 2.05× fewer buckets** (834 against
1707) **and a carve 3.8× faster** — all three over the full 435 inputs. One-vs-rest
carves the feature once per class and hands
the model several views of it; the ordinal carver hands it one. One caveat: one-vs-rest
orders each carving against a **binary** target, so the direction of its buckets relative
to the *ordinal* target is arbitrary — absolute values are the only fair comparison.

The clean pair is `MulticlassCarver` against `OrdinalCarver`: both emit one column per
feature at roughly the same bucket count, differing only in whether the order is used.
Across the 364 features both carved, they return the **identical bucketing on 210**; on
the 154 where they disagree, the ordinal carver wins **111 to 43** (two-sided sign test,
p ≈ 4×10⁻⁸). The direction is not chance, but the size is small: mean per-feature
difference +0.000107 tau-c. Against one-vs-rest it is a **dead heat** — 200 identical,
84 to 80 on the rest (p ≈ 0.81), mean difference +0.000010. Matching a geometry that spends twice the columns and twice the buckets is
the win; beating it was never the claim. Declare your ordinals — the library can only
use an order you tell it about.

**One limit, stated plainly.** Each arm was carved and scored, but none was carried
through feature selection and the XGBoost search, so nothing here claims an effect on
the final dev metric.

### 3.3 Two things we would reach for next time, and did not use here

Neither is in the re-run — the parts of the current library we would have wanted in
2025, described as what they replace rather than as something measured here.

- **Nested features.** Our 2025 helper carries a hand-written département-to-region
  mapping, to roll thin département buckets up into something populated enough to model.
  `NestedFeature` declares that hierarchy instead of encoding it: rare modalities fall
  back to their parent until every surviving bucket clears `min_freq`. The hand-rolled
  dict still runs the notebooks, and is the before-picture.
- **LLM-assisted qualification (MCP).** The most tedious hour of 2025 was typing ~40
  feature declarations by hand. AutoCarver now ships a local
  [MCP](https://modelcontextprotocol.io) [4] server that proposes the whole
  qualification from the CSV. We did not re-qualify this dataset through it, so treat
  the time saving as an argument, not a measurement.

### 3.4 So, did it actually win harder?

The honest scoreboard — the two changes we actually put through the pipeline.
Everything else is listed in §4 rather than scored:

| Change | Effort | Frequency (dev metric impact) | Severity (dev metric impact) |
|--------|--------|-------------------------------|------------------------------|
| Multiprocessing + DP search | one config arg | **3226 s → 120 s (26.9×)**, metric unchanged | **4725 s → 140 s (33.8×)**, metric unchanged |
| `OrdinalCarver` (tau-c) | one line | **2.07× fewer columns, 2.05× fewer buckets, 3.8× faster** than the 2025 geometry, for the same association per feature (§3.2) — but a structural comparison only, **not carried through to the dev metric** | n/a — continuous target |

Overall, end to end:

| Metric | 2025 (7.0.5) | 2026 (current) | verdict |
|---|---|---|---|
| Frequency — dev log loss | 0.9194 | 0.9199 | **indistinguishable** — seed spread is 18× the gap (below) |
| Severity — dev RMSE | 6613.8 | 6469.7 | **indistinguishable** — the 2.18 % gap is 1.01× the seed spread (below) |
| **`CHARGE` — dev RMSE** *(the challenge metric)* | 6639.9 | 6482.8 | **indistinguishable** — the 2.37 % gap is 1.01× the seed spread (below) |

**That top row is not a result, and it took a deliberate experiment to find out.** Then
the same experiment came for the other two.

The two frequency numbers differ by 0.00055. Both come from a seeded 300-trial Optuna
search, and the 2026 run reproduces to sixteen significant figures — so it is tempting to
read the gap as real. It is not. We re-ran the *identical* feature set under four different
TPE seeds, changing nothing else:

| Seed | Dev log loss |
|---|---|
| 42 *(the one reported)* | 0.9199 |
| 1 | 0.9246 |
| 7 | 0.9207 |
| 2026 | **0.9145** |

**Spread: 0.0101 — eighteen times the gap we were about to interpret.** One seed lands
*above* the 2025 baseline, another lands *below* it. The sign of "did the 2026 frequency
model beat the 2025 one?" is decided by the random seed, not by the pipeline.

A seed makes a run **reproducible**, not **representative** — easy to confuse when the
number comes back identical every time. Tune with a Bayesian search, report a single run,
and you owe your reader that spread.

Same check, severity side — 400 trials each, identical features:

| Seed | Dev RMSE | Dev `CHARGE` RMSE |
|---|---|---|
| 42 *(the one reported)* | 6469.7 | 6482.8 |
| 1 | 6612.9 | 6638.1 |
| 7 | 6591.5 | 6615.0 |
| 2026 | 6608.9 | 6633.2 |

Dev RMSE moves by **143.2 across those seeds — 2.16 % of the 2025 figure**, and
end-to-end `CHARGE` by **155.2 (2.34 %)**. So the 2.18 % and 2.37 % margins are 1.01× and
1.01× their own noise: the margin and the noise are the same size. Worse for the claim,
**seed 42 — the one the notebook reports — is the best of the four on both metrics**, and
the other three land within 0.4 % of the 2025 baseline. Averaged over the four, the margin
is **0.65 %** on severity and **0.72 %** on `CHARGE`. **The frequency side is a draw, and
on this evidence so is the severity side: the run we reported is the luckiest of four.**

The `CHARGE` row needed reconstructing: the 2025 notebook only ever computed `CHARGE` on
the unlabelled submission sample, so the era we won with had no end-to-end dev score. That
figure comes from replaying the pipeline and scoring the **saved** models, re-fitting
nothing — a replay that reproduces the 2025 notebook's own severity number to four decimal
places (6613.8176 against a recorded 6613.82).

**Named plainly: nothing here bought accuracy.** Multiprocessing and the DP search are
pure speed; `OrdinalCarver` bought a narrower matrix and a faster carve, but those arms
are a structural comparison and **no dev-metric gain is claimed** over carving the target
as if unordered. Nor did we run the ablation with no carving at all, so §2.1's claim about
where the lift came from stays a belief.

**A note on selection budgets.** The two targets want opposite splits: frequency — mean
claim count 0.69 %, tuned into `max_depth=1` stumps — drowns when you pile on carved
qualitative features, each carrying up to five mostly-empty buckets, while severity is
heavy-tailed and those same features help it. An even split is a compromise, not an
optimum.

## 4. Also shipped since 7.0.5, and not exercised here

Capabilities this dataset gave us no honest way to test. None is claimed to have done
anything here.

- **`DatetimeFeature`** — temporal fields carve natively against the target. The CAA data
  has no date columns (`AN_EXERC` and `ANNEE_ASSURANCE` are integers).
- **`OrdinalSelector`** — selection with ordinal-target association measures. We kept
  `ClassificationSelector` throughout to mirror 2025.
- **Wilson-score bucket testing** — a thin bucket is now merged when its frequency is
  *significantly* below `min_freq`, not when it merely dips under on one sample. We could
  not show it working here: an interval's width is driven by sample size, and at 306,888
  rows it has collapsed onto the point estimate. Turning it off gave the same 397 columns,
  the same 38 rejected features and one bucket's difference. This sample is not small.

## 5. The smallest complete example

Ten lines against any binary target. The line doing the work is
`fit_transform(..., X_dev=dev, y_dev=dev["Survived"])`: §2.1's argument in one call.

```bash
pip install autocarver
```

```python
# AutoCarver 7.7.3
import pandas as pd
from sklearn.model_selection import train_test_split
from AutoCarver import BinaryCarver, Features

data = pd.read_csv("titanic.csv")
train, dev = train_test_split(data, test_size=0.33, stratify=data["Survived"], random_state=42)

features = Features(categoricals=["Sex"], numericals=["Age", "Fare"], ordinals={"Pclass": ["1", "2", "3"]})
carver = BinaryCarver(features=features, min_freq=0.05, max_n_mod=5)
train_carved = carver.fit_transform(train, train["Survived"], X_dev=dev, y_dev=dev["Survived"])
carver.summary   # your features, as auditable buckets
```

## 6. Takeaways

Earlier we asked how you'd handle a signal this rare. Our answer:

- **Factor the problem.** Frequency × severity beats one monolithic model here, and each
  half is easier to debug.
- **Bin like you mean it.** Supervised, dev-validated binning was the step we spent the
  most effort on and would keep first; we did not measure its lift in isolation. It is
  also the only part of the pipeline a reviewer can *read*. *Who decides where your
  features get cut today?*
- **Weight the rare signal** — in the loss and across stages. *Where does your pipeline
  let the majority class win?*
- **Tooling compounds — on speed.** A year of releases turned manual steps into
  one-liners and the slow step into a fast one. It did not move a metric.
- **Measure your seed spread before you read a margin.** Four extra runs per model turned
  the two apparent wins — severity and `CHARGE` — into draws. It is the cheapest honesty check in this article, and the
  one we would run first next time.

## Everything behind this article

- **The code**, every notebook stored alongside the outputs that produced these numbers —
  [github.com/mdefrance/caa-challenge](https://github.com/mdefrance/caa-challenge)
- **Run it without installing anything** — the
  [frequency](https://www.kaggle.com/code/mariodefrance/caa-frequency-model) and
  [severity](https://www.kaggle.com/code/mariodefrance/caa-amount-model) notebooks on Kaggle,
  against a [mirror of the data](https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025)
- **AutoCarver** — [docs](https://autocarver.readthedocs.io) ·
  [source](https://github.com/mdefrance/AutoCarver)

---

## Setup — exactly what was measured

Every number in §3 comes from four notebook runs on **one machine, serialised**
(08:36 → 12:46 on 2026-08-11).

| | |
|---|---|
| CPU / RAM | Intel Coffee Lake, 12 logical cores · 32 GB |
| GPU | GTX 1650 Max-Q — XGBoost runs `device="cuda"` |
| Python | 3.11.7 |
| **2025 arm** | AutoCarver **7.0.5**, scikit-learn 1.9.0, numpy 2.0.2, **single-process** |
| **2026 arm** | AutoCarver **7.7.3**, scikit-learn 1.8.0, numpy 2.4.6, **`n_jobs=6`**, Optuna seeded (`TPESampler(seed=42)`) |
| XGBoost / Optuna | 3.2.0 / 4.9.0 — 300 trials (frequency), 400 (severity), identical between eras |
| Data | ENS *Challenge Data* #161 [1] plus a BDIFF fire-history extract [3], both Etalab Licence Ouverte 2.0; the exact extract is mirrored on Kaggle |

**Known differences beyond the carver**, so you can discount them yourself: the 2026 arm
selects with the current library's **default measures** rather than the 2025 thresholds;
the 2026 severity model carves at `min_freq=0.02` where 2025 used `0.03`; and scikit-learn
and numpy differ by a minor version. The wall-clock comparison is unaffected — same
machine, same data, same carve.

**The 2026 candidate set is narrower than 2025's.** 2025 carved the quantitative features
a second time and let those buckets compete in the qualitative pool, and re-offered the
frequency model's carved columns to the severity model. Neither is reproduced here.

**The 2026 arm is seeded and the 2025 baseline is not.** The split and every
XGBoost estimator use a fixed seed in both eras, and the 2026 Optuna search adds
`TPESampler(seed=42)`, so a clean checkout reproduces the 2026 numbers exactly.
The 2025 numbers came from an unseeded search.

Reproducible is not the same as representative, so both searches were re-run under four
TPE seeds — `data/ab_arms/seed_variance.csv` for frequency,
`data/ab_arms/seed_variance_amount.csv` for severity. **Every accuracy figure in this
article is stated against its own measured seed spread**, and §3.4 reports what that does
to all three of them.

**The dev set is selection-contaminated** — 400 trials chose against it — so dev
`CHARGE` RMSE is optimistic rather than held out. Severity and `CHARGE` RMSE are
unweighted: the frequency-error weights of §2.3 enter the fit, not the score. Our placement was
on the ENS private leaderboard at the hackathon's close (see the introduction); that
challenge is still accepting submissions, so we quote no leaderboard figure, old or
current, anywhere in this article.

Every number is reproducible from PyPI releases; the `autocarver` floor is deliberate,
as the README explains.

---

## References

[1] Crédit Agricole Assurances, *AssurPrime : Saurez-vous prédire la prime d'assurance ?*
(2025), ENS Challenge Data #161 — https://challengedata.ens.fr/challenges/161

[2] M. Defrance, *AutoCarver* (2026), PyPI / GitHub —
https://github.com/mdefrance/AutoCarver

[3] Institut national de l'information géographique et forestière, *Base de Données sur les
Incendies de Forêts en France (BDIFF)* (2025), Ministère de l'Agriculture et de la
Souveraineté Alimentaire — https://bdiff.agriculture.gouv.fr

[4] Anthropic, *Model Context Protocol* (2024) — https://modelcontextprotocol.io

[5] G. Navas-Palencia, *OptBinning: The Python Optimal Binning library* (2025), v0.21.0 —
https://github.com/guillermo-navas-palencia/optbinning

[6] M. Couillaud, *Retour sur le hackathon de la Data Science Academy* (2025), LinkedIn —
https://www.linkedin.com/posts/myriam-couillaud-6885012_data-ia-innovation-activity-7343893386023649281-w-qQ

---

*With thanks to Crédit Agricole Assurances and ENS Challenge Data for an excellent public
competition, to the CY Tech Fintech students who worked through it with us, and to
Zacharie Buisson for building the winning solution with me.*

