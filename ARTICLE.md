# Which half of our winning model actually won?

*We spent a year believing supervised binning carried our insurance-competition entry.
Four seeds per arm and one shared feature set later, the credit belongs somewhere else.*

> By **Mario Defrance**, with **Zacharie Buisson**, who co-built the winning solution.
> Every number below is measured, and every notebook is stored with the outputs that
> produced it.

> *Disclosure: I am the author and maintainer of AutoCarver. Read the library sections
> with that in mind; every number is reproducible from the repository linked below.*

---

## The finish line, first

![SURFACE4: sixteen raw floor-area bands merged into two buckets at 1000 m², with each
band's share of the portfolio below](docs/hero_SURFACE4.svg)

*Tschuprow's T with the target, on held-out data: 0.0240 across the sixteen raw levels,
0.0358 across the two buckets — a 1.49× rise.*

That is one real feature from the challenge data. Floor area comes in sixteen bands, claim
frequency climbs roughly tenfold across them, and supervised binning cuts them **in two, at
1000 m²** — not for want of finer options, since ten of the bands clear the 2 % `min_freq`
floor, but because the carver's association measure ranked this cut above all of them
(`tools/make_hero_chart.py --explain`). The lower panel is about trust: every band from
4500 m² to 7000 m² covers under 2 % of the portfolio, so the most dramatic rates rest on a
handful of policies. Nobody picked that cut, and nobody had to defend it in a meeting. Do
that for 400 features and you have the quiet half of a winning model — the half
most pipelines settle with a `qcut`.

It took us to **first place** in the Crédit Agricole Assurances *Data Science Academy*
hackathon [6], out of more than 500 participants, ranked on the private leaderboard of the
public [ENS *Challenge Data* #161](https://challengedata.ens.fr/challenges/161) [1] as it
stood when the hackathon closed in spring 2025. That challenge is still open, so the live
leaderboard has moved on and nothing here is a leaderboard score. It doubled as the running
example in the Data Science course we taught at CY Tech.

A year later we re-ran the whole thing. The speed answer was easy: **133 minutes of carving
became 4.3**. The accuracy answer took three experiments and went badly for us. Every
year-on-year gain turned out smaller than the noise in our own tuning, and the belief
underneath all of it — that binning our features is what won — only half survived: binning
did beat not binning, but a rival library's bins did just as well, and the edge we had
credited to carving was really in **which features the selector kept**. This article is
that audit: four seeds per arm, every margin measured against its own spread, and one
feature set held constant to separate two steps we had always run together.

## 1. The problem

Insurers don't predict "will this policy have a claim and how much" in one shot. They
factor it, as actuaries have for decades:

```
expected cost  =  E[ number of claims ]  ×  E[ amount per claim ]
                  └──── frequency ────┘     └──── severity ────┘
```

The CAA challenge asked exactly this: predict **claim frequency** and **claim severity**
from anonymised policy and property features. Two things make it harder than it looks.
**Claims are rare** — almost every policy has zero, so a model that predicts "zero" for
everyone looks accurate and is useless. And **the targets behave nothing alike**: frequency
is a small count, severity a heavy-tailed amount, and one model for both blurs each.

Before reading on: **how would *you* handle a signal this rare?** Hold your answer against
what follows. So: two models.

## 2. The winning recipe (2025)

No giant model, no ensemble of ensembles. Three decisions did it: **factor the problem**
into frequency × severity, like actuaries do; **bin every feature against the target** with
the open-source [AutoCarver](https://github.com/mdefrance/AutoCarver) library (mine, per
the disclosure) [2]; and **weight the rare signal**. Only the first and third came through
this audit intact (§3.4).

### 2.1 The quiet workhorse: carve the features first

The least glamorous part, and the step we would defend first. Raw insurance features are a
mess: high-cardinality categoricals, ordinal levels with tiny populations, skewed numerics,
and `NaN`s that *mean something*. Throw those at a tree model and it spends its depth
rediscovering structure you could have handed it.

**Supervised binning** hands it that structure. AutoCarver keeps, for each feature, the
admissible grouping that maximises statistical association with the target — under a
minimum bin frequency, a cap on bins per feature, and (crucially) **validation on a
held-out dev set**, so a grouping that only works on train is rejected outright. Declare
your feature types once, carve everything in one `fit`:

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
`MulticlassCarver` carved **one-vs-rest**, so every raw feature came back as several
columns; that behaviour is `OneVsRestCarver` now. Every snippet in §2 is 2025 code, kept as
written; §3 is the current API.

**A word on [optbinning](https://github.com/guillermo-navas-palencia/optbinning).** The
obvious alternative [5]: where we search groupings heuristically, it states the merge of
CART prebins as a CP/MILP problem and solves that exactly, inside a time limit. We went a
different way for three reasons about our framing, not the library: we wanted every
grouping tested on a held-out sample, and `fit(x, y)` has no notion of one; our target is ordered,
where `MulticlassOptimalBinning` does not treat it so (§3.2); and its multiclass path is
numerical-only, while most of our 435 features are categorical or declared ordinals. Those
were reasons, not measurements. §3.4 measures them.

What that bought us: ordinals merged **in their natural order**; missing values **never
silently imputed**, since `dropna=True` carves `NaN` as its own association-scored bucket
(we ran `dropna=False` and let XGBoost route them natively — the point is that it is a
declared choice); buckets we could **read and defend**; and one uniform pipeline over
numeric, categorical and ordinal features. We then pre-selected by association and pruned
redundant features (`ClassificationSelector` does both in one pass).

### 2.2 Frequency: a weighted multiclass model

Almost no policy has more than two claims, so frequency became a **multiclass** problem
over `0 / 1 / 2+`. Inverse-frequency class weights stopped the model collapsing onto the
majority class, and the law of total expectation turns class probabilities back into an
expected frequency:

```
E[Y] = P(0)·E[Y|0] + P(1)·E[Y|1] + P(2+)·E[Y|2+]
```

Every log loss in this article is the **class-weighted** one, using those same
inverse-frequency weights — it is what the search minimised. Unweighted, a constant
base-rate model already scores 0.041; weighted, it scores 4.47, and a uniform ⅓ guess
scores 1.099. Read 0.92 against those.

### 2.3 Severity, and tuning honestly

The severity model only sees policies with claims. The trick we credit most, unmeasured in
isolation: **weight each observation by the frequency model's absolute error**, so it
concentrates where the first stage was wrong — a cheap, boosting-flavoured correction
across the two stages. XGBoost on top, tuned with Optuna, with one idea worth stealing: the
**number of pruned features was itself a hyper-parameter**.

Factor the problem, carve the features, weight the rare signal, tune the pruning.
**1st place in the hackathon.**

## 3. What a year of releases actually bought

We won with AutoCarver `7.0.5`. Several releases later we re-ran both pipelines — same
data, same train/dev split, same XGBoost search budget — on the current version
(`src/frequency_model_2026.ipynb`, `src/amount_model_2026.ipynb`), seeded throughout, so a
clean checkout reproduces the numbers below exactly.

Three caveats bound what follows, all detailed in Setup: the carver is not the only thing
that differs, since selection measures and the severity `min_freq` moved too; 2025 fed both
models a wider candidate set; and only the 2026 side is seeded, so the speed comparisons
are clean and the metric comparisons are not.

### 3.1 The slow step got fast: multiprocessing

Carving a multi-gigabyte dataset was the slowest step of our 2025 loop.

```python
# AutoCarver 7.7.3
from AutoCarver import OrdinalCarver
from AutoCarver.discretizers import ProcessingConfig

carver = OrdinalCarver(features=features, min_freq=0.02, max_n_mod=5, config=ProcessingConfig(n_jobs=6))
```

Carving wall-clock on the full dataset — 383,610 rows, ~435 qualitative features, one
machine, runs serialised:

| Pipeline | Carver | 7.0.5 (1 process) | today (6 workers) | speed-up |
|---|---|---|---|---|
| Frequency, qualitative | `MulticlassCarver` → `OrdinalCarver` | **2842 s** (47.4 min) | **120 s** (2.0 min) | **23.7×** |
| Frequency, quantitative | `MulticlassCarver` | 383 s (6.4 min) | *not carved — left to the selector* | — |
| Frequency, total | | **3226 s** (53.8 min) | **120 s** (2.0 min) | **26.9×** |
| Severity | `ContinuousCarver` | **4725 s** (78.7 min) | **140 s** (2.3 min) | **33.8×** |

**Where that factor comes from, honestly.** 7.0.5 was single-process and we ran today's
carve on 6 workers, so perfect scaling alone gives 6×. The remaining ~4× on frequency and
~5× on severity is the dynamic-programming top-k search that replaced brute-force
enumeration — half the headline is your cores, the other half a better algorithm. The 2026
severity carve also ran at a *lower* `min_freq` (0.02 vs 0.03), so 33.8× is conservative.
Unlike the accuracy numbers later on, these are not close calls: the same carve has come
back at 120.0, 126.8 and 129.4 s on this machine, a spread of 7.8 %, nowhere near a 27×
gap.

### 3.2 The right target geometry

Our claim-count target isn't just multiclass — `0 < 1 < 2+` is **ordered**, and 2025's
`MulticlassCarver` ignored that. `OrdinalCarver` optimises bins against an ordinal target
using Kendall's tau-c. Scored on a common yardstick, tau-c on the dev set, it matches the
2025 one-vs-rest geometry's association per feature while spending **2.07× fewer columns
and 2.05× fewer buckets, in a carve 3.8× faster**. Against the unordered multiclass
carving, which differs only in whether the order is used, it returns identical bucketings
on 210 of 364 features and wins 111 to 43 on the rest (two-sided sign test, p ≈ 4×10⁻⁸) — a
reliable direction, a small size: +0.000107 mean tau-c. Declare your ordinals. No arm here
was carried through selection and tuning, so none of it claims an effect on the final
metric.

Three other releases went untested for want of an honest test on this data
(`DatetimeFeature`, `NestedFeature`, Wilson-score bucket testing, which on 306,888 rows
changed one bucket), and a local [MCP](https://modelcontextprotocol.io) [4] server now
proposes feature declarations from the CSV, which we have not re-qualified this dataset
through.

### 3.3 So, did it actually win harder?

Two things changed in the carving step: multiprocessing and the DP search, which moved
only the clock, and `OrdinalCarver`, whose gain is structural and was never carried through
to the dev metric. End to end:

| Metric | 2025 (7.0.5) | 2026 (current) | verdict |
|---|---|---|---|
| Frequency — dev log loss | 0.9194 | 0.9199 | **indistinguishable** — seed spread 18× the gap |
| Severity — dev RMSE | 6613.8 | 6469.7 | **indistinguishable** — the 2.18 % gap is 1.01× the spread |
| **`CHARGE` — dev RMSE** *(the challenge metric)* | 6639.9 | 6482.8 | **indistinguishable** — the 2.37 % gap is 1.01× the spread |

**That top row is not a result, and it took a deliberate experiment to find out.** The two
frequency numbers differ by 0.00055, both from a seeded search that reproduces to sixteen
significant figures — so it is tempting to read the gap as real. It is not. We re-ran the
*identical* feature sets under four TPE seeds, changing nothing else:

| Seed | Frequency log loss | Severity RMSE | `CHARGE` RMSE |
|---|---|---|---|
| 42 *(reported)* | 0.9199 | **6469.7** | **6482.8** |
| 1 | 0.9246 | 6612.9 | 6638.1 |
| 7 | 0.9207 | 6591.5 | 6615.0 |
| 2026 | **0.9145** | 6608.9 | 6633.2 |

**Frequency spread: 0.0101 — eighteen times the gap we were about to interpret.** One seed
lands above the 2025 baseline, another below it, so the seed decides the sign. A seed makes a run **reproducible**, not **representative**, and the two are
easy to confuse when the number comes back identical every time.

Severity moves further. Dev RMSE spans **143.2 across those seeds, 2.16 % of the 2025
figure**, and `CHARGE` **155.2 (2.34 %)** — so the 2.18 % and 2.37 % margins are 1.01×
their own noise. Worse, **seed 42, the one the notebook reports, is the best of the four on
both**, and the other three land within 0.4 % of the 2025 baseline; averaged over the four
the margin is 0.65 % and 0.72 %. **Both sides are draws, and the run we published is the
luckiest of four.**

The `CHARGE` row had to be reconstructed: the 2025 notebook computed it only on the
unlabelled submission sample, so the era we won with had no end-to-end dev score.
`tools/replay_2025_charge.py` scores the **saved** 2025 models, re-fitting nothing, and
reproduces that notebook's severity figure to 3×10⁻⁶.

**Named plainly: nothing a year of releases added bought accuracy here.** Which leaves the
older, larger question, the one §2.1 asserted and never tested: was the binning worth
anything at all?

### 3.4 Three ways to bin, and one question underneath them

§2.1 gave reasons for carving, and reasons for not using optbinning. Reasons are cheap, so
we ran all three: the same pipeline end to end, both models, four seeds each, only the
binning step changed. The third arm carves nothing — ordinals rank-encoded from the
orderings the notebook already declares, categoricals integer-coded on train, numericals
untouched — the baseline a practitioner reaches for with no binning library, not a straw
man (`tools/ablation_matrix.py`).

The first of §2.1's three reasons turns out not to be a preference at all — handing
`BinningProcess` our three-class ordinal target fails outright:

```
ValueError: MulticlassOptimalBinning does not support categorical variables.
```

435 of the 555 frequency columns are categorical or declared ordinals, so that is the
dataset, not a corner case. The frequency arm therefore bins against the **binary collapse
`y > 0`**, optbinning's best-supported path for mixed types. Neither binner touches the
numericals: the pipeline leaves those to the selector.

![Frequency dev log loss, four seeds per arm. Each arm on its own hundred features:
AutoCarver 0.9199, no carving 0.9386, optbinning 0.9660, the three ranges clear of each
other. All three on the same hundred: optbinning 0.9136, AutoCarver 0.9199, no carving
0.9334, the two binners overlapping](docs/results_selection_flip.svg)

**Run the arms as pipelines and the differences are large; hold the feature set constant
and almost all of it disappears.** In the left panel each arm carves, selects its own 100
features, then tunes; in the right all three model the same 100 — AutoCarver's own
selection, in the order its selector ranked them — so only binning differs. AutoCarver
scores the same in both panels because that shared list *is* its own selection: one run,
not two.

On its own selection every pair separates with no overlap: AutoCarver over no carving by
2.45× the pooled seed spread, no carving over optbinning by 2.56×, AutoCarver over
optbinning by 3.50×. On the shared set, two findings.

**Binning earned its place.** Both binners beat carving nothing by 1.59× the pooled spread,
with no overlap, and each wins on all four seeds individually. §2.1 believed that for a
year without testing it; it holds.

**But the two libraries are indistinguishable as binners.** On identical features the gap
between them is 0.63 % of a log loss — 0.45× their pooled spread, ranges overlapping.
Equalising selection swings a 3.50× pipeline win by 0.05 log loss and reverses its sign, so
what separated the libraries here was not how they cut features but **which features their
selectors kept**. Note the asymmetry: the shared list is AutoCarver's own selection, and on
it optbinning bins the sixteen ordinals as nominal levels where the other two arms use the
declared order. An AutoCarver win there would have been weak evidence. It did not win.

Two more chances close nothing: rank-encoding the 238 ordinal bands, so it can use the
order its documentation prescribes, helps by 0.48× the spread (0.9660 → 0.9571), and
capping it at three bins to match AutoCarver's realised width *hurts*, at 0.9707.

![Severity on identical features: the three arms overlap on dev RMSE, while on top-decile
lift optbinning reaches 3.94 and AutoCarver 3.35 against no carving's
2.43](docs/results_severity_two_metrics.svg)

**Severity needs a different metric to say anything at all.** On the same 200 features dev
RMSE puts the arms at 6570.7, 6580.7 and 6612.8 — 0.09×, 0.58× and 0.89× of pooled spread,
so nothing separates — while a constant predictor, the training mean, scores 6617.6, within
47 of every one of them.

On **top-decile lift** — the mean claim among the 10 % of policies a model ranks highest,
over the overall mean, where no signal scores 1.00 — the picture changes. Both binners
clear the unbinned arm, optbinning decisively at 2.41× the pooled spread and AutoCarver at
0.74×, while the two binners draw at 0.44×. The same ordering as frequency, from a metric
RMSE could not see.

**Why RMSE is blind here.** It rewards overfitting on this target. The train-to-dev RMSE
ratio is 1.01 for every binned seed; the unbinned arm's seed 7 hit **0.31** — train RMSE
1994 against dev 6530 — and that memorising model posted the *best dev RMSE of its arm*,
while its lift stayed at 2.37.

![Top-decile lift against the tree depth the tuner chose, one point per seed: the badly
ranked seeds are the deep ones](docs/results_depth_mechanism.svg)

The mechanism is in the depths the tuner chose. Unbinned features let the search reach
depth 2, 3 or 9; binned features kept it at depth 1 on eleven of twelve seeds. Deep trees
fit the amounts and lose the ordering. **Binning acted as a regulariser**, and on a
heavy-tailed target that trade favours anyone who needs to know *which* policies are
expensive rather than how expensive. It is a bias, not a guarantee: AutoCarver's seed 2026
went to depth 4, and its lift collapsed to 2.17 when it did.

Three limits: one dataset and one downstream pipeline; the arms select the same features
only when forced to, which is why both comparisons are reported; and every severity arm
consumes the *same* frequency hand-off, so that comparison isolates the severity binner.

## 4. Takeaways

Earlier we asked how you'd handle a signal this rare. Our answer:

- **Factor the problem.** Frequency × severity beats one monolithic model here, and each
  half is easier to debug.
- **Bin like you mean it — then check which step actually paid.** Binning beat carving
  nothing by 1.59× the seed spread, but two libraries' bins were indistinguishable and the
  gap between the *pipelines* was four times the gap between the *binners* (§3.4). Ours won
  on feature selection, not on cuts.
- **A metric can be blind.** Dev RMSE could not separate any severity arm from predicting
  the mean, and preferred a model that overfit threefold. Top-decile lift separated them
  2.4× (§3.4).
- **Beat the dumbest baseline first.** Our severity model spent a year looking like it
  worked because 6613.8 dev RMSE reads like a number. The training mean scores 6617.6
  (§3.4).
- **Measure your seed spread before you read a margin.** Four extra runs per model turned
  two apparent wins into draws — the cheapest honesty check here, and the one we would run
  first next time.

## Everything behind this article

- **The code**, every notebook stored with the outputs that produced these numbers —
  [github.com/mdefrance/caa-challenge](https://github.com/mdefrance/caa-challenge)
- **Run it without installing anything** — the
  [frequency](https://www.kaggle.com/code/mariodefrance/caa-frequency-model) and
  [severity](https://www.kaggle.com/code/mariodefrance/caa-amount-model) notebooks on
  Kaggle, against a
  [mirror of the data](https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025)
- **AutoCarver** — [docs](https://autocarver.readthedocs.io/en/stable/) ·
  [source](https://github.com/mdefrance/AutoCarver)

---

## Setup — exactly what was measured

Every number in §3 comes from four notebook runs on **one machine, serialised**
(2026-08-11, 08:36 → 12:46).

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
selects with the current library's **default measures** rather than the 2025 thresholds; it
carves severity at `min_freq=0.02` where 2025 used `0.03`; scikit-learn and numpy differ by
a minor version; and 2025 fed both models a wider candidate set, carving the quantitative
features twice and re-offering the frequency model's carved columns to the severity model.
The wall-clock comparison is unaffected.

**The 2026 arm is seeded and the 2025 baseline is not.** The 2026 Optuna search adds
`TPESampler(seed=42)`; the 2025 numbers came from an unseeded search. Reproducible is not
representative, so both were re-run under four TPE seeds (`data/ab_arms/seed_variance.csv`,
`seed_variance_amount.csv`, and `optbinning_{frequency,amount}.csv` for §3.4). **Every
accuracy figure here is stated against its own measured seed spread.**

**The dev set is selection-contaminated** — 400 trials chose against it — so dev `CHARGE`
RMSE is optimistic rather than held out. Severity and `CHARGE` RMSE are unweighted: the
frequency-error weights of §2.3 enter the fit, not the score. No leaderboard figure, old or
current, is quoted anywhere in this article.

Every number is reproducible from PyPI releases, with one exception: the 2025 `CHARGE`
baseline needs the 2025 carvers and XGBoost models, build outputs of the original working
repo that run only under AutoCarver 7.0.5. `tools/replay_2025_charge.py` takes the path to
that checkout and writes its output and its control check to
`data/ab_arms/replay_2025_charge.json`.

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
competition, to the CY Tech students who worked through it with us, and to Zacharie Buisson
for building the winning solution with me.*

