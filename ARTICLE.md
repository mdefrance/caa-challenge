# Stop losing accuracy to manual binning: how we won an insurance data challenge by automating the quiet work of feature engineering

<!-- When this is cross-posted, set the platform copy's canonical URL to whichever
     version is the original. This file is the source of record. -->

> By **Mario Defrance**, with **Zacharie Buisson**, who co-built the winning solution.
> Every number below is measured, and the notebooks in this repository are stored with
> the outputs that produced them.

---

## The finish line, first

![SURFACE4, floor area: sixteen raw area bands whose claim count climbs roughly
tenfold, carved into two buckets at 1000 m². A second panel shows each band's share
of the portfolio on a log scale, with every band above 4500 m² falling under the 2 %
frequency floor](docs/hero_SURFACE4.svg)

That is one real feature from the challenge data, carved by the pipeline this
article describes. Floor area comes in sixteen bands; claim frequency climbs
roughly tenfold across them. Supervised binning cuts them **in two, at 1000 m²**,
and the second panel shows why it stops there: every band above 4500 m² covers
less than 2 % of the portfolio, so the rates that look most dramatic rest on a
handful of policies. Nobody picked that cut, and nobody had to defend it in a
meeting.

Do that for four hundred features and you have the quiet half of a winning
model — the half most pipelines leave to a default `qcut`, or to whoever last
had an opinion.

It took us to **first place** in the public [Crédit Agricole Assurances
challenge on insurance claim
prediction](https://challengedata.ens.fr/challenges/161). And when we re-ran
the pipeline a year later with today's tooling, **one config argument turned
47 minutes of carving into 2, and 79 minutes into 2.3** — while the end-to-end
metric improved. The rest of this article is how — and, just as
honestly, which shiny new features changed nothing at all.

## TL;DR

Last year, Zacharie Buisson and I finished **first** in the CAA challenge —
while using it as the running example in the Data Science course we taught to
final-year engineering students at CY Tech.

No giant model, no ensemble of ensembles. Three decisions did the work:

- **Factor the problem** into frequency × severity, like actuaries do.
- **Bin every feature against the target** with the open-source
  [AutoCarver](https://github.com/mdefrance/AutoCarver) library — this is
  where most of the lift came from.
- **Weight the rare signal**: class weights in the loss, error-weighted
  observations across the two stages.

A year later, AutoCarver has grown a lot. So we re-ran the challenge with the
current version and measured what each new feature is actually worth. The
numbers — including the ones that say "no difference" — are below.

```bash
pip install autocarver
```

---

## 1. The problem

Insurers don't predict "will this policy have a claim and how much" in one
shot. They factor it, the way actuaries have for decades:

```
expected cost  =  E[ number of claims ]  ×  E[ amount per claim ]
                  └──── frequency ────┘     └──── severity ────┘
```

The CAA challenge asked exactly this: predict **claim frequency** and **claim
severity** from anonymised policy and property features. Two things make it
harder than it looks:

1. **Claims are rare.** The overwhelming majority of policies have zero
   claims. A naive model predicts "zero" for everyone, looks accurate, and is
   useless — the signal lives in a thin tail.
2. **The targets behave nothing alike.** Frequency is a small count; severity
   is a heavy-tailed amount. One model for both blurs each.

Before reading on, it's worth asking yourself: **how would *you* handle a
signal this rare?** Oversample it? Reweight it? Trust the tree to find it?
Keep your answer in mind — comparing it against what follows is the fastest
way to get value out of this piece.

So: two models.

## 2. The winning recipe (2025)

### 2.1 The quiet workhorse: carve the features first

This was the least glamorous part and the biggest source of lift, so it goes
first.

Raw insurance features are a mess: high-cardinality categoricals, ordinal
levels with tiny populations, skewed numerics, and `NaN`s that *mean
something*. Throw those at a tree model and it spends its depth rediscovering
structure you could have handed it.

**Supervised binning** hands it that structure. AutoCarver searches the
admissible groupings of each feature's values and keeps the one that maximises
statistical association with the target — under a minimum bin frequency, a cap
on bins per feature, and (crucially) **validation on a held-out dev set**, so
a grouping that only works on train is rejected outright. Declare your feature
types once, carve everything in one `fit`:

```python
# AutoCarver 7.0.5, as written in 2025. Do not paste this into a current project —
# see §3 for the current equivalent, and the note below on MulticlassCarver.
from AutoCarver import Features, MulticlassCarver

features = Features(
    categoricals=["property_type", "region", ...],
    numericals=["insured_value", "surface", ...],
    ordinals={"quality_band": ["low", "mid", "high", "premium"]},
)

carver = MulticlassCarver(features=features, min_freq=0.02, max_n_mod=5)
train_carved = carver.fit_transform(train, train["n_claims"],
                                    X_dev=dev, y_dev=dev["n_claims"])

carver.summary  # every bucket: frequency, target rate, association
```

⚠️ **That snippet still runs today, and does something else.** In 7.0.5,
`MulticlassCarver` carved **one-vs-rest**: one binary carving per target class, so
every raw feature came back as several carved columns. That behaviour is
`OneVsRestCarver` now; `MulticlassCarver` means *one* carving per feature against
the full crosstab. Same import, different output — worth knowing before you copy
year-old code, here or anywhere else. Every snippet in §2 is 2025 code, kept as
written; §3 is the current API.

What that bought us on this dataset:

- ordinal features merged **in their natural order**, never collapsed by raw
  frequency;
- missing values **never silently imputed**: `dropna=True` carves `NaN` as its
  own bucket, association-scored like any other, which on insurance data is
  often real information. We ran `dropna=False` on this challenge and let
  XGBoost route missing values natively — the point is that it is a declared
  choice either way, not an accident of a default;
- buckets we could **read, sanity-check, and defend** — for every feature;
- one uniform pipeline over numeric, categorical and ordinal features.

We then pre-selected features by association with the target and pruned
redundant ones (`ClassificationSelector` does both in one pass).

### 2.2 Frequency: a weighted multiclass model

Almost no policy has more than two claims, so frequency became a **multiclass**
problem over `0 / 1 / 2+` claims. Inverse-frequency class weights stopped the
model from collapsing onto the majority class. To turn class probabilities
back into an expected frequency, the law of total expectation:

```
E[Y] = P(0)·E[Y|0] + P(1)·E[Y|1] + P(2+)·E[Y|2+]
```

Simple — and it's what lets a classifier serve a regression target cleanly.

### 2.3 Severity: spend capacity on the frequency model's mistakes

The severity model only sees policies with claims. The trick that moved the
needle: **weight each observation by the frequency model's absolute error**.
The severity model concentrates exactly where the first stage was wrong — a
cheap, boosting-flavoured correction across the two stages.

### 2.4 XGBoost, tuned honestly

XGBoost on top, tuned with Optuna. One idea worth stealing: the **number of
pruned features was itself a hyper-parameter** — Optuna decided how
aggressively to trim, jointly with depth, learning rate and regularisation.

Factor the problem, carve the features, weight the rare signal, tune the
pruning. **1st place.**

## 3. Re-running it with today's AutoCarver (2026)

We won with AutoCarver `7.0.5`. Several releases since landed squarely on pain
points we hit during the challenge, so we re-ran both pipelines — same data,
same train/dev split, same XGBoost search budget — on the current version and
measured what changed (`src/frequency_model_2026.ipynb` and
`src/amount_model_2026.ipynb`). The 2026 search is **seeded**, and so are the
split and every estimator, so a clean checkout reproduces the numbers below
exactly.

**Three caveats up front, because they bound everything that follows.**

*The carving step is not the only thing that differs.* The 2026 feature
*selection* runs on the current library's default measures rather than the 2025
thresholds, and the severity model carves at `min_freq=0.02` where 2025 used
`0.03`.

*2025 fed both models a wider candidate set,* in two specific ways we did not
reproduce. It carved the **quantitative** features a second time and let those
buckets compete in the qualitative pool; and it re-offered the **frequency
model's carved columns** to the severity model as a pool of their own. The 2026
pipeline carves one set per model and selects from it. So 2026 is the same
pipeline on a narrower candidate set, not a like-for-like re-run.

*Only one side of each metric comparison is seeded.* The 2025 baseline was tuned
by an unseeded search. Seeding 2026 makes 2026 reproducible; it does not make the
cross-era gap attributable. Speed comparisons are clean; metric comparisons carry
that caveat wherever they appear.

### 3.1 The slow step got fast: multiprocessing

Carving a multi-gigabyte dataset was the slowest step of our 2025 loop —
every re-run of the feature pipeline cost real competition time.

```python
# AutoCarver 7.7.2
from AutoCarver import OrdinalCarver
from AutoCarver.discretizers import ProcessingConfig

carver = OrdinalCarver(features=features, min_freq=0.02, max_n_mod=5,
                       config=ProcessingConfig(n_jobs=6))
```

Carving wall-clock, 7.0.5 vs today, on the full dataset — 383,610 rows, ~435
qualitative features, same machine, runs serialized so nothing competes for
cores:

| Pipeline | Carver | 7.0.5 (1 process) | today (6 workers) | |
|---|---|---|---|---|
| Frequency, qualitative | `MulticlassCarver` → `OrdinalCarver` | **2842 s** (47.4 min) | **120 s** (2.0 min) | **23.7×** |
| Frequency, quantitative | `MulticlassCarver` | 383 s (6.4 min) | *not carved — left to the selector* | — |
| Frequency, total | | **3226 s** (53.8 min) | **120 s** (2.0 min) | **26.9×** |
| Severity | `ContinuousCarver` | **4725 s** (78.7 min) | **140 s** (2.3 min) | **33.8×** |

One caveat on that table: the 2026 column was measured on an idle machine, and the
7.0.5 column comes from an earlier session. We know that matters, because running
the same 2026 notebooks under load stretched every carve by ~1.8× while producing
byte-identical output. Treat the ratios as the right order of magnitude, not as
four significant figures.

**Where that factor comes from, honestly.** 7.0.5 was single-process and we ran
today's carve on 6 workers, so perfect scaling alone would give 6×. The
remaining ~4× on frequency and ~5× on severity is the dynamic-programming top-k
search that replaced brute-force enumeration. Half the headline is "the library
now uses your cores"; the other half is a better algorithm. Both are real, and
neither is the whole story on its own.

The severity model is the heavier carve — a heavy-tailed continuous target over
the same ~435 qualitative features — so it is where the wall-clock difference
shows up most. It also carved at a *lower* `min_freq` in 2026 (0.02 vs 0.03),
which admits thinner buckets and means more work, so 31.3× is if anything a
conservative read.

Unlike the accuracy numbers later in this article, these are not close calls:
repeated carves on this machine vary by ~16 %, and a 27× gap sits nowhere near
that.

Faster carving isn't a vanity metric: it means more iterations of the
feature/model loop inside the same competition window. In 2025, one full carve
of both pipelines cost over two hours; it now costs four and a half minutes.

### 3.2 The right target geometry: `OrdinalCarver`

Our claim-count target isn't just multiclass — `0 < 1 < 2+` is **ordered**,
and `MulticlassCarver` ignores that ordering. AutoCarver now ships
`OrdinalCarver`, which optimises bins against an ordinal target using
Kendall's tau-c (tau-b and Somers' D are available too). The change is one
line:

```python
# AutoCarver 7.7.2
from AutoCarver import OrdinalCarver   # 2025 used MulticlassCarver, i.e. one-vs-rest

carver = OrdinalCarver(features=features, min_freq=0.02, max_n_mod=5)
```

Two things changed at once here, and they are worth separating. `OrdinalCarver`
uses the target's **order**, which `MulticlassCarver` cannot. It also produces
**one carved column per feature**, where the one-vs-rest carving of 2025 produced
several — a denser feature matrix for the same information, which matters when a
selection budget decides how many columns the model ever sees (§3.5).

So we measured it. Three arms, same library, same machine, same split, one
variable each: the 2025 `OneVsRestCarver` geometry, `MulticlassCarver` (one
carving per feature, target treated as unordered), and `OrdinalCarver` (one
carving per feature, target ordered). Scored on a **common yardstick** — Kendall's
tau-c on the **dev** set, computed after the fact against `0 < 1 < 2+` for every
carved column, whichever carver produced it. Each carver optimises its own
internal measure, so those are not comparable; tau-c is defined for any ordered
binning, so it is.

Matched on the 364 features all three arms carved:

| Arm | Carve | Columns | Buckets / feature | Best \|tau-c\| per feature |
|---|---|---|---|---|
| `OneVsRestCarver` *(2025)* | 482 s | **823** | 4.32 | 0.00132 |
| `MulticlassCarver` | 139 s | 420 | 2.20 | 0.00123 |
| `OrdinalCarver` *(2026)* | **127 s** | **397** | **2.13** | **0.00133** |

**The ordinal carver retains the same association per feature as the 2025 geometry,
and charges a fraction of the price for it: 2.07× fewer columns** (397 against 823),
**2.05× fewer buckets** (834 against 1707) **and a carve 3.8× faster** (127 s against
482 s), on the same 435 input features and the same library. One-vs-rest carves the
feature once per class and hands the model several views of it; the ordinal carver
hands it one.

**One caveat the table needs.** One-vs-rest orders each carving against a
**binary** target — `y=1` against the rest, then `y=2` against the rest — so the
direction of its buckets relative to the *ordinal* target is arbitrary, and a
large share of its columns score negative against it. Absolute values are the
only fair comparison, and that is what the table reports.

The clean pair is `MulticlassCarver` against `OrdinalCarver`: both emit one column
per feature at roughly the same bucket count, differing only in whether the target's
order is used. Across the 364 features both carved, they return the **identical
bucketing on 210** of them. On the 154 where they disagree, the ordinal carver wins
**111 to 43**. Using the order helps, and it never costs much when it does not: the
mean per-feature difference is +0.000107 tau-c.

Against one-vs-rest the association comparison is a **dead heat** — 200 identical,
84 to 80 on the rest, a mean difference of +0.000010. That is the point. Matching a
geometry that spends twice the columns and twice the buckets is the win; beating it
was never the claim.

So the honest answer to "did the ordinal geometry pay?": it buys a feature matrix
**half the width**, carved in a **fraction of the time**, for the same information
per feature. Declare your ordinals — the library can only use an order you tell it
about.

**One limit, stated plainly.** This is a structural comparison. Each arm was
carved and scored, but none was carried through feature selection and the XGBoost
search, so nothing here claims an effect on the final dev metric. What a different
carving geometry does to a tuned model is a separate experiment, and we did not
run it.

### 3.3 Statistically honest rare buckets

Rare-claim data means thin buckets, and a thin bucket can look great on train
by pure luck — right up until it drifts in production. AutoCarver now tests
bucket frequencies with a **Wilson score confidence interval** instead of a
hard cutoff: a bucket is merged when its frequency is *significantly* below
`min_freq`, not when it merely dips under a threshold on one sample. Combined
with the held-out dev validation that was already there, no grouping survives
on one lucky draw.

For anyone who has to stand behind a model in front of a validator, an
auditor, or next quarter's data, this is the quiet feature that matters most:
buckets that are stable by construction, not by hope.

We measured this one too, and the answer is **no measurable difference — on a
dataset this size.** Setting `min_freq_alpha=1.0` drives the z-score to zero, which
collapses the Wilson upper bound onto the raw proportion: the old hard cutoff,
exactly. So the arm is one argument.

| | Wilson CI on | Hard cutoff |
|---|---|---|
| Features rejected | 38 | 38 — the identical set |
| Carved columns | 397 | 397 |
| Buckets, total | 848 | 847 |

Per feature: **394 of 397 identical bucket counts**, a net difference of **one
bucket** across the whole dataset, and a dev tau-c difference of −0.000004.

**The reason is worth more than the result.** A confidence interval's width is
driven by sample size, and our training set has 306,888 rows. Take a bucket
sitting at 1.9 % against a 2 % threshold:

| Rows | Wilson upper bound | Kept, or merged? |
|---|---|---|
| **306,888** *(our data)* | 0.0195 | merged — same as a hard cutoff |
| 10,000 | 0.0219 | kept |
| 1,000 | 0.0295 | kept |
| 300 | 0.0416 | kept |

At 300k rows the interval has already collapsed onto the point estimate, so the
test can only disagree with a raw comparison on a knife edge — which is precisely
the three features it moved. **Wilson-score bucket testing is a small-sample
feature.** It would earn its place on a 10,000-row book, or on a segment you
carved down to a few thousand policies. On this one it is doing nothing, and we
would rather tell you when a feature is idle than imply it is working.

The severity side is unmeasured as its own arm; its continuous carve rejected
**1 of 431** features. We expect the same conclusion for the same reason — same
306,888 rows.

We did watch `min_freq` matter, though, in a way worth repeating: dropping the
severity threshold from `0.03` to `0.02` changed which features survived
carving at all (`ZONE` went from rejected to retained). Bucket-frequency
policy is not a cosmetic knob.

### 3.4 Less plumbing we used to hand-roll

- **Nested features** — hierarchies like *département → region* are declared,
  not hand-encoded. Our 2025 helper module still carries a hand-written mapping
  of every département to its region, there to roll thin département buckets up
  into something populated enough to model. `NestedFeature` derives that rollup
  from the data instead: rare modalities of the fine column fall back to their
  parent, level by level, until every surviving bucket clears `min_freq`.
- **LLM-assisted qualification (MCP)** — in 2025, the single most tedious
  hour of the challenge was typing out ~40 feature declarations by hand:
  squinting at value counts, deciding numeric vs categorical vs ordinal,
  getting the ordinal orderings right, fixing the typos. In 2026, AutoCarver
  ships a local [MCP](https://modelcontextprotocol.io) server: point an
  MCP-aware assistant at the CSV and it proposes the full qualification in
  about the time it takes to pour a coffee — you pair up with the AI on the
  boring work and keep the judgment call. What was an hour of typing is now a
  thirty-second review. Runs entirely on your machine; your data goes
  nowhere.

### 3.5 So, did it actually win harder?

The honest scoreboard — including the rows where the answer is "it's just
nicer":

| Change | Effort | Frequency (dev metric impact) | Severity (dev metric impact) |
|--------|--------|-------------------------------|------------------------------|
| Multiprocessing + DP search | one config arg | **speed 3226 s → 120 s (26.9×)**, metric unchanged | **speed 4725 s → 140 s (33.8×)**, metric unchanged |
| `OrdinalCarver` (tau-c) | one line | **2.07× fewer columns, 2.05× fewer buckets, 3.8× faster** than the 2025 one-vs-rest geometry, for the same association per feature (§3.2). **Not carried through to the dev metric** — the arms are a structural comparison only | n/a — continuous target |
| Wilson-CI bucket testing | default on | **no measurable change at this sample size** — 394/397 features identically bucketed, net 1 bucket, dev tau-c −0.000004. A small-sample feature; 306,888 rows collapse the interval | **no measured arm** — 1/431 rejected; same reasoning applies |
| Datetime features | small refactor | **n/a** — this dataset has no date columns | same |
| Nested features | small refactor | applicable (*département → region*), **no isolated arm** | same |
| MCP-assisted qualification | workflow | **~1 h → ~30 s** of human typing — our estimate of our own effort, not a benchmark | same |

Overall, end to end:

| Metric | 2025 (7.0.5) | 2026 (current) | |
|---|---|---|---|
| Frequency — dev log loss | **0.9194** *(0.9134 tuned)* | 0.9199 | **indistinguishable** — see below |
| Severity — dev RMSE | **6613.8** *(6610.1)* | **6469.7** | ⬆ **2.18 % better** |
| **`CHARGE` — dev RMSE** *(the challenge metric)* | **6639.9** | **6482.8** | ⬆ **2.37 % better** |

**That top row is not a result, and it took a deliberate experiment to find out.**

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

A seed makes a run **reproducible**. It does not make it **representative**, and those are
easy to confuse when the number comes back identical every time. If you tune with a
Bayesian search and report a single run, measure your own seed spread before you interpret
anything smaller than it.

The severity and `CHARGE` margins — 2.18 % and 2.37 % — sit two orders of magnitude above
that noise floor, so those stand. **The 2026 pipeline wins end to end on the challenge's
own metric, carried by the severity side, and the frequency side is a draw.**

That last row needed reconstructing. The 2025 notebook only ever computed `CHARGE` on the
submission sample, where there are no labels — so the era we actually won with had no
end-to-end dev score at all. The 2025 figure comes from replaying that pipeline and scoring
the **saved** models, re-fitting nothing. The check that it is honest: the replay
reproduces the 2025 notebook's own severity numbers to four decimal places (6613.8176
against a recorded 6613.82). Both eras' `CHARGE` figures use the identical construction,
`expected count × predicted amount`, each with its own frequency hand-off.

**The single biggest mover is speed, by a wide margin, and it is the only
result here that isn't a close call.** 26.2× and 31.3× sit far outside the ~16 %
run-to-run variation we measured; the accuracy numbers do not. Frequency came
out 0.35 % better, severity 0.04 % better, the end-to-end challenge metric
0.07 % better — and not one of those deltas survives contact with an unseeded
300/400-trial search plus the input-width difference from §3. We measured
**0.0072** of log-loss spread from re-running an identical configuration; the
frequency gap to 2025 is half that. Note also that both eras' dev
scores were used for tuning, so all of them are optimistic; the comparison is
fair because the contamination is symmetric, not because it is absent.

**Named plainly, the things that did not buy accuracy on this dataset.**
Multiprocessing and the DP search are pure speed — they compute the same
groupings faster, and we would not expect them to move a metric. `OrdinalCarver`
bought a feature matrix half the width and a carve four times faster, but **no
measured gain in dev association** over carving the target as if unordered.
Wilson-CI bucket testing changed **one bucket** across 397 features — it is a
small-sample mechanism and this sample is not small. Nested features have no
isolated arm here, and this dataset has no datetime fields at all. Use them for
the reasons in §3.2–3.4 — correctness, stability, a narrower matrix, less
hand-rolled plumbing — not because this article proved they score better.

**A selection budget is not a single knob, and the two targets disagree about it.**
A total budget is split across feature types, and *how* it is split matters more
than its size. Frequency is a **0.69 % positive-rate** target, and every tuned
model here was a `max_depth=1` stump ensemble: each carved qualitative feature
adds up to five mostly-empty buckets, and piling them on drowns a rare signal.
The severity target is heavy-tailed and continuous, and it wants the opposite —
carved qualitative features help it.

We measured both directions. Tilting the same budget toward carved qualitative
features costs the frequency model real log loss and gains the severity model
real RMSE; tilting it back reverses both. There is no single split that is right
for both targets, which is worth knowing before you accept whatever apportionment
your selector happens to use.

**The same change made severity worse.** It traded 58 carved qualitative features
for 58 raw quantitative ones and gave back 35 RMSE points. The two targets want
opposite things: a rare-event classification target is hurt by piling on carved
qualitative features, and a heavy-tailed continuous target is helped by them.
There is no single split that is right for both.

**And the challenge metric moved the wrong way.** `CHARGE` is expected count ×
predicted amount, severity dominates its magnitude, and severity's regression
outweighs frequency's gain: 6600.8 → 6630.9. The arm under inspection improved
and the number that actually decides the competition got slightly worse. All
three 2026 configurations still beat 2025 end-to-end, so the year-on-year
comparison holds either way — but it takes all three metrics to see that
improving one of them was not free.

The lesson isn't "AutoCarver got worse" — the carving is fine, and the same
carved features score 0.9200 when you feed the model the right mix. It's that
**a selection budget is not a single knob**: the qualitative/quantitative balance
can matter more than the total, it pulls in opposite directions on different
targets, and a two-stage pipeline will happily let you optimise one stage into a
worse end-to-end result.

## 4. Takeaways — and the same questions, back to you

Earlier we asked how you'd handle a signal this rare. Here's our answer,
condensed — hold it against yours:

- **Factor the problem.** Frequency × severity beats one monolithic model on
  insurance data, and each half is easier to debug. *Does your problem have a
  natural factorisation you're currently ignoring?*
- **Bin like you mean it.** Supervised, association-maximising, dev-validated
  binning gave us more lift than any amount of model tuning. It's also the
  only part of the pipeline a regulator or a reviewer can *read*. *Who — or
  what — decides where your features get cut today?*
- **Weight the rare signal** — in the loss (class weights) and across stages
  (error-weighted severity). *Where does your pipeline silently let the
  majority class win?*
- **Tooling compounds.** A year of releases turned our manual steps into
  one-liners and the slow step into a fast one. Pick tools that keep moving.

If your answers differ from ours, that's the interesting part — the
comparison is the takeaway.

## Also shipped since 7.0.5, and not exercised here

Two capabilities landed in the library that this dataset gave us no honest way to
test, so they get a mention rather than a row in the scoreboard:

- **`DatetimeFeature`** — temporal fields carve natively, against the target,
  instead of hand-rolled epoch arithmetic. The CAA data has no date columns
  (`AN_EXERC` and `ANNEE_ASSURANCE` are integers), so there was nothing here to
  carve.
- **`OrdinalSelector`** — feature selection with ordinal-target association
  measures. We kept `ClassificationSelector` throughout to mirror 2025.

Both are documented; neither is claimed to have done anything for this challenge.

## Try it on your own features

Ten lines against any binary target — here, the Titanic:

```python
# AutoCarver 7.7.2
import pandas as pd
from sklearn.model_selection import train_test_split
from AutoCarver import BinaryCarver, Features

data = pd.read_csv("titanic.csv")
train, dev = train_test_split(data, test_size=0.33, stratify=data["Survived"], random_state=42)

features = Features(categoricals=["Sex"], numericals=["Age", "Fare"],
                    ordinals={"Pclass": ["1", "2", "3"]})
carver = BinaryCarver(features=features, min_freq=0.05, max_n_mod=5)
train_carved = carver.fit_transform(train, train["Survived"], X_dev=dev, y_dev=dev["Survived"])
print(carver.summary)   # your features, as auditable buckets
```

Docs and worked notebooks: [autocarver.readthedocs.io](https://autocarver.readthedocs.io) ·
Source: [github.com/mdefrance/AutoCarver](https://github.com/mdefrance/AutoCarver) —
if it earns a place in your pipeline, a ⭐ helps others find it.

Full challenge code: `github.com/…/caa-challenge-frequency` (this repo) ·
Runnable notebook on Kaggle: *(link pending — repo URL and Kaggle notebook are
both blocked on the publish-target decision)*.

---

## Setup — exactly what was measured

Every number in §3 comes from four notebook runs on **one machine, serialized**
(08:36 → 12:46 on 2026-08-11), so no two runs competed for cores.

| | |
|---|---|
| CPU / RAM | Intel Coffee Lake, 12 logical cores · 32 GB |
| GPU | GTX 1650 Max-Q — XGBoost runs `device="cuda"` |
| Python | 3.11.7 |
| **2025 arm** | AutoCarver **7.0.5**, scikit-learn 1.9.0, numpy 2.0.2, **single-process** |
| **2026 arm** | AutoCarver **7.7.2**, scikit-learn 1.8.0, numpy 2.4.6, xgboost 3.2.0, optuna 4.9.0, **`n_jobs=6`**, Optuna seeded (`TPESampler(seed=42)`) |
| XGBoost / Optuna | 3.2.0 / 4.9.0 — 300 trials (frequency), 400 (severity), identical between eras |

Known differences beyond the carver, stated so you can discount them yourself:
the 2026 arm selects features with the current library's **default measures**
rather than the 2025 thresholds; the 2026 severity model carves at
`min_freq=0.02` where 2025 used `0.03`; scikit-learn and numpy differ by a
minor version between the two environments; and 2025 fed both models a wider
feature matrix (§3). The wall-clock comparison is unaffected by all of these —
it is the same machine, same data, same carve.

Three more limits on the accuracy numbers.

**The 2026 arm is seeded and the 2025 baseline is not.** The split and every
XGBoost estimator use a fixed seed in both eras, and the 2026 Optuna search adds
`TPESampler(seed=42)`, so a clean checkout reproduces the 2026 numbers exactly.
The 2025 numbers came from an unseeded search.

Reproducible is not the same as representative. We measured the frequency search's
own seed spread — same features, same 300 trials, four seeds — at **0.0101 dev log
loss**, which is why §3.5 calls the frequency arms a draw rather than reading their
0.00055 difference as a result. **No accuracy claim in this article rests on a
margin smaller than that**, and the two that remain (severity 2.18 %, `CHARGE`
2.37 %) clear it by two orders of magnitude.

**The 2026 candidate set is narrower than 2025's.** 2025 carved the quantitative
features a second time and let those buckets compete in the qualitative pool, and
it re-offered the frequency model's carved columns to the severity model as a pool
of their own. Neither is reproduced here (§3). Same pipeline, fewer candidates.

**The dev set is selection-contaminated** — 400 trials chose against it — so dev
`CHARGE` RMSE is optimistic rather than held out; train and dev are reported
together so the gap is visible. **The ENS leaderboard cannot be re-submitted, so
no leaderboard delta is claimed anywhere in this article.**

Every number here is reproducible from released versions on PyPI — no patched
checkout required. The repository pins a floor on `autocarver` deliberately:
earlier releases still import and run, but they apportion a selection budget
across feature types differently (§3.5) and rank ordinal candidate groupings
differently (§3.2). They fail silently rather than loudly — same code, different
numbers.

---

*With thanks to Crédit Agricole Assurances and ENS Challenge Data for running
an excellent public competition, to the final-year CY Tech Data Science
students who worked through it with us, and to Zacharie Buisson for building
the winning solution with me.*

