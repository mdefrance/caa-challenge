"""Frequency arm of the optbinning-vs-AutoCarver ablation.

Same pipeline as ``src/frequency_model_2026.ipynb`` with **one** step swapped: where the
notebook carves the qualitative block with ``OrdinalCarver`` (and hands the numericals to
the selector raw), this harness bins *every* input column with optbinning's
``BinningProcess``. Everything downstream -- the AutoCarver ``ClassificationSelector``
budget of 100, ``src/objectives.py``, 300 Optuna trials, the 3-class ordinal target and
the inverse-frequency sample weights -- is the notebook's own code, unchanged.

Four TPE seeds (42, 1, 7, 2026), matching ``tools/seed_variance.py``, because the
AutoCarver baseline in ``data/ab_arms/seed_variance.csv`` spans 0.0101 log loss across
those same seeds: a single-seed comparison could not distinguish an arm from noise.

The binning target is the **binary collapse** ``y > 0``, not the 3-class ordinal.
``tools/probe_optbinning_multiclass.py`` records why: optbinning's multiclass path raises
``ValueError: MulticlassOptimalBinning does not support categorical variables``. The
binary collapse is optbinning's best-supported path for mixed-type data and is a choice
made in optbinning's favour. The *downstream* target stays the 3-class ordinal, identical
to the AutoCarver arm.

Env:
  ABLATION_SMOKE=1  20k-row subsample, <=40 variables, 3 trials, seed 42 only.
  CAA_REPO          repository root override.
"""

import contextlib
import io as _io
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "frequency_model_2026.ipynb")
OUT = os.path.join(REPO, "data", "ab_arms", "optbinning_frequency.csv")

SMOKE = os.environ.get("ABLATION_SMOKE") == "1"

SEEDS = [42] if SMOKE else [42, 1, 7, 2026]
N_TRIALS = 3 if SMOKE else 300
N_BEST_FEATURES = 20 if SMOKE else 100

# matched to the 2026 AutoCarver arm: OrdinalCarver's min_freq / max_n_mod defaults
MIN_PREBIN_SIZE = 0.02
MAX_N_BINS = 5
N_JOBS = 6

# the AutoCarver baseline this arm is compared against (data/ab_arms/seed_variance.csv)
AUTOCARVER_DEV = {42: 0.9199100734725815, 1: 0.9246143245272799, 7: 0.9207397238839483, 2026: 0.9145278708853299}


def exec_prefix() -> dict:
    """Exec the notebook's code cells up to (not including) the OrdinalCarver fit.

    The boundary is found by source content, not by index, so an edit upstream in the
    notebook cannot silently shift it. After the prefix the globals hold ``x_train`` /
    ``x_dev`` (raw, post-``Processor``), ``y_ord_train`` / ``y_ord_dev``, ``w_train`` /
    ``w_dev`` and the three column-type lists.
    """
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "OrdinalCarver(" in src and "fit_transform" in src:
            print(f"[prefix] stop before code cell {i} (OrdinalCarver fit_transform)", flush=True)
            break
        t0 = time.perf_counter()
        exec(compile(src, f"<cell {i}>", "exec"), g)
        print(f"[prefix] cell {i} ok ({time.perf_counter() - t0:.1f}s)", flush=True)
    for name in ("x_train", "x_dev", "y_ord_train", "y_ord_dev", "w_train", "w_dev",
                 "categorical_columns", "ordinal_columns", "numerical_columns"):
        if name not in g:
            raise RuntimeError(f"prefix did not define {name!r}; the notebook boundary moved")
    return g


def bin_with_optbinning(x_train, x_dev, y_binary, categoricals, numericals):
    """Fit BinningProcess on train only, return (label frames, code frames, meta).

    Two transforms of the same fitted process:

    * ``metric="bins"`` -- the bin *labels*, i.e. the carved-style categorical output the
      plan calls for. WoE is deliberately not used: it would hand optbinning a supervised
      numeric encoding the AutoCarver arm never gets.
    * ``metric="indices"`` -- the same partition as integer codes, which is what XGBoost
      consumes (the AutoCarver arm's carved output is likewise integer-coded, because
      ``BaseCarver._default_ordinal_encoding`` is True). ``metric_missing`` /
      ``metric_special`` are set to ``"empirical"`` so the missing bin gets its own index
      instead of colliding with bin 0, which is what the integer default would do.

    The two are asserted to be a bijection per column, so the codes carry exactly the
    information the labels do.
    """
    from optbinning import BinningProcess

    variable_names = list(categoricals) + list(numericals)
    binning_process = BinningProcess(
        variable_names=variable_names,
        categorical_variables=list(categoricals),
        min_prebin_size=MIN_PREBIN_SIZE,
        max_n_bins=MAX_N_BINS,
        n_jobs=N_JOBS,
    )

    t0 = time.perf_counter()
    binning_process.fit(x_train[variable_names], y_binary)
    fit_seconds = time.perf_counter() - t0
    print(f"[binning] BinningProcess.fit: {fit_seconds:.1f}s on {N_JOBS} workers, "
          f"{len(variable_names)} variables, {len(x_train)} rows", flush=True)

    t1 = time.perf_counter()
    labels_train = binning_process.transform(x_train[variable_names], metric="bins")
    labels_dev = binning_process.transform(x_dev[variable_names], metric="bins")
    label_seconds = time.perf_counter() - t1

    codes_train = binning_process.transform(
        x_train[variable_names], metric="indices",
        metric_special="empirical", metric_missing="empirical",
    )
    codes_dev = binning_process.transform(
        x_dev[variable_names], metric="indices",
        metric_special="empirical", metric_missing="empirical",
    )
    print(f"[binning] transform(metric='bins') train+dev: {label_seconds:.1f}s "
          f"-> {labels_train.shape}", flush=True)

    # labels <-> codes must be a bijection, or the modelled frame is not the binned frame
    mismatched = [c for c in labels_train.columns
                  if labels_train[c].nunique(dropna=False) != codes_train[c].nunique(dropna=False)]
    if mismatched:
        raise RuntimeError(f"label/code cardinality mismatch on {mismatched[:10]}")

    summary = binning_process.summary()
    meta = {
        "binning_fit_seconds": round(fit_seconds, 1),
        "binning_transform_seconds": round(label_seconds, 1),
        "n_input_features": len(variable_names),
        "n_binned_columns": labels_train.shape[1],
        "n_dropped_by_optbinning": len(variable_names) - labels_train.shape[1],
        "n_single_bin_columns": int((labels_train.nunique(dropna=False) == 1).sum()),
        "mean_bins_per_column": round(float(labels_train.nunique(dropna=False).mean()), 3),
        "optbinning_status_counts": dict(summary["status"].value_counts()),
    }
    return labels_train, labels_dev, codes_train, codes_dev, meta, summary


def main():
    g = exec_prefix()
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_ord_train, y_ord_dev = g["y_ord_train"], g["y_ord_dev"]
    w_train, w_dev = g["w_train"], g["w_dev"]

    categoricals = list(g["categorical_columns"]) + list(g["ordinal_columns"])
    numericals = list(g["numerical_columns"])

    if SMOKE:
        n_rows = 20_000
        n_dev = 5_000
        # w_train / w_dev are positional numpy arrays from train_test_split, so the
        # subsample has to be the first N rows of the frames, not a random draw
        x_train, x_dev = x_train.iloc[:n_rows], x_dev.iloc[:n_dev]
        y_ord_train, y_ord_dev = y_ord_train.iloc[:n_rows], y_ord_dev.iloc[:n_dev]
        w_train, w_dev = w_train[:n_rows], w_dev[:n_dev]
        categoricals, numericals = categoricals[:25], numericals[:15]
        print(f"[SMOKE] {len(x_train)} train / {len(x_dev)} dev rows, "
              f"{len(categoricals)} cat + {len(numericals)} num variables", flush=True)

    # ---- the one swapped step -------------------------------------------------------
    y_binary = (y_ord_train > 0).astype(int)
    print(f"[binning] binary collapse y>0: {int(y_binary.sum())} events / {len(y_binary)} rows "
          f"(3-class counts {dict(y_ord_train.value_counts().sort_index())})", flush=True)

    labels_train, labels_dev, codes_train, codes_dev, meta, summary = bin_with_optbinning(
        x_train, x_dev, y_binary, categoricals, numericals
    )
    print("[binning] " + " | ".join(f"{k}={v}" for k, v in meta.items()), flush=True)

    # Nothing else from the raw frames is needed downstream: the frequency notebook feeds
    # the selector and Optuna the feature block only, with y_ord_* and w_* carried
    # separately. The severity hand-off (frequency_2026.csv) is deliberately NOT rewritten
    # -- both severity arms consume the same AutoCarver hand-off so only the binning of
    # the severity features differs there.

    # ---- notebook cell 9's selector, verbatim config --------------------------------
    from AutoCarver import Features
    from AutoCarver.selectors import ClassificationSelector, CramervFilter, SelectionConfig, SpearmanFilter

    binned_columns = list(codes_train.columns)
    features = Features(categoricals=binned_columns)
    config = SelectionConfig(
        qualitative_filters=[CramervFilter(threshold=0.9)],
        quantitative_filters=[SpearmanFilter(threshold=0.9)],
    )
    selector = ClassificationSelector(features, n_best_features=N_BEST_FEATURES, config=config)
    t0 = time.perf_counter()
    selector.fit(codes_train, y_ord_train)
    select_seconds = time.perf_counter() - t0
    best_features = selector.selected_features.names
    meta["selection_seconds"] = round(select_seconds, 1)
    meta["n_selected"] = len(best_features)
    print(f"[selector] selected {len(best_features)}/{len(binned_columns)} "
          f"in {select_seconds:.1f}s", flush=True)

    # ---- the notebook's search, one run per seed ------------------------------------
    import optuna
    from objectives import get_best_multiclass_model, get_multiclass_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        objective = get_multiclass_objective(
            codes_train[best_features], y_ord_train,
            codes_dev[best_features], y_ord_dev,
            w_train=w_train, w_dev=w_dev,
        )
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=N_TRIALS)

        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            get_best_multiclass_model(
                study.best_params,
                codes_train[best_features], y_ord_train,
                codes_dev[best_features], y_ord_dev,
                w_train=w_train, w_dev=w_dev, train_on_full=False,
            )
        text = buf.getvalue()
        train = float(re.search(r"Log Loss Train:\s*([0-9.eE+-]+)", text).group(1))
        dev = float(re.search(r"Log Loss Dev:\s*([0-9.eE+-]+)", text).group(1))
        rows.append({
            "seed": seed,
            "log_loss_train": train,
            "log_loss_dev": dev,
            "max_depth": study.best_params.get("max_depth"),
            "n_estimators": study.best_params.get("n_estimators"),
            "minutes": round((time.perf_counter() - t0) / 60, 1),
        })
        ac = AUTOCARVER_DEV.get(seed)
        note = f"  (AutoCarver same seed {ac:.6f}, gap {dev - ac:+.6f})" if ac and not SMOKE else ""
        print(f"seed {seed:5d}: dev {dev:.10f}  train {train:.10f}  "
              f"depth {rows[-1]['max_depth']}  {rows[-1]['minutes']} min{note}", flush=True)

    df = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# arm: optbinning {_optbinning_version()} BinningProcess"
                 f" (min_prebin_size={MIN_PREBIN_SIZE}, max_n_bins={MAX_N_BINS},"
                 f" n_jobs={N_JOBS}, metric=bins/indices)\n")
        fh.write("# binning target: binary collapse y>0; downstream target: 3-class ordinal\n")
        fh.write(f"# n_trials: {N_TRIALS} | seeds: {SEEDS} | smoke: {SMOKE}\n")
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        df.to_csv(fh, index=False)

    summary.to_csv(os.path.join(REPO, "data", "ab_arms", "optbinning_frequency_binning_summary.csv"),
                   index=False)

    print("\n" + df.to_string(index=False))
    spread = df.log_loss_dev.max() - df.log_loss_dev.min()
    print(f"\ndev log loss: min {df.log_loss_dev.min():.6f} | max {df.log_loss_dev.max():.6f} "
          f"| mean {df.log_loss_dev.mean():.6f} | spread {spread:.6f} | std {df.log_loss_dev.std():.6f}")
    if not SMOKE:
        ac = pd.Series(AUTOCARVER_DEV)
        ac_spread = ac.max() - ac.min()
        gap = df.log_loss_dev.mean() - ac.mean()
        pooled = (spread + ac_spread) / 2
        print(f"AutoCarver: mean {ac.mean():.6f} | spread {ac_spread:.6f}")
        print(f"between-arm gap (optbinning - AutoCarver, mean of 4 seeds): {gap:+.6f}")
        print(f"pooled seed spread: {pooled:.6f}  -> gap = {gap / pooled:+.2f} x pooled spread")
    print(f"\nwrote {OUT}")


def _optbinning_version() -> str:
    import optbinning

    return optbinning.__version__


if __name__ == "__main__":
    main()
