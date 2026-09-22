"""Severity arm of the optbinning-vs-AutoCarver ablation.

Same pipeline as ``src/amount_model_2026.ipynb`` with **one** step swapped: where the
notebook carves the qualitative block with ``ContinuousCarver`` (and hands the numericals
to the selector raw), this harness bins *every* input column with optbinning's
``BinningProcess``. The rest -- the frequency hand-off join, the boosting-flavoured
weights, the ``RegressionSelector`` budget of 200, ``src/objectives.py``, 400 Optuna
trials with ``reg:tweedie`` -- is the notebook's own code.

No target collapse is needed here: optbinning's continuous path
(``ContinuousOptimalBinning``) *does* accept ``categorical_variables``, unlike its
multiclass path (see ``tools/probe_optbinning_multiclass.py``). The binning target is
therefore the raw continuous ``CM``, exactly as in the AutoCarver arm.

Four TPE seeds (42, 1, 7, 2026), matching ``tools/seed_variance_amount.py``.

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
NB = os.path.join(REPO, "src", "amount_model_2026.ipynb")
OUT = os.path.join(REPO, "data", "ab_arms", "optbinning_amount.csv")

SMOKE = os.environ.get("ABLATION_SMOKE") == "1"

SEEDS = [42] if SMOKE else [42, 1, 7, 2026]
N_TRIALS = 3 if SMOKE else 400
N_BEST_FEATURES = 20 if SMOKE else 200

MIN_PREBIN_SIZE = 0.02
MAX_N_BINS = 5
N_JOBS = 6

FREQ_VERSION = "2026"
# columns the notebook's cell 7 join brings in and cells 8/14 then consume
HANDOFF_COLUMNS = ["pred_1", "pred_2", "pred_sum", "FREQ", "CHARGE", "ANNEE_ASSURANCE"]

# the AutoCarver baseline this arm is compared against (data/ab_arms/seed_variance_amount.csv)
AUTOCARVER_DEV_RMSE = {42: 6469.6972025111745, 1: 6612.8590125244245,
                       7: 6591.495195675813, 2026: 6608.91158121388}
AUTOCARVER_DEV_CHARGE = {42: 6482.842161889131, 1: 6638.054627204837,
                         7: 6614.95599811788, 2026: 6633.158545260598}


def exec_prefix() -> dict:
    """Exec the notebook's code cells up to (not including) the ContinuousCarver fit."""
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "ContinuousCarver(" in src and "fit_transform" in src:
            print(f"[prefix] stop before code cell {i} (ContinuousCarver fit_transform)", flush=True)
            break
        t0 = time.perf_counter()
        exec(compile(src, f"<cell {i}>", "exec"), g)
        print(f"[prefix] cell {i} ok ({time.perf_counter() - t0:.1f}s)", flush=True)
    for name in ("x_train", "x_dev", "y_train", "y_dev", "target",
                 "categorical_columns", "ordinal_columns", "numerical_columns"):
        if name not in g:
            raise RuntimeError(f"prefix did not define {name!r}; the notebook boundary moved")
    return g


def bin_with_optbinning(x_train, x_dev, y_continuous, categoricals, numericals):
    """Fit BinningProcess on train only; return labels, integer codes and metadata.

    Identical contract to the frequency harness: ``metric="bins"`` is the reported
    carved-style output, ``metric="indices"`` (with empirical missing/special bins, so
    "Missing" does not collide with bin 0) is what XGBoost consumes, and the two are
    checked to be a per-column bijection.
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
    binning_process.fit(x_train[variable_names], y_continuous)
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


def charge_rmse(frame, target, xgb, selected_features, rmse):
    """Cell 14's end-to-end CHARGE score: expected claim count x predicted amount."""
    true_charge = (
        frame["CHARGE"] if "CHARGE" in frame.columns else target.loc[frame.index, "CHARGE"]
    )
    pred_charge = frame["pred_sum"] * xgb.predict(frame[selected_features])
    return rmse(true_charge, pred_charge)


def main():
    g = exec_prefix()
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_train, y_dev = g["y_train"], g["y_dev"]
    target = g["target"]

    categoricals = list(g["categorical_columns"]) + list(g["ordinal_columns"])
    numericals = list(g["numerical_columns"])

    if SMOKE:
        n_rows, n_dev = 20_000, 5_000
        x_train, x_dev = x_train.iloc[:n_rows], x_dev.iloc[:n_dev]
        y_train, y_dev = y_train.iloc[:n_rows], y_dev.iloc[:n_dev]
        categoricals, numericals = categoricals[:25], numericals[:15]
        print(f"[SMOKE] {len(x_train)} train / {len(x_dev)} dev rows, "
              f"{len(categoricals)} cat + {len(numericals)} num variables", flush=True)

    # ---- the one swapped step -------------------------------------------------------
    print(f"[binning] continuous target CM: mean {y_train.mean():.2f}, "
          f"max {y_train.max():.2f}, {int((y_train > 0).sum())} non-zero", flush=True)
    labels_train, labels_dev, codes_train, codes_dev, meta, summary = bin_with_optbinning(
        x_train, x_dev, y_train, categoricals, numericals
    )
    print("[binning] " + " | ".join(f"{k}={v}" for k, v in meta.items()), flush=True)
    _ = labels_train, labels_dev

    # ---- notebook cell 7: join the frequency hand-off -------------------------------
    # only the hand-off columns, not the frequency notebook's carved feature block: the
    # feature set here must be exactly the optbinning-binned columns
    merged = pd.read_csv(os.path.join(REPO, "data", f"frequency_{FREQ_VERSION}.csv"),
                         low_memory=False)
    merged.set_index("ID", inplace=True)
    handoff = merged[[c for c in HANDOFF_COLUMNS if c in merged.columns]]
    missing = [c for c in HANDOFF_COLUMNS if c not in merged.columns]
    if missing:
        raise RuntimeError(f"frequency hand-off is missing {missing}")
    x_train_b = codes_train.join(handoff)
    x_dev_b = codes_dev.join(handoff)
    print(f"[handoff] joined {list(handoff.columns)} -> {x_train_b.shape}", flush=True)

    # ---- notebook cell 8: boosting-flavoured weights --------------------------------
    w_train = (x_train_b["pred_sum"] - x_train_b["FREQ"]).abs()
    w_dev = (x_dev_b["pred_sum"] - x_dev_b["FREQ"]).abs()

    # ---- notebook cell 9's selector, verbatim config --------------------------------
    from AutoCarver import Features
    from AutoCarver.selectors import (
        CramervFilter, DistanceMeasure, RegressionSelector, SelectionConfig, SpearmanFilter,
    )

    binned_columns = list(codes_train.columns)
    features = Features(categoricals=binned_columns)
    config = SelectionConfig(
        quantitative_measures=[DistanceMeasure(threshold=0.002)],
        quantitative_filters=[SpearmanFilter(threshold=0.9)],
        qualitative_filters=[CramervFilter(threshold=0.9)],
    )
    selector = RegressionSelector(features, n_best_features=N_BEST_FEATURES, config=config)
    t0 = time.perf_counter()
    selector.fit(x_train_b, y_train)
    select_seconds = time.perf_counter() - t0
    best_features = selector.selected_features.names
    meta["selection_seconds"] = round(select_seconds, 1)
    meta["n_selected"] = len(best_features)
    print(f"[selector] selected {len(best_features)}/{len(binned_columns)} "
          f"in {select_seconds:.1f}s", flush=True)

    # ---- the notebook's search, one run per seed ------------------------------------
    import optuna
    from sklearn.metrics import root_mean_squared_error
    from objectives import get_best_regression_model, get_regression_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # tweedie requires a non-negative target -- copied from notebook cell 11
    y_transform = lambda u: u.where(u >= 0, 0)

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        objective = get_regression_objective(
            x_train_b[best_features], y_transform(y_train),
            x_dev_b[best_features], y_transform(y_dev),
            objective="reg:tweedie", w_train=w_train, w_dev=w_dev,
        )
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=N_TRIALS)

        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            selected_features, xgb = get_best_regression_model(
                study.best_params,
                x_train_b[best_features], y_transform(y_train),
                x_dev_b[best_features], y_transform(y_dev),
                w_train=w_train, w_dev=w_dev, train_on_full=False,
            )
        text = buf.getvalue()
        train = float(re.search(r"RMSE Train:\s*([0-9.eE+-]+)", text).group(1))
        dev = float(re.search(r"RMSE Dev:\s*([0-9.eE+-]+)", text).group(1))

        chg_train = charge_rmse(x_train_b, target, xgb, selected_features, root_mean_squared_error)
        chg_dev = charge_rmse(x_dev_b, target, xgb, selected_features, root_mean_squared_error)

        rows.append({
            "seed": seed,
            "rmse_train": train,
            "rmse_dev": dev,
            "charge_rmse_train": chg_train,
            "charge_rmse_dev": chg_dev,
            "max_depth": study.best_params.get("max_depth"),
            "n_estimators": study.best_params.get("n_estimators"),
            "minutes": round((time.perf_counter() - t0) / 60, 1),
        })
        ac = AUTOCARVER_DEV_RMSE.get(seed)
        note = f"  (AutoCarver same seed {ac:.4f}, gap {dev - ac:+.4f})" if ac and not SMOKE else ""
        print(f"seed {seed:5d}: dev RMSE {dev:.10f}  dev CHARGE {chg_dev:.10f}  "
              f"train RMSE {train:.10f}  depth {rows[-1]['max_depth']}  "
              f"{rows[-1]['minutes']} min{note}", flush=True)

    df = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# arm: optbinning {_optbinning_version()} BinningProcess"
                 f" (min_prebin_size={MIN_PREBIN_SIZE}, max_n_bins={MAX_N_BINS},"
                 f" n_jobs={N_JOBS}, metric=bins/indices)\n")
        fh.write("# binning target: continuous CM (no collapse -- the continuous path"
                 " accepts categorical_variables)\n")
        fh.write(f"# n_trials: {N_TRIALS} | seeds: {SEEDS} | smoke: {SMOKE}\n")
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        df.to_csv(fh, index=False)

    summary.to_csv(os.path.join(REPO, "data", "ab_arms", "optbinning_amount_binning_summary.csv"),
                   index=False)

    print("\n" + df.to_string(index=False))
    if not SMOKE:
        for col, baseline, label in (
            ("rmse_dev", AUTOCARVER_DEV_RMSE, "severity dev RMSE"),
            ("charge_rmse_dev", AUTOCARVER_DEV_CHARGE, "CHARGE dev RMSE"),
        ):
            ours, ac = df[col], pd.Series(baseline)
            spread, ac_spread = ours.max() - ours.min(), ac.max() - ac.min()
            pooled = (spread + ac_spread) / 2
            gap = ours.mean() - ac.mean()
            print(f"\n{label}: optbinning min {ours.min():.4f} | max {ours.max():.4f} "
                  f"| mean {ours.mean():.4f} | spread {spread:.4f}")
            print(f"{' ' * len(label)}  AutoCarver mean {ac.mean():.4f} | spread {ac_spread:.4f}")
            print(f"{' ' * len(label)}  gap {gap:+.4f} = {gap / pooled:+.2f} x pooled spread "
                  f"({pooled:.4f})")
    print(f"\nwrote {OUT}")


def _optbinning_version() -> str:
    import optbinning

    return optbinning.__version__


if __name__ == "__main__":
    main()
