"""No-carving ablation: the same pipeline with no supervised binning at all.

The article's §2.1 believes supervised binning is where the bulk of the lift came from,
and has never tested it. This is that test. Everything downstream is unchanged -- the same
selector and budget, the same objectives, the same trial counts, the same four seeds -- so
the only thing removed is the carving step.

The arm is deliberately a *competent* baseline rather than a straw man:

* numericals are passed through untouched;
* ordinals are **rank-encoded from the orderings the notebook already declares**, so the
  feature order the carver would have used is preserved here too;
* categoricals become integer codes fitted on train, with unseen levels at -1.

That is what a practitioner reaches for with no binning library at hand, and it keeps the
declared feature types doing real work, so a loss against the carved arms cannot be
blamed on throwing the type declarations away.

Usage::

    uv run --no-sync python tools/ablation_no_carving.py --model frequency
    uv run --no-sync python tools/ablation_no_carving.py --model amount

`ABLATION_SMOKE=1` subsamples rows/features and runs 3 trials on one seed.
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
SEEDS = [42, 1, 7, 2026]
SMOKE = os.environ.get("ABLATION_SMOKE") == "1"

CONFIG = {
    "frequency": {
        "notebook": "frequency_model_2026.ipynb",
        "stop": lambda src: "OrdinalCarver(" in src and "fit_transform" in src,
        "n_best": 100,
        "n_trials": 300,
        "out": "no_carving_frequency.csv",
    },
    "amount": {
        "notebook": "amount_model_2026.ipynb",
        "stop": lambda src: "ContinuousCarver" in src and "fit_transform" in src,
        "n_best": 200,
        "n_trials": 400,
        "out": "no_carving_amount.csv",
    },
}


def notebook_prefix(name, stop):
    """Exec the notebook's code cells up to (not including) the carving cell."""
    nb = json.loads((REPO / "src" / name).read_text(encoding="utf-8"))
    os.chdir(REPO / "src")
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if stop(src):
            return g
        exec(compile(src, "<cell>", "exec"), g)
    raise RuntimeError("never reached the carving cell -- the stop condition is wrong")


def encode(x_train, x_dev, cats, ordinal_columns, nums):
    """Type-aware encoding, fitted on train: ranks for ordinals, codes for categoricals."""
    import pandas as pd

    ords = list(ordinal_columns)
    train = x_train[cats + ords + nums].copy()
    dev = x_dev[cats + ords + nums].copy()
    meta = {"ordinals_from_declared": 0, "ordinals_from_prefix": 0, "ordinals_unmapped": []}

    for column in ords:
        levels = ordinal_columns[column] if isinstance(ordinal_columns, dict) else None
        tr, dv = x_train[column].astype("string"), x_dev[column].astype("string")
        codes_tr = codes_dv = None
        if levels:
            mapping = {level: rank for rank, level in enumerate(levels)}
            candidate = tr.map(mapping)
            if candidate.notna().sum() >= tr.notna().sum():
                codes_tr, codes_dv = candidate, dv.map(mapping)
                meta["ordinals_from_declared"] += 1
        if codes_tr is None:
            parsed = tr.str.extract(r"^\s*(\d+)", expand=False)
            if parsed.notna().sum() == tr.notna().sum():
                codes_tr = parsed
                codes_dv = dv.str.extract(r"^\s*(\d+)", expand=False)
                meta["ordinals_from_prefix"] += 1
        if codes_tr is None:
            meta["ordinals_unmapped"].append(column)
            categories = pd.Index(x_train[column].astype("string").dropna().unique())
            codes_tr = tr.map({v: i for i, v in enumerate(categories)})
            codes_dv = dv.map({v: i for i, v in enumerate(categories)})
        train[column] = pd.to_numeric(codes_tr, errors="coerce")
        dev[column] = pd.to_numeric(codes_dv, errors="coerce")

    for column in cats:
        categories = pd.Index(x_train[column].astype("string").dropna().unique())
        lookup = {value: index for index, value in enumerate(categories)}
        train[column] = x_train[column].astype("string").map(lookup).fillna(-1).astype("int32")
        dev[column] = x_dev[column].astype("string").map(lookup).fillna(-1).astype("int32")

    for column in nums:  # already numeric, but the 2025 engineering leaves some as object
        train[column] = pd.to_numeric(train[column], errors="coerce")
        dev[column] = pd.to_numeric(dev[column], errors="coerce")

    meta["n_categoricals"] = len(cats)
    meta["n_ordinals"] = len(ords)
    meta["n_numericals"] = len(nums)
    meta["max_categorical_cardinality"] = int(
        max((x_train[c].nunique() for c in cats), default=0)
    )
    return train, dev, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CONFIG), required=True)
    args = parser.parse_args()
    spec = CONFIG[args.model]

    g = notebook_prefix(spec["notebook"], spec["stop"])
    import numpy as np
    import pandas as pd
    from AutoCarver import Features

    cats = list(g["categorical_columns"])
    ordinal_columns = g["ordinal_columns"]
    nums = list(g["numerical_columns"])
    x_train, x_dev = g["x_train"], g["x_dev"]
    if SMOKE:
        cats, nums = cats[:20], nums[:10]
        ordinal_columns = (
            {k: ordinal_columns[k] for k in list(ordinal_columns)[:10]}
            if isinstance(ordinal_columns, dict)
            else list(ordinal_columns)[:10]
        )
        keep = x_train.index[:20000]
        x_train = x_train.loc[keep]
        x_dev = x_dev.loc[x_dev.index[:5000]]

    t0 = time.perf_counter()
    train_enc, dev_enc, meta = encode(x_train, x_dev, cats, ordinal_columns, nums)
    encode_seconds = time.perf_counter() - t0
    print(
        f"[encode] {encode_seconds:.1f}s | {meta['n_categoricals']} categoricals "
        f"(max cardinality {meta['max_categorical_cardinality']}), {meta['n_ordinals']} "
        f"ordinals ({meta['ordinals_from_declared']} from declared levels, "
        f"{meta['ordinals_from_prefix']} from prefix, {len(meta['ordinals_unmapped'])} "
        f"fell back to codes), {meta['n_numericals']} numericals",
        flush=True,
    )

    ords = list(ordinal_columns)
    # categoricals stay nominal; rank-encoded ordinals and raw numericals are ordered,
    # so they go in as quantitatives -- the same split of roles the carved arms use
    features = Features(categoricals=cats) + Features(numericals=ords + nums)

    from AutoCarver.selectors import (
        CramervFilter,
        SelectionConfig,
        SpearmanFilter,
    )

    n_best = 5 if SMOKE else spec["n_best"]
    t1 = time.perf_counter()
    if args.model == "frequency":
        from AutoCarver.selectors import ClassificationSelector

        config = SelectionConfig(
            qualitative_filters=[CramervFilter(threshold=0.9)],
            quantitative_filters=[SpearmanFilter(threshold=0.9)],
        )
        selector = ClassificationSelector(features, n_best_features=n_best, config=config)
        y_select = g["y_ord_train"] if not SMOKE else g["y_ord_train"][: len(train_enc)]
        selector.fit(train_enc, y_select)
    else:
        from AutoCarver.selectors import DistanceMeasure, RegressionSelector

        config = SelectionConfig(
            quantitative_measures=[DistanceMeasure(threshold=0.002)],
            quantitative_filters=[SpearmanFilter(threshold=0.9)],
            qualitative_filters=[CramervFilter(threshold=0.9)],
        )
        selector = RegressionSelector(features, n_best_features=n_best, config=config)
        selector.fit(train_enc, g["y_train"].loc[train_enc.index])
    best_features = list(selector.selected_features.names)
    selection_seconds = time.perf_counter() - t1
    print(f"[select] {selection_seconds:.1f}s -> {len(best_features)} features", flush=True)

    # --- the severity arm needs the frequency hand-off and its error weights -----------
    if args.model == "amount":
        merged = pd.read_csv(REPO / "data" / "frequency_2026.csv", low_memory=False)
        merged.set_index("ID", inplace=True)
        carried = [c for c in merged.columns if c not in train_enc.columns]
        train_enc = train_enc.join(merged[carried])
        dev_enc = dev_enc.join(merged[carried])
        w_train = (train_enc["pred_sum"] - train_enc["FREQ"]).abs()
        w_dev = (dev_enc["pred_sum"] - dev_enc["FREQ"]).abs()
    else:
        w_train, w_dev = g["w_train"], g["w_dev"]
        if SMOKE:
            w_train, w_dev = w_train[: len(train_enc)], w_dev[: len(dev_enc)]

    import optuna
    from objectives import (
        get_best_multiclass_model,
        get_best_regression_model,
        get_multiclass_objective,
        get_regression_objective,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = 3 if SMOKE else spec["n_trials"]
    seeds = SEEDS[:1] if SMOKE else SEEDS
    rows = []

    for seed in seeds:
        t2 = time.perf_counter()
        if args.model == "frequency":
            y_train = g["y_ord_train"][: len(train_enc)] if SMOKE else g["y_ord_train"]
            y_dev = g["y_ord_dev"][: len(dev_enc)] if SMOKE else g["y_ord_dev"]
            objective = get_multiclass_objective(
                train_enc[best_features], y_train, dev_enc[best_features], y_dev,
                w_train=w_train, w_dev=w_dev,
            )
            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=n_trials)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                get_best_multiclass_model(
                    study.best_params, train_enc[best_features], y_train,
                    dev_enc[best_features], y_dev,
                    w_train=w_train, w_dev=w_dev, train_on_full=False,
                )
            text = buffer.getvalue()
            row = {
                "seed": seed,
                "log_loss_train": float(re.search(r"Log Loss Train:\s*([0-9.]+)", text).group(1)),
                "log_loss_dev": float(re.search(r"Log Loss Dev:\s*([0-9.]+)", text).group(1)),
            }
        else:
            y_train = g["y_train"].loc[train_enc.index]
            y_dev = g["y_dev"].loc[dev_enc.index]
            y_transform = lambda u: u.where(u >= 0, 0)
            objective = get_regression_objective(
                train_enc[best_features], y_transform(y_train),
                dev_enc[best_features], y_transform(y_dev),
                objective="reg:tweedie", w_train=w_train, w_dev=w_dev,
            )
            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=n_trials)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                selected_features, xgb = get_best_regression_model(
                    study.best_params, train_enc[best_features], y_transform(y_train),
                    dev_enc[best_features], y_transform(y_dev),
                    w_train=w_train, w_dev=w_dev, train_on_full=False,
                )
            text = buffer.getvalue()
            row = {
                "seed": seed,
                "rmse_train": float(re.search(r"RMSE Train:\s*([0-9.]+)", text).group(1)),
                "rmse_dev": float(re.search(r"RMSE Dev:\s*([0-9.]+)", text).group(1)),
            }
            from sklearn.metrics import root_mean_squared_error

            target = g["target"]
            for label, frame in (("train", train_enc), ("dev", dev_enc)):
                true_charge = (
                    frame["CHARGE"] if "CHARGE" in frame.columns
                    else target.loc[frame.index, "CHARGE"]
                )
                predicted = frame["pred_sum"].to_numpy() * xgb.predict(frame[selected_features])
                row[f"charge_rmse_{label}"] = float(
                    root_mean_squared_error(true_charge, predicted)
                )

        row["max_depth"] = study.best_params.get("max_depth")
        row["n_estimators"] = study.best_params.get("n_estimators")
        row["minutes"] = round((time.perf_counter() - t2) / 60, 1)
        rows.append(row)
        print(f"seed {seed:5d}: " + "  ".join(f"{k} {v}" for k, v in row.items() if k != "seed"), flush=True)

    frame = pd.DataFrame(rows)
    out = REPO / "data" / "ab_arms" / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# arm: no carving -- ordinals rank-encoded from declared levels, categoricals",
        "#   integer-coded on train, numericals untouched. Selector, budget, objectives and",
        "#   trial counts identical to the carved arms; only the binning step is removed.",
        f"# n_trials: {n_trials} | seeds: {seeds} | smoke: {SMOKE}",
        f"# encode_seconds: {encode_seconds:.1f}",
        f"# selection_seconds: {selection_seconds:.1f}",
        f"# n_input_features: {len(cats) + len(ords) + len(nums)}",
        f"# n_selected: {len(best_features)}",
        f"# max_categorical_cardinality: {meta['max_categorical_cardinality']}",
        f"# ordinals_from_declared_levels: {meta['ordinals_from_declared']}",
        f"# ordinals_from_prefix: {meta['ordinals_from_prefix']}",
        f"# ordinals_fell_back_to_codes: {len(meta['ordinals_unmapped'])}",
    ]
    with open(out, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(header) + "\n")
        frame.to_csv(handle, index=False)
    print(f"\nwrote {out}")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
