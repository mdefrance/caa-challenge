"""One harness for the sharpening experiments in `work/PLAN-tds-revision.md` §4.

Everything downstream of binning is the notebook's own code -- same objectives, same trial
counts, same four seeds -- so any arm defined here differs from another only in the ways
its flags say.

Axes
----
``--arm``       autocarver | optbinning | nocarve
``--features``  full   -- run the selector on the arm's own binned frame (as §3.5 did)
                fixed  -- skip selection and model a shared, binning-agnostic feature list,
                          which isolates binning from selection
``--optb-max-bins``   optbinning's ``max_n_bins`` (default 5); lower it to match
                      AutoCarver's realised width
``--optb-ordinals``   categorical -- the 238 ordinal band columns as nominal levels, which
                                     is what §3.5 ran
                      ranks       -- rank-encoded from the notebook's declared orderings and
                                     passed as numerical, the only way optbinning can use a
                                     feature's order

The shared list for ``--features fixed`` is written by ``--emit-fixed-list``, which runs the
pipeline's own selection: carve with AutoCarver, then select. It is therefore **not**
neutral, and the asymmetry is the point -- on AutoCarver's own chosen features, another arm
matching it is strong evidence, and AutoCarver winning is weak evidence.

Severity runs also report metrics with more dynamic range than RMSE, because on this
dataset RMSE cannot separate any arm from a constant: Spearman correlation, top-decile
lift, and Tweedie deviance at a fixed power (the tuned power differs per arm, so a tuned
one would not be comparable).

Examples::

    uv run --no-sync python tools/ablation_matrix.py --model frequency --emit-fixed-list
    uv run --no-sync python tools/ablation_matrix.py --model frequency --arm autocarver \
        --features fixed --tag fixed_autocarver
    uv run --no-sync python tools/ablation_matrix.py --model frequency --arm optbinning \
        --optb-ordinals ranks --tag ranks_optbinning

``ABLATION_SMOKE=1`` subsamples and runs 3 trials on one seed.
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
TWEEDIE_POWER = 1.7  # fixed, so the deviance is comparable across arms

SPEC = {
    "frequency": {
        "notebook": "frequency_model_2026.ipynb",
        "stop": lambda src: "OrdinalCarver(" in src and "fit_transform" in src,
        "n_best": 100,
        "n_trials": 300,
    },
    "amount": {
        "notebook": "amount_model_2026.ipynb",
        "stop": lambda src: "ContinuousCarver" in src and "fit_transform" in src,
        "n_best": 200,
        "n_trials": 400,
    },
}


def notebook_prefix(name, stop):
    nb = json.loads((REPO / "src" / name).read_text(encoding="utf-8"))
    os.chdir(REPO / "src")
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if stop(src):
            return g
        exec(compile(src, "<cell>", "exec"), g)
    raise RuntimeError("never reached the carving cell")


def rank_encode_ordinals(x_train, x_dev, ordinal_columns):
    """Ordinal band labels -> integer ranks, using the notebook's declared orderings."""
    import pandas as pd

    train, dev, meta = {}, {}, {"from_declared": 0, "from_prefix": 0, "unmapped": []}
    for column in list(ordinal_columns):
        levels = ordinal_columns[column] if isinstance(ordinal_columns, dict) else None
        tr, dv = x_train[column].astype("string"), x_dev[column].astype("string")
        codes_tr = codes_dv = None
        if levels:
            mapping = {level: rank for rank, level in enumerate(levels)}
            candidate = tr.map(mapping)
            if candidate.notna().sum() >= tr.notna().sum():
                codes_tr, codes_dv = candidate, dv.map(mapping)
                meta["from_declared"] += 1
        if codes_tr is None:
            parsed = tr.str.extract(r"^\s*(\d+)", expand=False)
            if parsed.notna().sum() == tr.notna().sum():
                codes_tr = parsed
                codes_dv = dv.str.extract(r"^\s*(\d+)", expand=False)
                meta["from_prefix"] += 1
        if codes_tr is None:
            meta["unmapped"].append(column)
            continue
        train[column] = pd.to_numeric(codes_tr, errors="coerce")
        dev[column] = pd.to_numeric(codes_dv, errors="coerce")
    return train, dev, meta


def build_nocarve(x_train, x_dev, cats, ordinal_columns, nums):
    """Type-aware encoding with no supervised binning (the §3.5 unbinned arm)."""
    import pandas as pd

    ords = list(ordinal_columns)
    train = x_train[cats + ords + nums].copy()
    dev = x_dev[cats + ords + nums].copy()
    ranks_tr, ranks_dv, meta = rank_encode_ordinals(x_train, x_dev, ordinal_columns)
    for column in ords:
        if column in ranks_tr:
            train[column], dev[column] = ranks_tr[column], ranks_dv[column]
        else:
            categories = pd.Index(x_train[column].astype("string").dropna().unique())
            lookup = {v: i for i, v in enumerate(categories)}
            train[column] = x_train[column].astype("string").map(lookup)
            dev[column] = x_dev[column].astype("string").map(lookup)
    for column in cats:
        categories = pd.Index(x_train[column].astype("string").dropna().unique())
        lookup = {v: i for i, v in enumerate(categories)}
        train[column] = x_train[column].astype("string").map(lookup).fillna(-1).astype("int32")
        dev[column] = x_dev[column].astype("string").map(lookup).fillna(-1).astype("int32")
    for column in nums:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        dev[column] = pd.to_numeric(dev[column], errors="coerce")
    return train, dev, {"ordinals_rank_encoded": meta["from_declared"] + meta["from_prefix"]}


def build_optbinning(x_train, x_dev, y_binning, cats, ordinal_columns, nums, args):
    """Bin with optbinning, matching what the pipeline hands AutoCarver.

    The pipeline carves qualitative features only and leaves numericals to the selector,
    so binning the numericals here too would compare different scopes, not different
    binners. With ``--optb-bin-numericals no`` (the default) the numericals pass through
    untouched in both arms, which is the parity the reference benchmark insists on:
    "Protocol, identical for every library".
    """
    from optbinning import BinningProcess
    import pandas as pd

    ords = list(ordinal_columns)
    bin_numericals = args.optb_bin_numericals == "yes"
    variable_names = cats + ords + (nums if bin_numericals else [])
    frame_tr = x_train[cats + ords + nums].copy()
    frame_dv = x_dev[cats + ords + nums].copy()
    categorical_variables = list(cats)
    meta = {}
    if args.optb_ordinals == "ranks":
        ranks_tr, ranks_dv, rmeta = rank_encode_ordinals(x_train, x_dev, ordinal_columns)
        for column, series in ranks_tr.items():
            frame_tr[column] = series
            frame_dv[column] = ranks_dv[column]
        categorical_variables += rmeta["unmapped"]
        meta["ordinals_rank_encoded"] = rmeta["from_declared"] + rmeta["from_prefix"]
    else:
        categorical_variables += ords
        meta["ordinals_rank_encoded"] = 0

    process = BinningProcess(
        variable_names=variable_names,
        categorical_variables=categorical_variables,
        min_prebin_size=0.02,
        max_n_bins=args.optb_max_bins,
        n_jobs=6,
    )
    t0 = time.perf_counter()
    process.fit(frame_tr[variable_names], y_binning)
    meta["binning_fit_seconds"] = round(time.perf_counter() - t0, 1)
    kwargs = dict(metric="indices", metric_special="empirical", metric_missing="empirical")
    train = process.transform(frame_tr[variable_names], **kwargs)
    dev = process.transform(frame_dv[variable_names], **kwargs)
    if not bin_numericals:
        # numericals ride along raw, exactly as they do in the AutoCarver arm
        train = pd.concat([train, x_train[nums].apply(pd.to_numeric, errors="coerce")], axis=1)
        dev = pd.concat([dev, x_dev[nums].apply(pd.to_numeric, errors="coerce")], axis=1)
    meta["n_binned_columns"] = len(variable_names)
    meta["n_raw_numericals_passed_through"] = 0 if bin_numericals else len(nums)
    summary = process.summary()
    meta["total_bins"] = int(summary.n_bins.sum())
    meta["mean_bins"] = round(float(summary.n_bins.mean()), 3)
    meta["n_single_bin"] = int((summary.n_bins <= 1).sum())
    return train, dev, meta


def build_autocarver(x_train, x_dev, y_train, y_dev, cats, ordinal_columns, nums, model):
    from AutoCarver import ContinuousCarver, Features, OrdinalCarver
    from AutoCarver.discretizers import ProcessingConfig

    config = ProcessingConfig(
        dropna=False, copy=True, verbose=False, n_jobs=6, min_freq_alpha=0.05
    )
    qualitatives = Features(categoricals=cats, ordinals=ordinal_columns)
    if model == "frequency":
        carver = OrdinalCarver(features=qualitatives, target_scale="level", config=config)
    else:
        carver = ContinuousCarver(features=qualitatives, config=config)
    t0 = time.perf_counter()
    train = carver.fit_transform(x_train.copy(), y_train, X_dev=x_dev.copy(), y_dev=y_dev)
    fit_seconds = time.perf_counter() - t0
    dev = carver.transform(x_dev.copy())
    kept = [f.version for f in carver.features]
    columns = [c for c in kept if c in train.columns and c in dev.columns]
    meta = {
        "binning_fit_seconds": round(fit_seconds, 1),
        "n_dropped_by_veto": len(cats) + len(list(ordinal_columns)) - len(columns),
        "total_bins": int(sum(len(f.labels) for f in carver.features if f.version in columns)),
    }
    keep = columns + [c for c in nums if c in train.columns]
    return train[keep], dev[keep], meta, qualitatives


def severity_extras(y_true, predicted):
    """Metrics with more dynamic range than RMSE on a heavy-tailed target."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_tweedie_deviance

    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    rho = float(spearmanr(truth, pred).statistic)
    frame = pd.DataFrame({"truth": truth, "pred": pred})
    cut = frame.pred.quantile(0.9)
    top = frame[frame.pred >= cut]
    lift = float(top.truth.mean() / frame.truth.mean()) if frame.truth.mean() else float("nan")
    safe_pred = np.clip(pred, 1e-6, None)
    safe_truth = np.clip(truth, 0, None)
    deviance = float(mean_tweedie_deviance(safe_truth, safe_pred, power=TWEEDIE_POWER))
    return {"spearman_dev": rho, "top_decile_lift_dev": lift, "tweedie_deviance_dev": deviance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPEC), required=True)
    parser.add_argument("--arm", choices=["autocarver", "optbinning", "nocarve"])
    parser.add_argument("--features", choices=["full", "fixed"], default="full")
    parser.add_argument("--optb-max-bins", type=int, default=5)
    parser.add_argument("--optb-ordinals", choices=["categorical", "ranks"], default="categorical")
    parser.add_argument("--optb-bin-numericals", choices=["yes", "no"], default="no",
                        help="bin the numericals too (the pipeline's AutoCarver arm does not)")
    parser.add_argument("--tag")
    parser.add_argument("--emit-fixed-list", action="store_true",
                        help="select on the unbinned frame and save the shared feature list")
    args = parser.parse_args()
    spec = SPEC[args.model]
    if not args.emit_fixed_list and not (args.arm and args.tag):
        parser.error("--arm and --tag are required unless --emit-fixed-list")

    g = notebook_prefix(spec["notebook"], spec["stop"])
    import numpy as np
    import pandas as pd
    from AutoCarver import Features

    cats = list(g["categorical_columns"])
    ordinal_columns = g["ordinal_columns"]
    nums = list(g["numerical_columns"])
    x_train, x_dev = g["x_train"], g["x_dev"]
    if args.model == "frequency":
        y_train, y_dev = g["y_ord_train"], g["y_ord_dev"]
    else:
        y_train, y_dev = g["y_train"], g["y_dev"]

    if SMOKE:
        cats, nums = cats[:20], nums[:10]
        keys = list(ordinal_columns)[:10]
        ordinal_columns = ({k: ordinal_columns[k] for k in keys}
                           if isinstance(ordinal_columns, dict) else keys)
        x_train, x_dev = x_train.iloc[:20000], x_dev.iloc[:5000]
        y_train = y_train[:20000] if not hasattr(y_train, "iloc") else y_train.iloc[:20000]
        y_dev = y_dev[:5000] if not hasattr(y_dev, "iloc") else y_dev.iloc[:5000]

    ords = list(ordinal_columns)
    fixed_path = REPO / "data" / "ab_arms" / f"fixed_features_{args.model}.json"

    # --- emit the shared, binning-agnostic feature list --------------------------------
    if args.emit_fixed_list:
        train_c, _, _, carved = build_autocarver(
            x_train, x_dev, y_train, y_dev, cats, ordinal_columns, nums, args.model
        )
        names = run_selector(
            args.model, carved + Features(numericals=nums), train_c, y_train, spec["n_best"]
        )
        payload = {
            "model": args.model,
            "n_selected": len(names),
            "selected": names,
            "note": (
                "selected on the AUTOCARVER-carved frame -- the pipeline's own selection. "
                "This favours AutoCarver by construction: on this list, an AutoCarver win is "
                "weak evidence and an AutoCarver non-win is strong evidence."
            ),
        }
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {fixed_path} ({len(names)} features)")
        return 0

    # --- restrict to the shared list before binning ------------------------------------
    meta = {}
    if args.features == "fixed":
        shared = json.loads(fixed_path.read_text(encoding="utf-8"))["selected"]
        cats = [c for c in cats if c in shared]
        nums = [c for c in nums if c in shared]
        keys = [c for c in ords if c in shared]
        ordinal_columns = ({k: ordinal_columns[k] for k in keys}
                           if isinstance(ordinal_columns, dict) else keys)
        ords = keys
        meta["fixed_feature_list"] = fixed_path.name
        print(f"fixed set: {len(cats)} categorical, {len(ords)} ordinal, {len(nums)} numerical "
              f"= {len(cats) + len(ords) + len(nums)} of {len(shared)}", flush=True)

    # --- build the modelled frame ------------------------------------------------------
    if args.arm == "nocarve":
        train_b, dev_b, arm_meta = build_nocarve(x_train, x_dev, cats, ordinal_columns, nums)
    elif args.arm == "optbinning":
        y_binning = ((np.asarray(y_train) > 0).astype(int) if args.model == "frequency"
                     else np.asarray(y_train, dtype=float))
        train_b, dev_b, arm_meta = build_optbinning(
            x_train, x_dev, y_binning, cats, ordinal_columns, nums, args
        )
    else:
        train_b, dev_b, arm_meta, carved_features = build_autocarver(
            x_train, x_dev, y_train, y_dev, cats, ordinal_columns, nums, args.model
        )
    meta.update(arm_meta)
    print(f"[{args.arm}] modelled frame {train_b.shape} | {arm_meta}", flush=True)

    # --- selection, or not -------------------------------------------------------------
    if args.features == "fixed":
        # Feed the columns in the shared list's OWN order, which is the order the
        # pipeline's selector ranked them in. Rebuilding the frame grouped by type would
        # change each column's position, and position is not neutral: the search tunes
        # colsample_bytree / colsample_bylevel, so XGBoost's per-tree column draw picks
        # by position. Same features in a different order is a different model.
        shared_order = json.loads(fixed_path.read_text(encoding="utf-8"))["selected"]
        available = {c for c in train_b.columns if c in dev_b.columns}
        best_features = [c for c in shared_order if c in available]
        best_features += [c for c in train_b.columns
                          if c in available and c not in shared_order]
        selection_seconds = 0.0
    else:
        if args.arm == "nocarve":
            features = Features(categoricals=cats) + Features(numericals=ords + nums)
        elif args.arm == "autocarver":
            features = carved_features + Features(numericals=nums)
        else:
            binned = [c for c in train_b.columns if c not in nums]
            passed = [c for c in nums if c in train_b.columns]
            features = Features(categoricals=binned) + Features(numericals=passed)
        t0 = time.perf_counter()
        best_features = run_selector(args.model, features, train_b, y_train, spec["n_best"])
        selection_seconds = time.perf_counter() - t0
    print(f"[select] {selection_seconds:.1f}s -> {len(best_features)} features", flush=True)
    if args.features == "full":
        # kept so between-arm selection overlap can be computed without re-selecting
        sidecar = REPO / "data" / "ab_arms" / f"matrix_{args.model}_{args.tag}_selected.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"model": args.model, "arm": args.arm,
                                       "tag": args.tag, "selected": best_features}, indent=2),
                           encoding="utf-8")

    # --- severity needs the frequency hand-off and its error weights --------------------
    if args.model == "amount":
        merged = pd.read_csv(REPO / "data" / "frequency_2026.csv", low_memory=False)
        merged.set_index("ID", inplace=True)
        carried = [c for c in merged.columns if c not in train_b.columns]
        train_b = train_b.join(merged[carried])
        dev_b = dev_b.join(merged[carried])
        w_train = (train_b["pred_sum"] - train_b["FREQ"]).abs()
        w_dev = (dev_b["pred_sum"] - dev_b["FREQ"]).abs()
    else:
        w_train, w_dev = g["w_train"], g["w_dev"]
        if SMOKE:
            w_train, w_dev = w_train[:len(train_b)], w_dev[:len(dev_b)]

    import optuna
    from objectives import (
        get_best_multiclass_model,
        get_best_regression_model,
        get_multiclass_objective,
        get_regression_objective,
    )
    from sklearn.metrics import root_mean_squared_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = 3 if SMOKE else spec["n_trials"]
    seeds = SEEDS[:1] if SMOKE else SEEDS
    rows = []

    for seed in seeds:
        t0 = time.perf_counter()
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        if args.model == "frequency":
            study.optimize(
                get_multiclass_objective(train_b[best_features], y_train,
                                         dev_b[best_features], y_dev,
                                         w_train=w_train, w_dev=w_dev),
                n_trials=n_trials,
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                get_best_multiclass_model(
                    study.best_params, train_b[best_features], y_train,
                    dev_b[best_features], y_dev,
                    w_train=w_train, w_dev=w_dev, train_on_full=False,
                )
            text = buffer.getvalue()
            row = {
                "seed": seed,
                "log_loss_train": float(re.search(r"Log Loss Train:\s*([0-9.]+)", text).group(1)),
                "log_loss_dev": float(re.search(r"Log Loss Dev:\s*([0-9.]+)", text).group(1)),
            }
        else:
            y_transform = lambda u: u.where(u >= 0, 0)
            study.optimize(
                get_regression_objective(train_b[best_features], y_transform(y_train),
                                         dev_b[best_features], y_transform(y_dev),
                                         objective="reg:tweedie",
                                         w_train=w_train, w_dev=w_dev),
                n_trials=n_trials,
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                selected_features, xgb = get_best_regression_model(
                    study.best_params, train_b[best_features], y_transform(y_train),
                    dev_b[best_features], y_transform(y_dev),
                    w_train=w_train, w_dev=w_dev, train_on_full=False,
                )
            text = buffer.getvalue()
            row = {
                "seed": seed,
                "rmse_train": float(re.search(r"RMSE Train:\s*([0-9.]+)", text).group(1)),
                "rmse_dev": float(re.search(r"RMSE Dev:\s*([0-9.]+)", text).group(1)),
            }
            target = g["target"]
            dev_pred = xgb.predict(dev_b[selected_features])
            for label, frame, predicted in (("train", train_b, None), ("dev", dev_b, dev_pred)):
                predicted = xgb.predict(frame[selected_features]) if predicted is None else predicted
                true_charge = (frame["CHARGE"] if "CHARGE" in frame.columns
                               else target.loc[frame.index, "CHARGE"])
                row[f"charge_rmse_{label}"] = float(
                    root_mean_squared_error(true_charge, frame["pred_sum"].to_numpy() * predicted)
                )
            row.update(severity_extras(y_dev, dev_pred))

        row["max_depth"] = study.best_params.get("max_depth")
        row["n_estimators"] = study.best_params.get("n_estimators")
        row["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        rows.append(row)
        print(f"seed {seed:5d}: " + "  ".join(f"{k} {v}" for k, v in row.items() if k != "seed"),
              flush=True)

    frame = pd.DataFrame(rows)
    out = REPO / "data" / "ab_arms" / f"matrix_{args.model}_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# model: {args.model} | arm: {args.arm} | features: {args.features}",
        f"# optb_max_bins: {args.optb_max_bins} | optb_ordinals: {args.optb_ordinals}"
        f" | optb_bin_numericals: {args.optb_bin_numericals}",
        f"# n_trials: {n_trials} | seeds: {seeds} | smoke: {SMOKE}",
        f"# n_modelled_features: {len(best_features)}",
        f"# selection_seconds: {selection_seconds:.1f}",
        f"# tweedie_power_for_deviance: {TWEEDIE_POWER}",
    ] + [f"# {k}: {v}" for k, v in meta.items()]
    with open(out, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(header) + "\n")
        frame.to_csv(handle, index=False)
    print(f"\nwrote {out}")
    print(frame.to_string(index=False))
    return 0


def run_selector(model, features, frame, y, n_best):
    from AutoCarver.selectors import (
        ClassificationSelector,
        CramervFilter,
        DistanceMeasure,
        RegressionSelector,
        SelectionConfig,
        SpearmanFilter,
    )

    n_best = 5 if SMOKE else n_best
    if model == "frequency":
        config = SelectionConfig(
            qualitative_filters=[CramervFilter(threshold=0.9)],
            quantitative_filters=[SpearmanFilter(threshold=0.9)],
        )
        selector = ClassificationSelector(features, n_best_features=n_best, config=config)
    else:
        config = SelectionConfig(
            quantitative_measures=[DistanceMeasure(threshold=0.002)],
            quantitative_filters=[SpearmanFilter(threshold=0.9)],
            qualitative_filters=[CramervFilter(threshold=0.9)],
        )
        selector = RegressionSelector(features, n_best_features=n_best, config=config)
    selector.fit(frame, y)
    return list(selector.selected_features.names)


if __name__ == "__main__":
    raise SystemExit(main())
