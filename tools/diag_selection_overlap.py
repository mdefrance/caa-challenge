"""Answer three questions about the optbinning comparison, from real outputs.

1. Are the ordinal columns numeric-valued? optbinning's `dtype="numerical"` covers
   "continuous and ordinal variables"; the ablation declared all 238 of ours as
   `categorical_variables`, i.e. nominal. If they are numeric-valued, that declaration
   threw away the feature ordering and the frequency arm handicapped optbinning.
2. Do the two arms select the same features?
3. What does declaring the ordinals correctly change -- bins, and the selected set?

Carves, bins and selects only. No Optuna, so this is minutes rather than hours.

    uv run --no-sync python tools/diag_selection_overlap.py
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
NB = REPO / "src" / "frequency_model_2026.ipynb"
OUT = REPO / "data" / "ab_arms" / "selection_overlap.json"

MIN_FREQ = 0.02
MAX_N_MOD = 5
N_JOBS = 6
N_BEST = 100


def notebook_prefix():
    """Exec the 2026 frequency notebook up to (not including) the OrdinalCarver fit."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    os.chdir(REPO / "src")
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "OrdinalCarver(" in src and "fit_transform" in src:
            break
        exec(compile(src, "<cell>", "exec"), g)
    return g


def select(features, frame, y, label):
    from AutoCarver.selectors import (
        ClassificationSelector,
        CramervFilter,
        SelectionConfig,
        SpearmanFilter,
    )

    config = SelectionConfig(
        qualitative_filters=[CramervFilter(threshold=0.9)],
        quantitative_filters=[SpearmanFilter(threshold=0.9)],
    )
    selector = ClassificationSelector(features, n_best_features=N_BEST, config=config)
    t0 = time.perf_counter()
    selector.fit(frame, y)
    names = list(selector.selected_features.names)
    print(f"[{label}] selected {len(names)} in {time.perf_counter() - t0:.1f}s", flush=True)
    return names


def main() -> int:
    g = notebook_prefix()
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_ord_train, y_ord_dev = g["y_ord_train"], g["y_ord_dev"]
    cats = list(g["categorical_columns"])
    ords = list(g["ordinal_columns"])
    nums = list(g["numerical_columns"])
    print(f"columns: {len(cats)} categorical, {len(ords)} ordinal, {len(nums)} numerical", flush=True)

    import numpy as np
    import pandas as pd

    # --- 1. are the ordinal columns numeric-valued? ------------------------------------
    report = {"n_categorical": len(cats), "n_ordinal": len(ords), "n_numerical": len(nums)}
    ord_dtypes = x_train[ords].dtypes.astype(str).value_counts().to_dict()
    coercible, not_coercible = [], []
    for column in ords:
        series = x_train[column]
        if pd.api.types.is_numeric_dtype(series):
            coercible.append(column)
            continue
        try:
            pd.to_numeric(series, errors="raise")
            coercible.append(column)
        except (ValueError, TypeError):
            not_coercible.append(column)
    report["ordinal_dtypes"] = ord_dtypes
    report["n_ordinal_numeric_valued"] = len(coercible)
    report["n_ordinal_not_numeric"] = len(not_coercible)
    report["ordinal_not_numeric_sample"] = not_coercible[:10]
    print(f"ordinal dtypes: {ord_dtypes}", flush=True)
    print(f"ordinals numeric-valued: {len(coercible)} / {len(ords)}", flush=True)
    if not_coercible:
        print(f"  not numeric, e.g.: {not_coercible[:5]}", flush=True)
        for column in not_coercible[:3]:
            print(f"   {column}: {x_train[column].dropna().unique()[:5]}", flush=True)

    # --- 2. AutoCarver arm: carve, then select ----------------------------------------
    from AutoCarver import Features, OrdinalCarver
    from AutoCarver.discretizers import ProcessingConfig

    config = ProcessingConfig(
        dropna=False, copy=True, verbose=False, n_jobs=N_JOBS, min_freq_alpha=0.05
    )
    qualitatives = Features(categoricals=cats, ordinals=g["ordinal_columns"])
    carver = OrdinalCarver(features=qualitatives, target_scale="level", config=config)
    t0 = time.perf_counter()
    ac_train = carver.fit_transform(x_train.copy(), y_ord_train, X_dev=x_dev.copy(), y_dev=y_ord_dev)
    print(f"[autocarver] carve {time.perf_counter() - t0:.1f}s", flush=True)
    ac_selected = select(carver.features + Features(numericals=nums), ac_train, y_ord_train, "autocarver")

    # --- 3. optbinning, both declarations ----------------------------------------------
    from optbinning import BinningProcess

    def optbinning_arm(categorical_variables, label):
        variable_names = cats + ords + nums
        process = BinningProcess(
            variable_names=variable_names,
            categorical_variables=list(categorical_variables),
            min_prebin_size=MIN_FREQ,
            max_n_bins=MAX_N_MOD,
            n_jobs=N_JOBS,
        )
        frame = x_train[variable_names]
        if label == "ordinals_as_numerical":
            # optbinning's "numerical" dtype is what its docs prescribe for ordinal
            # variables; that requires the values themselves to be numeric
            frame = frame.copy()
            for column in coercible:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        t0 = time.perf_counter()
        process.fit(frame, (np.asarray(y_ord_train) > 0).astype(int))
        fit_s = time.perf_counter() - t0
        summary = process.summary()
        codes = process.transform(
            frame, metric="indices", metric_special="empirical", metric_missing="empirical"
        )
        n_single = int((summary.n_bins <= 1).sum())
        print(
            f"[optbinning:{label}] fit {fit_s:.1f}s | mean bins {summary.n_bins.mean():.2f} "
            f"| total bins {int(summary.n_bins.sum())} | single-bin {n_single}",
            flush=True,
        )
        selected = select(Features(categoricals=list(codes.columns)), codes, y_ord_train, f"optbinning:{label}")
        return {
            "fit_seconds": round(fit_s, 1),
            "mean_bins": float(summary.n_bins.mean()),
            "total_bins": int(summary.n_bins.sum()),
            "n_single_bin": n_single,
            "selected": selected,
        }

    arms = {
        "ordinals_as_categorical": optbinning_arm(cats + ords, "ordinals_as_categorical"),
        "ordinals_as_numerical": optbinning_arm(cats, "ordinals_as_numerical"),
    }

    # --- overlaps ----------------------------------------------------------------------
    report["autocarver_selected"] = ac_selected
    report["optbinning_arms"] = arms
    ac_set = set(ac_selected)
    for name, arm in arms.items():
        shared = ac_set & set(arm["selected"])
        report[f"overlap_autocarver_vs_{name}"] = len(shared)
        print(
            f"overlap AutoCarver vs optbinning[{name}]: {len(shared)} / {len(ac_selected)} "
            f"({100 * len(shared) / max(len(ac_selected), 1):.0f} %)",
            flush=True,
        )
    both_optb = set(arms["ordinals_as_categorical"]["selected"]) & set(
        arms["ordinals_as_numerical"]["selected"]
    )
    report["overlap_between_optbinning_arms"] = len(both_optb)
    print(f"overlap between the two optbinning declarations: {len(both_optb)} / {N_BEST}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
