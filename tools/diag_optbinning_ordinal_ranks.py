"""Give optbinning its best shot at our ordinal features, and see whether it matters.

The ablation declared all 238 ordinal columns to optbinning as `categorical_variables`.
optbinning's docs put ordinal variables under `dtype="numerical"` instead -- but that path
needs numeric values, and ours are labelled bands ('01. <= 1', '02. <= 3', ...). So the
categorical declaration was forced, not careless.

A competent optbinning user would not stop there: the labels carry their own rank in the
prefix, so they can be mapped to integers and passed as numerical, which is the only way
optbinning can use a feature's order. This script builds that arm and reports what it
changes -- bins, and the selected 100 -- against the two declarations already measured in
`data/ab_arms/selection_overlap.json`.

Binning and selection only, no Optuna.

    uv run --no-sync python tools/diag_optbinning_ordinal_ranks.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
NB = REPO / "src" / "frequency_model_2026.ipynb"
PRIOR = REPO / "data" / "ab_arms" / "selection_overlap.json"
OUT = REPO / "data" / "ab_arms" / "optbinning_ordinal_ranks.json"

MIN_FREQ, MAX_N_MOD, N_JOBS, N_BEST = 0.02, 5, 6, 100


def notebook_prefix():
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


def main() -> int:
    prior = json.loads(PRIOR.read_text(encoding="utf-8")) if PRIOR.exists() else {}
    g = notebook_prefix()
    import numpy as np
    import pandas as pd
    from optbinning import BinningProcess
    from AutoCarver import Features
    from AutoCarver.selectors import (
        ClassificationSelector,
        CramervFilter,
        SelectionConfig,
        SpearmanFilter,
    )

    x_train = g["x_train"]
    y_ord_train = g["y_ord_train"]
    cats, nums = list(g["categorical_columns"]), list(g["numerical_columns"])
    ordinal_columns = g["ordinal_columns"]
    ords = list(ordinal_columns)

    # --- rank-encode the ordinals, using the declared order where we have it -----------
    # the notebook declares ordinals as {column: [level, level, ...]}; fall back to the
    # numeric prefix the labels carry ('01. <= 1' -> 1) when it is not a mapping
    frame = x_train[cats + ords + nums].copy()
    declared, prefix_parsed, unmapped = 0, 0, []
    for column in ords:
        levels = ordinal_columns[column] if isinstance(ordinal_columns, dict) else None
        series = x_train[column].astype("string")
        if levels:
            mapping = {level: rank for rank, level in enumerate(levels)}
            codes = series.map(mapping)
            if codes.notna().sum() >= series.notna().sum():
                declared += 1
            else:  # declared levels did not cover the observed values
                codes = None
        else:
            codes = None
        if codes is None:
            parsed = series.str.extract(r"^\s*(\d+)", expand=False)
            if parsed.notna().sum() == series.notna().sum():
                codes = parsed.astype("Float64")
                prefix_parsed += 1
            else:
                unmapped.append(column)
                continue
        frame[column] = pd.to_numeric(codes, errors="coerce")
    print(f"ordinals rank-encoded: {declared} from declared levels, {prefix_parsed} from the "
          f"label prefix, {len(unmapped)} left categorical", flush=True)

    categorical_variables = cats + unmapped
    variable_names = cats + ords + nums
    process = BinningProcess(
        variable_names=variable_names,
        categorical_variables=categorical_variables,
        min_prebin_size=MIN_FREQ,
        max_n_bins=MAX_N_MOD,
        n_jobs=N_JOBS,
    )
    t0 = time.perf_counter()
    process.fit(frame, (np.asarray(y_ord_train) > 0).astype(int))
    fit_s = time.perf_counter() - t0
    summary = process.summary()
    codes_frame = process.transform(
        frame, metric="indices", metric_special="empirical", metric_missing="empirical"
    )
    print(
        f"[optbinning:ordinals_rank_encoded] fit {fit_s:.1f}s | mean bins "
        f"{summary.n_bins.mean():.2f} | total bins {int(summary.n_bins.sum())} | "
        f"single-bin {int((summary.n_bins <= 1).sum())}",
        flush=True,
    )

    config = SelectionConfig(
        qualitative_filters=[CramervFilter(threshold=0.9)],
        quantitative_filters=[SpearmanFilter(threshold=0.9)],
    )
    selector = ClassificationSelector(
        Features(categoricals=list(codes_frame.columns)), n_best_features=N_BEST, config=config
    )
    t1 = time.perf_counter()
    selector.fit(codes_frame, y_ord_train)
    selected = list(selector.selected_features.names)
    print(f"selected {len(selected)} in {time.perf_counter() - t1:.1f}s", flush=True)

    report = {
        "n_ordinals_rank_encoded_from_declared": declared,
        "n_ordinals_rank_encoded_from_prefix": prefix_parsed,
        "n_ordinals_left_categorical": len(unmapped),
        "fit_seconds": round(fit_s, 1),
        "mean_bins": float(summary.n_bins.mean()),
        "total_bins": int(summary.n_bins.sum()),
        "n_single_bin": int((summary.n_bins <= 1).sum()),
        "selected": selected,
    }
    for name, arm in (prior.get("optbinning_arms") or {}).items():
        shared = set(selected) & set(arm["selected"])
        report[f"overlap_vs_{name}"] = len(shared)
        print(f"overlap vs optbinning[{name}]: {len(shared)} / {N_BEST} "
              f"(bins {arm['total_bins']} -> {report['total_bins']})", flush=True)
    if prior.get("autocarver_selected"):
        shared = set(selected) & set(prior["autocarver_selected"])
        report["overlap_vs_autocarver"] = len(shared)
        print(f"overlap vs AutoCarver: {len(shared)} / {N_BEST}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
