"""Probe: what does optbinning's BinningProcess do with the 3-class ordinal target?

ARTICLE.md 2.1 claims optbinning's multiclass path (MulticlassOptimalBinning) is
numerical-only and therefore cannot bin this dataset's ~400 categorical/ordinal columns.
This script *measures* that claim instead of asserting it: it execs the frequency
notebook's prefix (cells 0..5, stopping before the OrdinalCarver), takes a small
subsample, and fits BinningProcess with the untouched 3-class target
`collapse_count(y) in {0, 1, 2}`.

Whatever happens -- exception text, or per-variable status in the binning table -- is
printed verbatim and is what justifies the binary collapse used by the full ablation.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "frequency_model_2026.ipynb")

N_ROWS = 20_000
N_CAT = 15
N_ORD = 15
N_NUM = 15


def exec_prefix():
    """Exec code cells 0..5 -- everything before the OrdinalCarver fit_transform."""
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "OrdinalCarver(" in src and "fit_transform" in src:
            print(f"[prefix] stopping before code cell {i} (OrdinalCarver fit_transform)")
            break
        t0 = time.perf_counter()
        exec(compile(src, f"<cell {i}>", "exec"), g)
        print(f"[prefix] cell {i} ok ({time.perf_counter() - t0:.1f}s)", flush=True)
    return g


def main():
    g = exec_prefix()
    x_train = g["x_train"]
    y_ord_train = g["y_ord_train"]
    categorical_columns = list(g["categorical_columns"])
    ordinal_columns = list(g["ordinal_columns"])
    numerical_columns = list(g["numerical_columns"])

    print(
        f"\ninputs: {len(categorical_columns)} categorical, {len(ordinal_columns)} ordinal, "
        f"{len(numerical_columns)} numerical, {x_train.shape[0]} train rows"
    )
    print("3-class target value counts (train):")
    print(y_ord_train.value_counts().sort_index().to_string())

    x_small = x_train.iloc[:N_ROWS]
    y_small = y_ord_train.iloc[:N_ROWS]
    cats = categorical_columns[:N_CAT] + ordinal_columns[:N_ORD]
    nums = numerical_columns[:N_NUM]
    variable_names = cats + nums
    print(
        f"\nprobe subsample: {len(x_small)} rows, {len(cats)} categorical/ordinal + "
        f"{len(nums)} numerical = {len(variable_names)} variables"
    )
    print("probe target classes:", sorted(y_small.unique()))

    from optbinning import BinningProcess

    for label, y in (
        ("3-class ordinal target (0 / 1 / 2+)", y_small),
        ("binary collapse y > 0", (y_small > 0).astype(int)),
    ):
        print("\n" + "=" * 78)
        print(f"### fit with {label}")
        print("=" * 78, flush=True)
        bp = BinningProcess(
            variable_names=variable_names,
            categorical_variables=cats,
            min_prebin_size=0.02,
            max_n_bins=5,
            n_jobs=6,
        )
        t0 = time.perf_counter()
        try:
            bp.fit(x_small[variable_names], y)
        except Exception:  # pylint: disable=broad-except
            print(f"FAILED after {time.perf_counter() - t0:.1f}s with:")
            traceback.print_exc(file=sys.stdout)
            continue
        print(f"fit OK in {time.perf_counter() - t0:.1f}s")
        try:
            summary = bp.summary()
            print(f"summary rows: {len(summary)}")
            print("status counts:")
            print(summary["status"].value_counts().to_string())
            print(summary.to_string())
        except Exception:  # pylint: disable=broad-except
            print("summary() raised:")
            traceback.print_exc(file=sys.stdout)
        try:
            out = bp.transform(x_small[variable_names], metric="bins")
            print(f"transform(metric='bins') OK -> {out.shape}")
            print(out.head(3).to_string())
        except Exception:  # pylint: disable=broad-except
            print("transform(metric='bins') raised:")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
