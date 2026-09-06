"""Carve once, then run both selectors on the identical carved frame.

The notebook's cells resolve data as "../data/", so this runs from src/ exactly as
nbconvert does.
"""

import json
import os
import sys
from pathlib import Path

# repository root, so the script runs from anywhere; CAA_REPO overrides it
REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "frequency_model_2026.ipynb")


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())

    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "Selector" in src:  # stop right before feature selection
            break
        exec(compile(src, f"<cell {i}>", "exec"), g)
    print("carved frame:", g["x_train"].shape, flush=True)

    from AutoCarver.selectors import (
        ClassificationSelector,
        CramervFilter,
        OrdinalSelector,
        SelectionConfig,
        SpearmanFilter,
    )

    qual, quant = g["qualitatives"], g["quantitatives"]
    qual_names = set(qual.versions)
    print(
        f"candidates: {len(qual_names)} carved qualitative + {len(quant.versions)} raw quantitative",
        flush=True,
    )

    picks = {}
    for name, cls in [
        ("ClassificationSelector", ClassificationSelector),
        ("OrdinalSelector", OrdinalSelector),
    ]:
        config = SelectionConfig(
            qualitative_filters=[CramervFilter(threshold=0.9)],
            quantitative_filters=[SpearmanFilter(threshold=0.9)],
        )
        selector = cls(qual + quant, n_best_features=100, config=config)
        selector.fit(g["x_train"], g["y_ord_train"])
        chosen = list(selector.selected_features.names)
        n_qual = sum(1 for f in chosen if f in qual_names)
        picks[name] = chosen
        print(
            f"{name:24s} -> {len(chosen):3d} selected | "
            f"{n_qual} carved qualitative / {len(chosen) - n_qual} raw quantitative",
            flush=True,
        )

    a, b = set(picks["ClassificationSelector"]), set(picks["OrdinalSelector"])
    print(f"\noverlap: {len(a & b)} of 100 in common")
    print(f"only ClassificationSelector ({len(a - b)}): {sorted(a - b)[:15]}")
    print(f"only OrdinalSelector       ({len(b - a)}): {sorted(b - a)[:15]}")


if __name__ == "__main__":
    main()
