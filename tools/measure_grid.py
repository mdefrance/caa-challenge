"""Why did OrdinalSelector pick differently? Carve once, then vary the measures.

The two selectors accept disjoint measure families and cannot be crossed:

    ClassificationSelector  qualitative: Tschuprowt / Cramerv / Chi2
                            quantitative: KruskalEta / KruskalEps / Kruskal / R
    OrdinalSelector         qualitative: KruskalEta / KruskalEps / Kruskal
                            quantitative: Spearman / Pearson / Distance

So this varies within each class, and compares the two full candidate rankings.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# repository root, so the script runs from anywhere; CAA_REPO overrides it
REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "frequency_model_2026.ipynb")
OUT = os.path.join(REPO, "data", "ab_arms", "selector_measure_grid.csv")

SHIPPED = "Classification | Tschuprowt + KruskalEta  (shipped)"
RUN19 = "Ordinal        | KruskalEta + Spearman    (Run 19)"


def carve_once():
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "Selector" in src:
            break
        exec(compile(src, f"<cell {i}>", "exec"), g)
    return g


def main():
    g = carve_once()
    print("carved frame:", g["x_train"].shape, flush=True)

    from AutoCarver.selectors import (
        ClassificationSelector,
        CramervFilter,
        CramervMeasure,
        DistanceMeasure,
        KruskalEpsilonSquaredMeasure,
        OrdinalSelector,
        PearsonMeasure,
        SelectionConfig,
        SpearmanFilter,
    )

    qual, quant = g["carver"].features, g["quantitatives"]  # carved set on 7.7 and 7.8
    qual_names = set(qual.versions)
    print(
        f"candidates: {len(qual_names)} carved qualitative + {len(quant.versions)} raw quantitative",
        flush=True,
    )

    filters = dict(
        qualitative_filters=[CramervFilter(threshold=0.9)],
        quantitative_filters=[SpearmanFilter(threshold=0.9)],
    )
    grid = {
        SHIPPED: (ClassificationSelector, {}),
        "Classification | Cramerv    + KruskalEta": (
            ClassificationSelector,
            dict(qualitative_measures=[CramervMeasure()]),
        ),
        RUN19: (OrdinalSelector, {}),
        "Ordinal        | KruskalEps + Spearman": (
            OrdinalSelector,
            dict(qualitative_measures=[KruskalEpsilonSquaredMeasure()]),
        ),
        "Ordinal        | KruskalEta + Pearson": (
            OrdinalSelector,
            dict(quantitative_measures=[PearsonMeasure()]),
        ),
        "Ordinal        | KruskalEta + Distance": (
            OrdinalSelector,
            dict(quantitative_measures=[DistanceMeasure()]),
        ),
    }

    picks, summaries = {}, {}
    for label, (cls, measures) in grid.items():
        try:
            selector = cls(
                qual + quant,
                n_best_features=100,
                config=SelectionConfig(**filters, **measures),
            )
            selector.fit(g["x_train"], g["y_ord_train"])
        except Exception as exc:
            print(f"{label:44s} SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            continue
        chosen = list(selector.selected_features.names)
        picks[label] = chosen
        summaries[label] = selector.summary
        n_qual = sum(1 for f in chosen if f in qual_names)
        print(
            f"{label:44s} {n_qual:3d} qual / {len(chosen) - n_qual:3d} quant",
            flush=True,
        )

    base = set(picks[SHIPPED])
    print("\noverlap with the shipped pick:")
    for label, chosen in picks.items():
        print(f"  {label:44s} {len(base & set(chosen)):3d} / 100")

    rows = []
    for label, summary in summaries.items():
        cols = [
            c
            for c in ("feature", "measure", "association", "rank", "selected")
            if c in summary.columns
        ]
        s = summary[cols].copy()
        s["feature"] = s["feature"].astype(str)
        s.insert(0, "config", label)
        rows.append(s)
    allrows = pd.concat(rows, ignore_index=True)
    allrows.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(allrows)} rows)")

    if "rank" not in allrows.columns:
        print("no rank column in summary; columns =", list(summaries[SHIPPED].columns))
        return

    wide = (
        allrows[allrows.config.isin([SHIPPED, RUN19])]
        .pivot_table(index="feature", columns="config", values="rank", aggfunc="min")
        .dropna()
    )
    print(f"\nrank agreement across {len(wide)} candidates scored by both:")
    print(
        f"  Spearman rank correlation: {wide[SHIPPED].corr(wide[RUN19], method='spearman'):.4f}"
    )
    shift = (wide[SHIPPED] - wide[RUN19]).abs()
    print(
        f"  median |rank shift| {shift.median():.1f} | 90th pct {shift.quantile(0.9):.1f} | max {shift.max():.0f}"
    )

    # where the budget actually bites: the features that swap in and out at the cut
    a, b = set(picks[SHIPPED]), set(picks[RUN19])
    for name, diff, other in (
        ("only shipped", a - b, RUN19),
        ("only Run 19", b - a, SHIPPED),
    ):
        sub = (
            wide.loc[[f for f in wide.index if f.split("'")[1] in diff]]
            if len(wide)
            else wide
        )
        if len(sub):
            print(f"\n{name} ({len(diff)}) — their rank under each config:")
            print(sub.sort_values(SHIPPED).to_string())


if __name__ == "__main__":
    main()
