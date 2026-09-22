"""Summarise the BDIFF extract behind the article's section 1 figures into a committed JSON.

`data/Incendies.csv` is the fire-history extract joined in by
`src/data_toolkit.py::add_wildfire_features`. It is gitignored like every other data file,
so the claims in section 1 that describe it would otherwise be uncheckable from a clean
clone.
This script writes the handful of aggregates the article quotes to
`data/bdiff_extract_summary.json`, which *is* committed, and
`tools/check_article_numbers.py` asserts the article against that file -- re-deriving it
from the raw CSV as well whenever the CSV happens to be present.

    uv run --no-sync python tools/summarise_bdiff.py

The extract is a snapshot: BDIFF is updated continuously, so a fresh export will not
reproduce these figures. The one behind the article is mirrored on Kaggle (data/README.md).
"""

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
INCENDIES_CSV = REPO / "data" / "Incendies.csv"
SUMMARY_JSON = REPO / "data" / "bdiff_extract_summary.json"


def summarise(path: Path = INCENDIES_CSV) -> dict:
    """The aggregates the article quotes, read as `data_toolkit` reads the file."""
    fires = pd.read_csv(path, sep=",")

    def column(prefix: str, suffix: str = "") -> str:
        """Headers are accented French; match on their ASCII parts to stay encoding-proof."""
        (name,) = [c for c in fires.columns
                   if c.startswith(prefix) and c.endswith(suffix)]
        return name

    year = column("Ann")          # Annee
    burnt = column("Surface parcourue")
    dept = column("D", "partement")

    by_year = fires.groupby(year)[burnt].sum() / 1e4  # m2 -> hectares
    peak_year = int(by_year.idxmax())

    return {
        "source": "data/Incendies.csv (BDIFF extract, Etalab Licence Ouverte 2.0)",
        "n_fires": int(len(fires)),
        "n_departements": int(fires[dept].nunique()),
        "year_min": int(fires[year].min()),
        "year_max": int(fires[year].max()),
        "hectares_total": round(float(by_year.sum()), 1),
        "peak_year": peak_year,
        "hectares_peak_year": round(float(by_year.loc[peak_year]), 1),
        "hectares_by_year": {str(y): round(float(h), 1) for y, h in by_year.items()},
        "natures": {
            str(k): int(v)
            for k, v in fires["Nature"].fillna("Unknown").value_counts().items()
        },
    }


def main() -> None:
    summary = summarise()
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"wrote {SUMMARY_JSON.relative_to(REPO)}")
    for key in ("n_fires", "n_departements", "year_min", "year_max",
                "hectares_total", "peak_year", "hectares_peak_year"):
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
