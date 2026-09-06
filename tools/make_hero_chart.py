"""Build the article's hero chart: what supervised binning actually does to one feature.

Carves a single feature with the pinned AutoCarver, exactly as `frequency_model_2026.ipynb`
does, and plots its raw modalities against the buckets the carver assigned them to.

The picture is the argument. On the default feature, `SURFACE4` (floor area), sixteen area
bands carry a claim count that climbs roughly tenfold, and the carver cuts them in two at
1000 m². The lower panel shows why it stops there: every band above 4500 m² sits under
`min_freq`, so their apparent rates rest on a handful of policies. Nobody chose that cut.

Runs in ~40 s -- it reads three columns and carves one feature:

    uv run --no-sync python tools/make_hero_chart.py
    uv run --no-sync python tools/make_hero_chart.py --feature ACTIVIT2 --kind categorical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split

from AutoCarver import Features, OrdinalCarver
from AutoCarver.discretizers import ProcessingConfig

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "docs"

# Floor area, 16 declared-ordinal bands. The strongest tau-c of any raw column in the
# frequency model, and the clearest picture of what carving does: a monotone rise, one cut,
# and a long tail of bands too thin to cut on.
DEFAULT_FEATURE = "SURFACE4"
DEFAULT_KIND = "ordinal"

# Human-readable names, from the challenge's own data dictionary (Descriptif_donnees.xlsx).
# The modality codes themselves are anonymised by the insurer; the variable meaning is not.
LABELS = {
    "ACTIVIT2": "Activité — business activity of the insured entity",
    "SURFACE4": "Données de surface — floor area, m²",
    "SURFACE6": "Données de surface — floor area, m²",
    "TAILLE1": "Taille de risque — insured value band",
    "TMM_VOR_MMAX_A": "Données de température — highest monthly mean, °C",
    "NBJRR10_MMAX_A": "Données de pluie — days with ≥10 mm of rain, monthly max",
}

# A few ordinals are stored as strings that do not sort into their own order
# ('500' lands after '4500'). These mirror the order the notebooks declare.
ORDINAL_LEVELS = {
    "SURFACE4": [
        "0",
        "500",
        "1000",
        "1500",
        "2000",
        "2500",
        "3000",
        "3500",
        "4000",
        "4500",
        "5000",
        "5500",
        "6000",
        "6500",
        "7000",
        "7000+",
    ],
}
ORDINAL_LEVELS["SURFACE6"] = ORDINAL_LEVELS["SURFACE4"]

# carving config, identical to frequency_model_2026.ipynb
MIN_FREQ = 0.02
MAX_N_MOD = 5
MIN_FREQ_ALPHA = 0.05

# palette: one colour per carved bucket, ordered low -> high claim rate
BUCKET_COLOURS = ["#1b3a5c", "#2f6f9f", "#63a6c8", "#a8cfe0", "#dbeaf2"]
RAW_EDGE = "#ffffff"
TEXT = "#12222f"
MUTED = "#6b7f8c"


def load(feature: str) -> tuple[pd.Series, pd.Series]:
    """Loads just what is needed, in the notebooks' row order, and rebuilds their target."""
    x = pd.read_csv(
        DATA / "train_input_Z61KlZo.csv",
        usecols=["ID", feature, "ANNEE_ASSURANCE"],
        low_memory=False,
    ).set_index("ID")
    y = pd.read_csv(
        DATA / "train_output_DzPxaPY.csv",
        usecols=["ID", "FREQ", "ANNEE_ASSURANCE"],
        low_memory=False,
    ).set_index("ID")

    data = x.join(y.drop("ANNEE_ASSURANCE", axis=1))
    target = (data["FREQ"] * data["ANNEE_ASSURANCE"]).astype(int)
    return data[[feature]], target


def collapse_count(counts: pd.Series) -> pd.Series:
    """0 / 1 / 2+, the notebooks' ordinal target."""
    return counts.clip(upper=2)


def carve(frame: pd.DataFrame, y_ordinal: pd.Series, feature: str, kind: str):
    """Carves the one feature on the notebooks' train/dev split and returns the carver.

    The split is positional and seeded, so loading a subset of columns reproduces the
    notebooks' split exactly as long as the row order matches.
    """
    x_train, x_dev, y_train, y_dev = train_test_split(
        frame,
        y_ordinal,
        test_size=0.2,
        random_state=42,
        stratify=y_ordinal,
    )

    if kind == "ordinal":
        # declared order, exactly as the notebooks build ordinal_columns
        seen = set(frame[feature].dropna().unique())
        levels = (
            [lvl for lvl in ORDINAL_LEVELS[feature] if lvl in seen]
            if feature in ORDINAL_LEVELS
            else list(frame[feature].value_counts().sort_index().index)
        )
        features = Features(ordinals={feature: levels})
    else:
        # a categorical has no declared order; the carver sorts its modalities by target
        # rate before searching, so plot them in that order too
        levels = list(y_train.groupby(x_train[feature]).mean().sort_values().index)
        features = Features(categoricals=[feature])
    carver = OrdinalCarver(
        features=features,
        target_scale="level",
        min_freq=MIN_FREQ,
        max_n_mod=MAX_N_MOD,
        config=ProcessingConfig(
            dropna=False, copy=True, verbose=False, min_freq_alpha=MIN_FREQ_ALPHA
        ),
    )
    carver.fit(x_train, y_train, X_dev=x_dev, y_dev=y_dev)
    return carver, levels, x_train, y_train


def bucket_of_each_level(carver, feature: str, levels: list) -> pd.Series:
    """Maps every raw level to the bucket label the carver assigned it.

    Read off the fitted carver by transforming a one-row-per-level frame, so this does not
    depend on the internal representation of a grouping.
    """
    probe = pd.DataFrame({feature: levels})
    carved = carver.transform(probe)[feature]
    return pd.Series(carved.to_numpy(), index=pd.Index(levels, name=feature))


def build(feature: str, kind: str) -> Path:
    frame, target = load(feature)
    y_ordinal = collapse_count(target)
    carver, levels, x_train, y_train = carve(frame, y_ordinal, feature, kind)

    assignment = bucket_of_each_level(carver, feature, levels)

    per_level = pd.DataFrame(
        {
            "share": x_train[feature].value_counts(normalize=True),
            "claim_rate": y_train.groupby(x_train[feature]).mean(),
        }
    ).reindex(levels)
    per_level["bucket"] = assignment

    # buckets ordered by their own claim rate, so the colour ramp reads low -> high
    per_bucket = (
        pd.DataFrame(
            {
                "share": x_train[feature].map(assignment).value_counts(normalize=True),
                "claim_rate": y_train.groupby(x_train[feature].map(assignment)).mean(),
            }
        )
        .dropna()
        .sort_values("claim_rate")
    )
    order = {name: i for i, name in enumerate(per_bucket.index)}
    colours = {
        name: BUCKET_COLOURS[i % len(BUCKET_COLOURS)] for name, i in order.items()
    }

    # Levels absent from train carry no rate; drop them rather than leave gaps.
    per_level = per_level.dropna(subset=["claim_rate"])

    if kind == "categorical":
        # A categorical has no meaningful x-order, and the carver ranks its modalities on
        # its own ordinal-level score -- close to, but not identical to, the mean plotted
        # here. Sorting on the plotted mean alone interleaves the buckets and hides the
        # very thing the chart is about, so lay them out bucket by bucket instead.
        per_level = per_level.assign(
            _bucket_rank=per_level["bucket"].map(order)
        ).sort_values(["_bucket_rank", "claim_rate"])
    else:
        # An ordinal keeps its declared order: that order *is* the feature's meaning, and
        # the carver only ever merges adjacent levels, so the buckets come out contiguous.
        per_level = per_level.loc[[lvl for lvl in levels if lvl in per_level.index]]

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    positions = np.arange(len(per_level))

    # --- top: claim rate per raw level, coloured by assigned bucket -------------------
    top.bar(
        positions,
        per_level["claim_rate"],
        color=[colours.get(b, MUTED) for b in per_level["bucket"]],
        edgecolor=RAW_EDGE,
        linewidth=0.6,
        zorder=2,
    )

    # the bucket's own claim rate, drawn as a step across the levels it swallowed
    for name, group in per_level.groupby("bucket", sort=False):
        idx = [positions[i] for i, b in enumerate(per_level["bucket"]) if b == name]
        if not idx:
            continue
        rate = (
            per_bucket.loc[name, "claim_rate"] if name in per_bucket.index else np.nan
        )
        top.plot(
            [min(idx) - 0.45, max(idx) + 0.45],
            [rate, rate],
            color=TEXT,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=3,
        )

    top.set_ylabel("average claim count", color=TEXT, fontsize=11)
    top.tick_params(axis="y", colors=TEXT)
    top.spines[["top", "right"]].set_visible(False)
    top.spines[["left", "bottom"]].set_color(MUTED)
    top.grid(axis="y", color="#e6ecef", linewidth=0.8, zorder=0)
    top.set_axisbelow(True)

    n_buckets = per_level["bucket"].nunique()
    top.set_title(
        f"{feature}: {len(levels)} raw levels become {n_buckets} buckets",
        loc="left",
        fontsize=15,
        color=TEXT,
        pad=42,
        fontweight="bold",
    )
    subtitle = LABELS.get(feature)
    if subtitle:
        top.text(0, 1.052, subtitle, transform=top.transAxes, fontsize=10.5, color=TEXT)
    top.text(
        0,
        1.015,
        "bars: mean claim count of each raw level, coloured by the bucket it was merged "
        "into  ·  black rule: the bucket's own mean  ·  the quantity the carver ranks on",
        transform=top.transAxes,
        fontsize=9.5,
        color=MUTED,
    )

    # --- bottom: how much of the portfolio each level actually carries ----------------
    bottom.bar(
        positions,
        per_level["share"],
        color=[colours.get(b, MUTED) for b in per_level["bucket"]],
        edgecolor=RAW_EDGE,
        linewidth=0.6,
        alpha=0.75,
        zorder=2,
    )
    bottom.axhline(MIN_FREQ, color="#c0392b", linewidth=1.2, linestyle="--", zorder=3)
    bottom.text(
        len(positions) - 0.5,
        MIN_FREQ,
        f"  min_freq = {MIN_FREQ:.0%}",
        va="center",
        ha="left",
        fontsize=9,
        color="#c0392b",
    )
    # log scale: one dominant modality would otherwise flatten every rare level onto
    # zero, and the rare levels are exactly what min_freq is about
    bottom.set_yscale("log")
    bottom.set_ylabel("share of\nportfolio", color=TEXT, fontsize=10)
    bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    bottom.tick_params(axis="both", colors=TEXT)
    bottom.spines[["top", "right"]].set_visible(False)
    bottom.spines[["left", "bottom"]].set_color(MUTED)
    bottom.grid(axis="y", color="#e6ecef", linewidth=0.8, zorder=0)
    bottom.set_axisbelow(True)

    step = max(1, len(levels) // 25)
    bottom.set_xticks(positions[::step])
    bottom.set_xticklabels(
        [str(v) for v in per_level.index[::step]], rotation=0, fontsize=9
    )
    bottom.set_xlabel(feature, color=TEXT, fontsize=11, labelpad=8)
    bottom.set_xlim(-0.8, len(positions) - 0.2)

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"hero_{feature}"
    fig.savefig(
        stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"{feature}: {len(levels)} raw levels -> {n_buckets} buckets")
    print(per_bucket.to_string())
    print("wrote", stem.with_suffix(".png"), "and", stem.with_suffix(".svg"))
    return stem.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument(
        "--kind", default=DEFAULT_KIND, choices=["ordinal", "categorical"]
    )
    args = parser.parse_args()
    build(args.feature, args.kind)


if __name__ == "__main__":
    main()
