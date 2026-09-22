"""Build the article's hero chart: what supervised binning actually does to one feature.

Carves a single feature with the pinned AutoCarver, exactly as `frequency_model_2026.ipynb`
does, and plots its raw modalities against the buckets the carver assigned them to.

The picture is the argument. On the default feature, `SURFACE4` (floor area), sixteen area
bands carry a claim count that climbs roughly tenfold, and the carver cuts them in two at
1000 m². The lower panel is about trust, not about the cut: the bands from 4500 m² to
7000 m² sit under `min_freq`, so their apparent rates rest on a handful of policies.
(`7000+` does clear it -- it is the catch-all.) `min_freq` is not what produced two
buckets; `--explain` shows what did. Nobody chose that cut.

The badge quantifies it: Tschuprow's T between the feature and the target, before and
after carving, on train and on dev. T normalises chi-square by the table's degrees of
freedom, so collapsing thin, noisy levels can *raise* it -- which is the whole claim, and
the reason the badge is only drawn when the carved value actually comes out higher.

Outputs, all written to docs/:

  hero_<F>.svg       transparent, and theme-aware -- every ink colour is a CSS variable
                     with a `prefers-color-scheme: dark` override injected into the file
  hero_<F>.png       opaque light, for platforms that will not take an SVG
  hero_<F>_dark.png  the same on a dark ground

Runs in ~40 s -- it reads three columns and carves one feature:

    uv run --no-sync python tools/make_hero_chart.py
    uv run --no-sync python tools/make_hero_chart.py --feature ACTIVIT2 --kind categorical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects
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
    "ACTIVIT2": "ACTIVIT2: business activity of the insured entity",
    "SURFACE4": "SURFACE4: floor area, m²",
    "SURFACE6": "SURFACE6: floor area, m²",
    "TAILLE1": "TAILLE1: insured value band",
    "TMM_VOR_MMAX_A": "TMM_VOR_MMAX_A: highest monthly mean temperature, °C",
    "NBJRR10_MMAX_A": "NBJRR10_MMAX_A: days with ≥10 mm of rain, monthly max",
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

# Two palettes, same keys. The light one doubles as the set of sentinel hex strings the
# SVG post-processor swaps for CSS variables, so every colour that ends up in the figure
# has to come from here -- a hard-coded hex anywhere below would silently stay light.
# Buckets are an IDENTITY encoding -- which supervised bin a raw level was merged into --
# so the slots are distinct hues in a fixed order, not steps of one ramp. The earlier ramp
# put the two realised buckets two steps apart on the same blue, which readers reported as
# indistinguishable; it also failed a lightness band, a chroma floor and the normal-vision
# separation floor. These steps are the reference categorical order (blue, orange, aqua,
# violet, magenta), validated for adjacent pairs in both modes -- adjacent is the pairlist
# that matters here, because buckets are contiguous spans over an ordered feature, so only
# neighbouring buckets ever touch. Worst adjacent pair: CVD dE 9.2 light / 9.4 dark against
# an >= 8 target, normal-vision 27.6 / 19.7 against an >= 15 floor. The two hues this feature
# actually uses are 33.6 (light) and 31.8 (dark) apart.
LIGHT = {
    "b0": "#2a78d6",  # bucket slots, fixed order -- distinct hues, never cycled
    "b1": "#eb6834",
    "b2": "#1baf7a",
    "b3": "#4a3aa7",
    "b4": "#e87ba4",
    "edge": "#ffffff",  # bar edges: the page behind the chart
    "ink": "#12222f",
    "muted": "#6b7f8c",
    "grid": "#e6ecef",
    "accent": "#5a6b76",  # threshold rule: an annotation, so neutral, not a data hue
    "ground": "#ffffff",  # PNG background only; the SVG is transparent
}
DARK = {
    "b0": "#3987e5",
    "b1": "#d95926",
    "b2": "#199e70",
    "b3": "#9085e9",
    "b4": "#d55181",
    "edge": "#0d1117",
    "ink": "#e6edf3",
    "muted": "#8b98a5",
    "grid": "#2b3138",
    "accent": "#9aa7b2",
    "ground": "#0d1117",
}
BUCKET_KEYS = ["b0", "b1", "b2", "b3", "b4"]


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


def tschuprow_t(a: pd.Series, b: pd.Series) -> tuple[float, tuple[int, int]]:
    """Tschuprow's T between two categorical series, plus the crosstab's shape.

    T = sqrt( chi2 / (n * sqrt((r-1)(c-1))) ). Unlike Cramer's V it divides by the
    geometric mean of the degrees of freedom rather than the smaller dimension, so a
    table with many thin rows is penalised for having them. That is exactly why it is
    the interesting measure here: merging levels that carry no signal can raise it.

    chi2 is computed inline rather than pulled from scipy -- four lines, no dependency
    beyond what the notebooks already import, and the arithmetic stays visible.
    """
    ct = pd.crosstab(a, b)
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    r, c = ct.shape
    if min(r, c) < 2:
        return float("nan"), (r, c)
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * np.sqrt((r - 1) * (c - 1))))), (r, c)


def association(x_train, y_train, x_dev, y_dev, feature, assignment) -> dict:
    """Raw-vs-carved Tschuprow's T on both halves of the notebooks' split.

    Train is the half the carver searched, so a rise there is close to tautological --
    the carver is picking the grouping that maximises association. Dev is the honest
    number, with one caveat worth stating rather than hiding: `fit(..., X_dev=, y_dev=)`
    lets the carver *reject* a grouping that does not hold up on dev. It never chooses
    from dev, but dev is not untouched either.
    """
    out = {}
    for name, X, Y in (("train", x_train, y_train), ("dev", x_dev, y_dev)):
        raw = X[feature]
        t_raw, shape_raw = tschuprow_t(raw, Y)
        t_cut, shape_cut = tschuprow_t(raw.map(assignment), Y)
        out[name] = {
            "raw": t_raw,
            "carved": t_cut,
            "n_raw": shape_raw[0],
            "n_carved": shape_cut[0],
            "ratio": t_cut / t_raw if t_raw else float("nan"),
        }
    return out


def draw(per_level, per_bucket, colour_key, assoc, feature, levels, palette):
    """Renders the figure for one palette. Every colour is read from `palette`."""
    ink, muted = palette["ink"], palette["muted"]
    colours = {
        name: palette[BUCKET_KEYS[i % len(BUCKET_KEYS)]] for name, i in colour_key.items()
    }

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    # Transparent by default; the PNG writers paint a ground back in. The axes patches
    # stay transparent either way, so whatever is behind the figure shows through.
    fig.patch.set_alpha(0.0)
    for ax in (top, bottom):
        ax.patch.set_alpha(0.0)

    positions = np.arange(len(per_level))

    # --- top: claim rate per raw level, coloured by assigned bucket -------------------
    top.bar(
        positions,
        per_level["claim_rate"],
        color=[colours.get(b, muted) for b in per_level["bucket"]],
        edgecolor=palette["edge"],
        linewidth=0.6,
        zorder=2,
    )

    # the bucket's own claim rate, drawn as a step across the levels it swallowed
    for name, _group in per_level.groupby("bucket", sort=False):
        idx = [positions[i] for i, b in enumerate(per_level["bucket"]) if b == name]
        if not idx:
            continue
        rate = per_bucket.loc[name, "claim_rate"] if name in per_bucket.index else np.nan
        top.plot(
            [min(idx) - 0.45, max(idx) + 0.45],
            [rate, rate],
            color=ink,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=3,
        )
        # Direct label, sitting on the span it names: identity is then carried by text and
        # position as well as by hue, which is what the palette's light-mode contrast
        # check requires and what makes the figure survive a greyscale print.
        if np.isfinite(rate):
            span = per_level.index[min(idx)], per_level.index[max(idx)]
            label = f"{span[0]}" if span[0] == span[1] else f"{span[0]}–{span[1]}"
            # Anchored to the line, haloed in the surface colour: a bucket's own bars can
            # rise above its mean, and a bare label would then sit unreadable on one. A
            # halo keeps the bar's silhouette intact where a filled box would notch it.
            top.text(
                min(idx) - 0.4,
                rate,
                f" {label} · {rate:.4f} ",
                va="bottom",
                ha="left",
                fontsize=9.5,
                color=ink,
                zorder=4,
                path_effects=[
                    patheffects.withStroke(linewidth=3.0, foreground=palette["ground"])
                ],
            )

    top.set_ylabel("average claim count", color=ink, fontsize=11)
    top.tick_params(axis="y", colors=ink)
    top.spines[["top", "right"]].set_visible(False)
    top.spines[["left", "bottom"]].set_color(muted)
    top.grid(axis="y", color=palette["grid"], linewidth=0.8, zorder=0)
    top.set_axisbelow(True)

    n_buckets = per_level["bucket"].nunique()
    top.set_title(
        f"{feature}: {len(levels)} raw levels become {n_buckets} buckets",
        loc="left",
        fontsize=15,
        color=ink,
        pad=42,
        fontweight="bold",
    )
    subtitle = LABELS.get(feature)
    if subtitle:
        top.text(0, 1.052, subtitle, transform=top.transAxes, fontsize=10.5, color=ink)
    top.text(
        0,
        1.015,
        "bars: mean claim count per raw level  ·  line: mean of the bucket it was "
        "merged into",
        transform=top.transAxes,
        fontsize=9.5,
        color=muted,
    )

    # --- the badge: did carving actually buy association? ----------------------------
    # Drawn only when it did. A chart that reported a decrease under the same heading,
    # as though it were still the point, would be worse than no badge at all.
    dev, train = assoc["dev"], assoc["train"]
    if dev["carved"] > dev["raw"]:
        # Open a band of headroom above the tallest bar and stop the y ticks below it, so
        # no gridline runs through the badge. Extending the limit alone is not enough --
        # the locator would just place another tick inside the new space.
        ymax = float(per_level["claim_rate"].max())
        top.set_yticks([t for t in top.get_yticks() if 0 <= t <= ymax * 1.02])
        top.set_ylim(0, ymax * 1.36)

        top.text(
            0.007,
            0.99,
            "Tschuprow's T, feature vs target",
            transform=top.transAxes,
            va="top",
            fontsize=10,
            color=muted,
        )
        rows = (
            f"                dev      train\n"
            f"{dev['n_raw']:>3d} raw levels  {dev['raw']:.4f}   {train['raw']:.4f}\n"
            f"{dev['n_carved']:>3d} buckets     {dev['carved']:.4f}   {train['carved']:.4f}\n"
            f"               ×{dev['ratio']:.2f}    ×{train['ratio']:.2f}"
        )
        top.text(
            0.007,
            0.925,
            rows,
            transform=top.transAxes,
            va="top",
            fontsize=9.5,
            color=ink,
            family="DejaVu Sans Mono",
            linespacing=1.55,
        )

    # --- bottom: how much of the portfolio each level actually carries ----------------
    bottom.bar(
        positions,
        per_level["share"],
        color=[colours.get(b, muted) for b in per_level["bucket"]],
        edgecolor=palette["edge"],
        linewidth=0.6,
        alpha=0.75,
        zorder=2,
    )
    bottom.axhline(
        MIN_FREQ, color=palette["accent"], linewidth=1.2, linestyle="--", zorder=3
    )
    bottom.text(
        len(positions) - 0.5,
        MIN_FREQ,
        f"  min_freq = {MIN_FREQ:.0%}",
        va="center",
        ha="left",
        fontsize=9,
        color=palette["accent"],
    )
    # log scale: one dominant modality would otherwise flatten every rare level onto
    # zero, and the rare levels are exactly what min_freq is about
    bottom.set_yscale("log")
    bottom.set_ylabel("share of\nportfolio", color=ink, fontsize=10)
    bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    bottom.tick_params(axis="both", colors=ink)
    bottom.spines[["top", "right"]].set_visible(False)
    bottom.spines[["left", "bottom"]].set_color(muted)
    bottom.grid(axis="y", color=palette["grid"], linewidth=0.8, zorder=0)
    bottom.set_axisbelow(True)

    step = max(1, len(levels) // 25)
    bottom.set_xticks(positions[::step])
    bottom.set_xticklabels(
        [str(v) for v in per_level.index[::step]], rotation=0, fontsize=9
    )
    bottom.set_xlabel(feature, color=ink, fontsize=11, labelpad=8)
    bottom.set_xlim(-0.8, len(positions) - 0.2)

    # sharex means the top axes draws its own x tick marks too, labels hidden. Left at
    # the default they come out black, which is invisible on a dark ground.
    for ax in (top, bottom):
        ax.tick_params(axis="both", which="both", colors=ink)
    return fig


def themed_svg(path: Path) -> None:
    """Rewrites a saved SVG so it follows the reader's light/dark preference.

    Matplotlib writes colours as literal hex into style attributes. Every one of them
    came from LIGHT, so swapping each for `var(--hero-<key>)` and injecting a stylesheet
    that defines those variables twice -- once plainly, once under
    `prefers-color-scheme: dark` -- gives one file that renders on either ground. It
    works even when the SVG is embedded with <img>, which is how a Markdown image loads.

    Caveat worth knowing: this follows the *operating system* preference. A reader whose
    OS is light but who has forced a site into dark mode gets the light chart on a dark
    page. Nothing inside a standalone SVG can see a host page's own theme toggle.
    """
    svg = path.read_text(encoding="utf-8")
    for key, hex_value in LIGHT.items():
        if key == "ground":  # never drawn: the SVG is transparent
            continue
        svg = svg.replace(hex_value, f"var(--hero-{key})")

    light_vars = "\n".join(
        f"    --hero-{k}: {v};" for k, v in LIGHT.items() if k != "ground"
    )
    dark_vars = "\n".join(
        f"      --hero-{k}: {v};" for k, v in DARK.items() if k != "ground"
    )
    style = (
        "<style>\n"
        "  :root {\n"
        f"{light_vars}\n"
        "  }\n"
        "  @media (prefers-color-scheme: dark) {\n"
        "    :root {\n"
        f"{dark_vars}\n"
        "    }\n"
        "  }\n"
        "</style>\n"
    )
    # Matplotlib's own default for anything not given an explicit colour is black, and
    # every such element here is ink by role -- tick marks, mainly. Sweeping the leftovers
    # keeps one missed keyword argument from shipping an invisible mark on a dark ground.
    svg = svg.replace("#000000", "var(--hero-ink)")

    marker = "</defs>"
    if marker in svg:
        svg = svg.replace(marker, marker + "\n" + style, 1)
    else:  # no <defs>: fall back to just after the opening <svg ...> tag
        cut = svg.index(">", svg.index("<svg")) + 1
        svg = svg[:cut] + "\n" + style + svg[cut:]
    path.write_text(svg, encoding="utf-8")


def build(feature: str, kind: str) -> Path:
    frame, target = load(feature)
    y_ordinal = collapse_count(target)
    carver, levels, x_train, y_train = carve(frame, y_ordinal, feature, kind)

    assignment = bucket_of_each_level(carver, feature, levels)

    # the same split again, this time keeping the dev half, so the badge can report the
    # association on data the carver did not search
    _, x_dev, _, y_dev = train_test_split(
        frame, y_ordinal, test_size=0.2, random_state=42, stratify=y_ordinal
    )
    assoc = association(x_train, y_train, x_dev, y_dev, feature, assignment)

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
    colour_key = {name: i for i, name in enumerate(per_bucket.index)}

    # Levels absent from train carry no rate; drop them rather than leave gaps.
    per_level = per_level.dropna(subset=["claim_rate"])

    if kind == "categorical":
        # A categorical has no meaningful x-order, and the carver ranks its modalities on
        # its own ordinal-level score -- close to, but not identical to, the mean plotted
        # here. Sorting on the plotted mean alone interleaves the buckets and hides the
        # very thing the chart is about, so lay them out bucket by bucket instead.
        per_level = per_level.assign(
            _bucket_rank=per_level["bucket"].map(colour_key)
        ).sort_values(["_bucket_rank", "claim_rate"])
    else:
        # An ordinal keeps its declared order: that order *is* the feature's meaning, and
        # the carver only ever merges adjacent levels, so the buckets come out contiguous.
        per_level = per_level.loc[[lvl for lvl in levels if lvl in per_level.index]]

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"hero_{feature}"

    def opaque(fig, palette):
        """savefig's facecolor cannot beat an alpha of 0 -- set both, then save."""
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor(palette["ground"])

    fig = draw(per_level, per_bucket, colour_key, assoc, feature, levels, LIGHT)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    opaque(fig, LIGHT)
    fig.savefig(
        stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=LIGHT["ground"]
    )
    plt.close(fig)
    themed_svg(stem.with_suffix(".svg"))

    fig = draw(per_level, per_bucket, colour_key, assoc, feature, levels, DARK)
    opaque(fig, DARK)
    dark_png = OUT / f"hero_{feature}_dark.png"
    fig.savefig(dark_png, dpi=200, bbox_inches="tight", facecolor=DARK["ground"])
    plt.close(fig)

    n_buckets = per_level["bucket"].nunique()
    print(f"{feature}: {len(levels)} raw levels -> {n_buckets} buckets")
    print(per_bucket.to_string())
    for half, a in assoc.items():
        direction = "up" if a["carved"] > a["raw"] else "DOWN"
        print(
            f"Tschuprow's T {half:5s}: raw {a['raw']:.6f} ({a['n_raw']} levels) -> "
            f"carved {a['carved']:.6f} ({a['n_carved']}) = x{a['ratio']:.3f} {direction}"
        )
    if assoc["dev"]["carved"] <= assoc["dev"]["raw"]:
        print("badge omitted: carving did not raise dev association for this feature")
    print("wrote", stem.with_suffix(".svg"), stem.with_suffix(".png"), "and", dark_png)
    return stem.with_suffix(".png")


def explain(feature: str, kind: str) -> None:
    """Why does the carver stop where it stops? Separates `min_freq` from the measure.

    Two questions the chart cannot answer on its own, and that the article states an
    answer to, so they need checking rather than assuming:

    1. Does `min_freq` alone force the bucket count? Printing each raw level's share
       against the threshold answers it. On SURFACE4 ten of sixteen bands clear 2 %, so
       it does not: a grouping with more than two buckets was admissible.
    2. Was a finer grouping found on train and then rejected on dev? Fitting the same
       feature a second time *without* `X_dev`/`y_dev` answers it. If the train-only fit
       returns the same buckets, the dev check is not what produced them -- the carver's
       own association measure ranked that grouping highest to begin with.
    """
    frame, target = load(feature)
    y_ordinal = collapse_count(target)
    carver, levels, x_train, y_train = carve(frame, y_ordinal, feature, kind)
    assignment = bucket_of_each_level(carver, feature, levels)

    shares = x_train[feature].value_counts(normalize=True).reindex(levels)
    print(f"{feature}: raw level shares on train, against min_freq = {MIN_FREQ:.0%}")
    for level, share in shares.items():
        flag = "clears" if share >= MIN_FREQ else "under "
        print(f"  {str(level):>8s}  {share:8.5f}  {flag}")
    print(
        f"  {int((shares >= MIN_FREQ).sum())} of {len(levels)} levels clear min_freq, "
        f"and max_n_mod = {MAX_N_MOD}"
    )

    # the same fit without a dev set: does the dev check change the answer?
    if kind == "ordinal":
        features = Features(ordinals={feature: levels})
    else:
        features = Features(categoricals=[feature])
    train_only = OrdinalCarver(
        features=features,
        target_scale="level",
        min_freq=MIN_FREQ,
        max_n_mod=MAX_N_MOD,
        config=ProcessingConfig(
            dropna=False, copy=True, verbose=False, min_freq_alpha=MIN_FREQ_ALPHA
        ),
    )
    train_only.fit(x_train, y_train)
    without_dev = bucket_of_each_level(train_only, feature, levels)

    n_with, n_without = assignment.nunique(), without_dev.nunique()
    print(f"\nbuckets with dev-set validation: {n_with}")
    print(f"buckets fitted on train alone  : {n_without}")
    if n_with == n_without and assignment.equals(without_dev):
        print(
            "identical -- the dev check did not reject a finer grouping; the carver's "
            "association measure ranked this one highest on train in the first place"
        )
    else:
        print("different -- the dev check rejected the train-preferred grouping")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument(
        "--kind", default=DEFAULT_KIND, choices=["ordinal", "categorical"]
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print why the carver stopped at the bucket count it did, and exit",
    )
    args = parser.parse_args()
    if args.explain:
        explain(args.feature, args.kind)
        return
    build(args.feature, args.kind)


if __name__ == "__main__":
    main()
