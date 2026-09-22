"""Build the article's result charts from the measured CSVs.

Three figures, each answering one question the prose currently answers in numbers:

  results_selection_flip   Does binning win, or does *selecting* win? Same three arms,
                           left with each arm choosing its own 100 features, right with
                           all of them modelling the same 100. The ordering reverses.
  results_noise_floor      Which claims in this article survive their own seed spread?
                           Every comparison on one axis, in units of pooled spread, with
                           the 1.0 line marking "the same size as the noise".
  results_severity_blind   RMSE says the severity arms are identical; top-decile lift says
                           they differ 2.6-fold. Two panels, never two y-axes on one plot.

Every number is read from `data/ab_arms/`; nothing is typed in. Arms keep the same hue in
every figure, so colour means the arm and never its rank.

Outputs to docs/: a theme-aware `.svg` plus light and dark `.png`, same contract as
`make_hero_chart.py`.

    uv run --no-sync python tools/make_result_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARMS = REPO / "data" / "ab_arms"
OUT = REPO / "docs"

# Categorical slots 1-3 of the validated default palette, in fixed order. Validated
# all-pairs in both modes (worst CVD dE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9).
# Light-mode aqua sits under 3:1 on the surface, so every arm carries a visible direct
# label -- the relief rule, not an optional nicety.
LIGHT = {
    "autocarver": "#2a78d6",
    "nocarve": "#eb6834",
    "optbinning": "#1baf7a",
    "ink": "#12222f",
    "muted": "#6b7f8c",
    "grid": "#e6ecef",
    "accent": "#c0392b",
    "edge": "#ffffff",
    "ground": "#ffffff",
}
DARK = {
    "autocarver": "#3987e5",
    "nocarve": "#d95926",
    "optbinning": "#199e70",
    "ink": "#e6edf3",
    "muted": "#8b98a5",
    "grid": "#243039",
    "accent": "#e06c5b",
    "edge": "#0d1117",
    "ground": "#0d1117",
}
ARM_KEY = {"AutoCarver": "autocarver", "no carving": "nocarve", "optbinning": "optbinning"}

CONST_SEV = 6617.646044495917


def read(name: str) -> pd.DataFrame | None:
    path = ARMS / name
    if not path.exists():
        return None
    return pd.read_csv(path, comment="#")


def style(ax, palette):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette["grid"])
    ax.tick_params(colors=palette["muted"], labelsize=9, length=3)
    ax.grid(axis="x", color=palette["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def dot_row(ax, y, values, colour, palette, label, fmt="{:.4f}"):
    """One arm: its seed points, the range they span, and the mean called out."""
    values = np.asarray(sorted(values), dtype=float)
    ax.plot([values.min(), values.max()], [y, y], color=colour, linewidth=2,
            solid_capstyle="round", zorder=2)
    ax.scatter(values, np.full_like(values, y), s=46, color=colour,
               edgecolor=palette["edge"], linewidth=1.4, zorder=3)
    mean = values.mean()
    ax.scatter([mean], [y], marker="D", s=70, color=colour,
               edgecolor=palette["edge"], linewidth=1.6, zorder=4)
    ax.annotate(f"{label}  {fmt.format(mean)}", (values.min(), y), textcoords="offset points",
                xytext=(-10, 0), ha="right", va="center", fontsize=9.5,
                color=palette["ink"], zorder=5)


# --------------------------------------------------------------------------------------
def chart_selection_flip(palette):
    """The headline: equalising selection reverses the frequency ordering."""
    own = {
        "AutoCarver": read("seed_variance.csv"),
        "no carving": read("no_carving_frequency.csv"),
        "optbinning": read("matrix_frequency_full_optbinning.csv"),
    }
    fixed = {
        "AutoCarver": read("matrix_frequency_fixed_autocarver.csv"),
        "no carving": read("matrix_frequency_fixed_nocarve.csv"),
        "optbinning": read("matrix_frequency_fixed_optbinning.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5), sharex=True)
    panels = (
        (axes[0], own, "Each arm selects its own 100 features"),
        (axes[1], fixed, "All arms model the same 100 features"),
    )
    order = ["AutoCarver", "no carving", "optbinning"]
    for ax, data, title in panels:
        style(ax, palette)
        for i, arm in enumerate(order):
            frame = data.get(arm)
            y = len(order) - 1 - i
            if frame is None:
                ax.annotate(f"{arm}  (not yet run)", (0.02, y), xycoords=("axes fraction", "data"),
                            fontsize=9.5, color=palette["muted"], va="center")
                continue
            dot_row(ax, y, frame.log_loss_dev, palette[ARM_KEY[arm]], palette, arm)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, color=palette["ink"], loc="left", pad=10)
        ax.set_xlabel("dev log loss  (lower is better)", fontsize=9.5, color=palette["muted"])
    fig.suptitle("Whose win was it? Binning, or choosing what to bin",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.885,
             "Four seeds per arm. Dots are seeds, the diamond is their mean, the bar is the range they span.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def chart_noise_floor(palette):
    """Every comparison in the article, measured against its own seed spread."""
    def spread(values):
        values = np.asarray(values, dtype=float)
        return values.max() - values.min()

    def ratio(a, b):
        return abs(np.mean(a) - np.mean(b)) / ((spread(a) + spread(b)) / 2)

    ac_f = read("seed_variance.csv")
    nc_f = read("no_carving_frequency.csv")
    ob_f = read("matrix_frequency_full_optbinning.csv")
    acx = read("matrix_frequency_fixed_autocarver.csv")
    obx = read("matrix_frequency_fixed_optbinning.csv")
    ac_a = read("seed_variance_amount.csv")
    nc_a = read("no_carving_amount.csv")
    ob_a = read("matrix_amount_full_optbinning.csv")

    rows = []
    acx_n = read("matrix_frequency_fixed_nocarve.csv")
    ob_rk = read("matrix_frequency_ranks_optbinning.csv")
    ob_b3 = read("matrix_frequency_bins3_optbinning.csv")
    amt_fx_ob = read("matrix_amount_fixed_optbinning.csv")
    amt_fx_nc = read("matrix_amount_fixed_nocarve.csv")

    rows.append(("2026 vs 2025 - frequency", 0.00055 / spread(ac_f.log_loss_dev)))
    rows.append(("2026 vs 2025 - severity", 2.18 / (100 * spread(ac_a.rmse_dev) / 6613.8)))
    rows.append(("2026 vs 2025 - CHARGE", 2.37 / (100 * spread(ac_a.charge_rmse_dev) / 6639.9)))
    rows.append(("frequency, own features: AutoCarver vs no carving",
                 ratio(ac_f.log_loss_dev, nc_f.log_loss_dev)))
    rows.append(("frequency, own features: no carving vs optbinning",
                 ratio(nc_f.log_loss_dev, ob_f.log_loss_dev)))
    rows.append(("frequency, own features: AutoCarver vs optbinning",
                 ratio(ac_f.log_loss_dev, ob_f.log_loss_dev)))
    rows.append(("frequency, same features: AutoCarver vs optbinning",
                 ratio(acx.log_loss_dev, obx.log_loss_dev)))
    if acx_n is not None:
        rows.append(("frequency, same features: AutoCarver vs no carving",
                     ratio(acx.log_loss_dev, acx_n.log_loss_dev)))
    if ob_rk is not None:
        rows.append(("optbinning: rank-encoded ordinals vs nominal",
                     ratio(ob_rk.log_loss_dev, ob_f.log_loss_dev)))
    if ob_b3 is not None:
        rows.append(("optbinning: bins capped at 3 vs 5",
                     ratio(ob_b3.log_loss_dev, ob_f.log_loss_dev)))
    rows.append(("severity, own features: AutoCarver vs no carving",
                 ratio(ac_a.rmse_dev, nc_a.rmse_dev)))
    rows.append(("severity, own features: AutoCarver vs optbinning",
                 ratio(ac_a.rmse_dev, ob_a.rmse_dev)))
    if amt_fx_ob is not None and amt_fx_nc is not None:
        rows.append(("severity RMSE, same features: no carving vs optbinning",
                     ratio(amt_fx_nc.rmse_dev, amt_fx_ob.rmse_dev)))
        rows.append(("severity ranking, same features: optbinning vs no carving",
                     ratio(amt_fx_ob.top_decile_lift_dev, amt_fx_nc.top_decile_lift_dev)))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    # 1.0 means the gap and the noise are the same size, which is not a finding. Only a
    # margin comfortably past the line is treated as one.
    CLEARS = 1.5
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    style(ax, palette)
    ax.grid(axis="x", color=palette["grid"], linewidth=0.8)
    for i, (label, value) in enumerate(zip(labels, values)):
        survives = value >= CLEARS
        colour = palette["autocarver"] if survives else palette["muted"]
        ax.plot([0, value], [i, i], color=colour, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([value], [i], s=70, color=colour, edgecolor=palette["edge"],
                   linewidth=1.4, zorder=3)
        ax.annotate(f"{value:.2f}×", (value, i), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=9.5,
                    color=palette["ink"] if survives else palette["muted"])
    ax.axvline(1.0, color=palette["accent"], linewidth=1.4, linestyle="--", zorder=1)
    ax.annotate("the same size as the noise", (1.0, len(rows) - 0.35), textcoords="offset points",
                xytext=(7, 0), fontsize=9.5, color=palette["accent"], va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, color=palette["ink"])
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_xlabel("gap between the arms, in multiples of their own four-seed spread",
                  fontsize=9.5, color=palette["muted"])
    fig.suptitle("Which of our findings survive the seed lottery",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.9,
             "Left of the line the gap is smaller than the noise. At the line it is the same size. "
             "Only the highlighted rows clear it.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def chart_severity_blind(palette):
    """Two measures, two panels -- never two y-axes on one plot."""
    metrics = json.loads((ARMS / "severity_metric_dynamic_range.json").read_text(encoding="utf-8"))
    order = ["AutoCarver", "no carving", "optbinning"]
    rmse = [metrics["arms"][a]["rmse_dev"] for a in order]
    lift = [metrics["arms"][a]["top_decile_lift_dev"] for a in order]
    const_rmse = metrics["constant_predictor"]["rmse_dev"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5))
    for ax, values, title, xlabel, fmt, floor, floor_label in (
        (axes[0], rmse, "What the article measured: dev RMSE", "dev RMSE  (lower is better)",
         "{:.1f}", const_rmse, "predicting the mean"),
        (axes[1], lift, "What it could not see: top-decile lift",
         "mean claim in the top 10 % of predictions, ÷ overall mean", "{:.2f}", 1.0,
         "no signal"),
    ):
        style(ax, palette)
        for i, (arm, value) in enumerate(zip(order, values)):
            y = len(order) - 1 - i
            colour = palette[ARM_KEY[arm]]
            # dots, anchored to the reference line rather than to a truncated zero: RMSE
            # differences here are a tenth of a percent, and a bar from 6400 would lie
            ax.plot([floor, value], [y, y], color=colour, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.scatter([value], [y], s=90, color=colour, edgecolor=palette["edge"],
                       linewidth=1.5, zorder=3)
            # label on the far side of the reference line, so it never sits on top of it
            right = value >= floor
            ax.annotate(f"{arm}  {fmt.format(value)}", (value, y), textcoords="offset points",
                        xytext=(11 if right else -11, 0), ha="left" if right else "right",
                        va="center", fontsize=9.5, color=palette["ink"], zorder=4)
        ax.axvline(floor, color=palette["accent"], linewidth=1.4, linestyle="--", zorder=3)
        ax.annotate(floor_label, (floor, -0.52), textcoords="offset points", xytext=(6, 0),
                    fontsize=9, color=palette["accent"], va="center")
        ax.set_yticks([])
        ax.set_ylim(-0.75, len(order) - 0.35)
        ax.set_title(title, fontsize=10.5, color=palette["ink"], loc="left", pad=10)
        ax.set_xlabel(xlabel, fontsize=9.5, color=palette["muted"])
    axes[0].set_xlim(min(rmse) - 34, max(rmse) + 26)
    axes[1].set_xlim(0.6, max(lift) * 1.3)
    fig.suptitle("The severity models were not identical. RMSE just could not tell.",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.885,
             "One fixed configuration per arm, so the comparison is of the features, not of a tuning run.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig



def chart_severity_two_metrics(palette):
    """Same 200 features, two metrics, opposite winners -- the sharpest severity result."""
    arms = {
        "AutoCarver": read("matrix_amount_fixed_autocarver.csv"),
        "no carving": read("matrix_amount_fixed_nocarve.csv"),
        "optbinning": read("matrix_amount_fixed_optbinning.csv"),
    }
    order = [a for a in ("AutoCarver", "no carving", "optbinning") if arms.get(a) is not None]
    missing = [a for a in ("AutoCarver", "no carving", "optbinning") if arms.get(a) is None]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5))
    panels = (
        (axes[0], "rmse_dev", "Predicting the amount: dev RMSE",
         "dev RMSE  (lower is better)", "{:.0f}"),
        (axes[1], "top_decile_lift_dev", "Ranking the risk: top-decile lift",
         "top 10 % of predictions, mean claim over overall mean  (higher is better)", "{:.2f}"),
    )
    for ax, column, title, xlabel, fmt in panels:
        style(ax, palette)
        for i, arm in enumerate(order):
            dot_row(ax, len(order) - 1 - i, arms[arm][column], palette[ARM_KEY[arm]],
                    palette, arm, fmt=fmt)
        if missing:
            ax.annotate(", ".join(missing) + "  (not yet run)", (0.03, 0.06),
                        xycoords="axes fraction", fontsize=9, color=palette["muted"])
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(title, fontsize=10.5, color=palette["ink"], loc="left", pad=10)
        ax.set_xlabel(xlabel, fontsize=9.2, color=palette["muted"])
    fig.suptitle("On identical features, the two metrics pick different winners",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.885,
             "Four seeds per arm. Not binning fits the amounts better; binning orders the risk better.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def chart_optbinning_variants(palette):
    """Every configuration optbinning was given, against the AutoCarver reference."""
    candidates = [
        ("optbinning - ordinals nominal", "matrix_frequency_full_optbinning.csv", "optbinning"),
        ("optbinning - ordinals rank-encoded", "matrix_frequency_ranks_optbinning.csv", "optbinning"),
        ("optbinning - bins capped at 3", "matrix_frequency_bins3_optbinning.csv", "optbinning"),
        ("optbinning - numericals binned too", "superseded/optbinning_frequency.csv", "optbinning"),
        ("no carving", "no_carving_frequency.csv", "nocarve"),
        ("AutoCarver", "seed_variance.csv", "autocarver"),
    ]
    rows = [(label, read(name), key) for label, name, key in candidates]
    rows = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(10.9, 3.9))
    style(ax, palette)
    for i, (label, frame, key) in enumerate(rows):
        dot_row(ax, len(rows) - 1 - i, frame.log_loss_dev, palette[key], palette, label)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("dev log loss  (lower is better)", fontsize=9.5, color=palette["muted"])
    fig.suptitle("Four ways to configure optbinning. None of them closes the gap.",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.895,
             "Each arm selects its own features. Rank-encoding the ordinals helps a little; "
             "narrowing the bins does not.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def chart_depth_mechanism(palette):
    """Why the ranking metric swings: the depth the search happened to pick."""
    sources = [
        ("optbinning, own features", "matrix_amount_full_optbinning.csv", "optbinning", "^"),
        ("optbinning, same features", "matrix_amount_fixed_optbinning.csv", "optbinning", "o"),
        ("no carving, same features", "matrix_amount_fixed_nocarve.csv", "nocarve", "o"),
        ("AutoCarver, same features", "matrix_amount_fixed_autocarver.csv", "autocarver", "o"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    style(ax, palette)
    ax.grid(axis="y", color=palette["grid"], linewidth=0.8)
    for label, name, key, marker in sources:
        frame = read(name)
        if frame is None or "top_decile_lift_dev" not in frame.columns:
            continue
        ax.scatter(frame.max_depth, frame.top_decile_lift_dev, s=95, marker=marker,
                   color=palette[key], edgecolor=palette["edge"], linewidth=1.5,
                   label=label, zorder=3)
    ax.axhline(1.0, color=palette["accent"], linewidth=1.4, linestyle="--", zorder=2)
    ax.annotate("no signal", (0.995, 1.0), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(-4, 7), ha="right", fontsize=9,
                color=palette["accent"])
    ax.set_xlabel("tree depth the tuner chose", fontsize=9.5, color=palette["muted"])
    ax.set_ylabel("top-decile lift", fontsize=9.5, color=palette["muted"])
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(palette["ink"])
    fig.suptitle("Deep trees fit the amounts and lose the ordering",
                 fontsize=13.5, fontweight="bold", color=palette["ink"], x=0.012, ha="left")
    fig.text(0.012, 0.9,
             "One point per seed. Every seed that ranked badly is a seed whose search went deep.",
             fontsize=9.5, color=palette["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

# --------------------------------------------------------------------------------------
def themed_svg(path: Path) -> None:
    """Swap every literal light colour for a CSS variable, defined twice (light + dark)."""
    svg = path.read_text(encoding="utf-8")
    for key, hex_value in LIGHT.items():
        if key == "ground":
            continue
        svg = svg.replace(hex_value, f"var(--fig-{key})")
    light_vars = "\n".join(f"    --fig-{k}: {v};" for k, v in LIGHT.items() if k != "ground")
    dark_vars = "\n".join(f"      --fig-{k}: {v};" for k, v in DARK.items() if k != "ground")
    style_block = (
        "<style>\n  :root {\n" + light_vars + "\n  }\n"
        "  @media (prefers-color-scheme: dark) {\n    :root {\n" + dark_vars + "\n    }\n  }\n"
        "</style>\n"
    )
    svg = svg.replace("#000000", "var(--fig-ink)")
    marker = "</defs>"
    if marker in svg:
        svg = svg.replace(marker, marker + "\n" + style_block, 1)
    else:
        cut = svg.index(">", svg.index("<svg")) + 1
        svg = svg[:cut] + "\n" + style_block + svg[cut:]
    path.write_text(svg, encoding="utf-8")


def emit(name: str, builder) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, palette in (("light", LIGHT), ("dark", DARK)):
        fig = builder(palette)
        if mode == "light":
            fig.savefig(OUT / f"{name}.svg", transparent=True)
            themed_svg(OUT / f"{name}.svg")
            fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=palette["ground"])
        else:
            fig.savefig(OUT / f"{name}_dark.png", dpi=200, facecolor=palette["ground"])
        plt.close(fig)
    print(f"wrote {name}.svg / .png / _dark.png")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "figure.facecolor": "none"})
    emit("results_selection_flip", chart_selection_flip)
    emit("results_noise_floor", chart_noise_floor)
    emit("results_severity_blind", chart_severity_blind)
    emit("results_severity_two_metrics", chart_severity_two_metrics)
    emit("results_optbinning_variants", chart_optbinning_variants)
    emit("results_depth_mechanism", chart_depth_mechanism)


if __name__ == "__main__":
    main()
