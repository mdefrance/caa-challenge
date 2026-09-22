"""Re-derive every arithmetic claim in ARTICLE.md from its inputs, and print PASS/FAIL.

The article quotes ratios, percentages and totals that a reader will spot-check with a
calculator. This script is that calculator, run in advance: each line states the inputs,
the claimed figure, and whether the two agree to the precision the article prints them at.

It does not re-run any model. The inputs are the measured figures the notebooks and
scripts in this repository produced -- notebook outputs for the era comparisons,
`data/ab_arms/*.csv` for the carver arms and the seed sweeps. What it checks is that the
*derived* numbers in the prose follow from them.

    uv run --no-sync python tools/check_article_numbers.py
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
SEED_AMOUNT_CSV = REPO / "data" / "ab_arms" / "seed_variance_amount.csv"
SEED_FREQ_CSV = REPO / "data" / "ab_arms" / "seed_variance.csv"

# --- measured inputs, each with the file it came from -----------------------------
# Carving wall-clock, as printed by the notebooks themselves. The article rounds these
# to whole seconds, so the checks below work from the unrounded figures and compare
# against the rounded ones -- summing the rounded values instead would be off by a
# second on the frequency total (2842 + 383 = 3225, but 2842.3 + 383.4 = 3225.7 -> 3226).
#   src/frequency_model_2025_timed.ipynb cell 18: "[2025] qualitative carving: 2842.3s"
#   src/frequency_model_2025_timed.ipynb cell 24: "[2025] quantitative carving: 383.4s"
#   src/amount_model_2025_timed.ipynb    cell 15: "[2025] qualitative carving: 4724.7s"
#   src/frequency_model_2026.ipynb       cell 13: "[2026] qualitative carving: 120.0s"
#   src/amount_model_2026.ipynb          cell 10: "[2026] qualitative carving: 139.8s"
T_FREQ_QUAL_2025, T_FREQ_QUAL_2026 = 2842.3, 120.0
T_FREQ_QUANT_2025 = 383.4
T_FREQ_TOTAL_2025 = T_FREQ_QUAL_2025 + T_FREQ_QUANT_2025
T_FREQ_TOTAL_2026 = 120.0
T_SEV_2025, T_SEV_2026 = 4724.7, 139.8

# carver arms, data/ab_arms/summary.csv (article §3.2)
ONEVSREST_S, ORDINAL_S = 482, 127
ONEVSREST_COLS, ORDINAL_COLS = 823, 397
ONEVSREST_BUCKETS, ORDINAL_BUCKETS = 1707, 834

# end-to-end metrics, notebook outputs (article §3.4)
SEV_2025, SEV_2026 = 6613.8, 6469.7
CHG_2025, CHG_2026 = 6639.9, 6482.8
FREQ_2025, FREQ_2026 = 0.91936, 0.9199100734725815

results: list[tuple[bool, str]] = []


def check(label: str, got: float, claimed: float, tol: float) -> None:
    ok = abs(got - claimed) <= tol
    results.append((ok, f"{'PASS' if ok else 'FAIL'}  {label}: got {got:.6g}, article says {claimed:.6g}"))


# --- §3.1 speed-ups ---------------------------------------------------------------
check("2842/120 frequency qualitative speed-up",
      T_FREQ_QUAL_2025 / T_FREQ_QUAL_2026, 23.7, 0.05)
check("3226/120 frequency total speed-up",
      T_FREQ_TOTAL_2025 / T_FREQ_TOTAL_2026, 26.9, 0.05)
check("4725/140 severity speed-up", T_SEV_2025 / T_SEV_2026, 33.8, 0.05)
check("2842.3+383.4 rounds to the table's 3226 s",
      round(T_FREQ_QUAL_2025 + T_FREQ_QUANT_2025), 3226, 0)
check("table's rounded carve times", round(T_FREQ_QUAL_2025), 2842, 0)
check("table's rounded quantitative carve", round(T_FREQ_QUANT_2025), 383, 0)
check("table's rounded severity carve", round(T_SEV_2025), 4725, 0)
check("table's rounded 2026 severity carve", round(T_SEV_2026), 140, 0)
check("3226+4725 = 133 min of 2025 carving",
      (T_FREQ_TOTAL_2025 + T_SEV_2025) / 60, 133, 0.5)
check("120+140 = 4.3 min of 2026 carving",
      (T_FREQ_TOTAL_2026 + T_SEV_2026) / 60, 4.3, 0.05)
check("2842 s = 47.4 min", T_FREQ_QUAL_2025 / 60, 47.4, 0.05)
check("3226 s = 53.8 min", T_FREQ_TOTAL_2025 / 60, 53.8, 0.05)
check("4725 s = 78.7 min", T_SEV_2025 / 60, 78.7, 0.05)

# --- §3.2 carver arms -------------------------------------------------------------
check("482/127 ordinal carve speed-up", ONEVSREST_S / ORDINAL_S, 3.8, 0.05)
check("823/397 fewer columns", ONEVSREST_COLS / ORDINAL_COLS, 2.07, 0.005)
check("1707/834 fewer buckets", ONEVSREST_BUCKETS / ORDINAL_BUCKETS, 2.05, 0.005)

# --- §3.2 matched-364 comparison, re-derived from per_column_tau_c.csv -------------
# The article's per-feature columns are matched: the 364 features every arm carved into
# two or more buckets. Carve seconds and column counts in the same table are over all
# 435 inputs (data/ab_arms/summary.csv), which the table's headers now say.
TAU_CSV = REPO / "data" / "ab_arms" / "per_column_tau_c.csv"
if TAU_CSV.exists():
    import pandas as pd

    tau = pd.read_csv(TAU_CSV)
    tau = tau[tau.arm != "A4 ordinal, Wilson off"]
    present = set.intersection(*(set(g.feature) for _, g in tau.groupby("arm")))
    matched = set(present)
    for _, g in tau.groupby("arm"):
        g = g[g.feature.isin(present)]
        matched &= set(g.groupby("feature").n_buckets.max()[lambda s: s >= 2].index)

    check("§3.2 matched feature count", len(matched), 364, 0)

    m = tau[tau.feature.isin(matched)]
    claimed_bpf = {"A1 one-vs-rest": 4.32, "A2 multiclass": 2.20, "A3 ordinal": 2.13}
    claimed_tau = {"A1 one-vs-rest": 0.00132, "A2 multiclass": 0.00123, "A3 ordinal": 0.00133}
    best = {}
    for arm, g in m.groupby("arm"):
        check(f"§3.2 {arm} buckets/feature",
              g.n_buckets.sum() / len(matched), claimed_bpf[arm], 0.005)
        b = g.assign(t=g.tau_c.abs()).groupby("feature").t.max()
        best[arm] = b
        check(f"§3.2 {arm} best |tau-c| per feature", b.mean(), claimed_tau[arm], 5e-6)

    ordinal = best["A3 ordinal"]
    for other, (ident, win, lose, mean_d) in {
        "A2 multiclass": (210, 111, 43, 0.000107),
        "A1 one-vs-rest": (200, 84, 80, 0.000010),
    }.items():
        diff = (ordinal - best[other]).reindex(sorted(matched))
        check(f"§3.2 vs {other}: identical", int((diff.abs() < 1e-12).sum()), ident, 0)
        check(f"§3.2 vs {other}: ordinal wins", int((diff > 1e-12).sum()), win, 0)
        check(f"§3.2 vs {other}: ordinal loses", int((diff < -1e-12).sum()), lose, 0)
        check(f"§3.2 vs {other}: mean difference", diff.mean(), mean_d, 5e-7)
else:
    results.append((False, f"FAIL  {TAU_CSV} missing"))

# --- §3.2 sign tests --------------------------------------------------------------
try:
    from scipy.stats import binomtest

    check("sign test 111/154 p-value", binomtest(111, 154, 0.5).pvalue, 4e-8, 1e-8)
    check("sign test 84/164 p-value", binomtest(84, 164, 0.5).pvalue, 0.81, 0.01)
except ImportError:  # pragma: no cover
    results.append((False, "FAIL  sign tests: scipy not importable"))

# --- §3.4 era margins -------------------------------------------------------------
check("(6613.8-6469.7)/6613.8 severity margin %",
      100 * (SEV_2025 - SEV_2026) / SEV_2025, 2.18, 0.005)
check("(6639.9-6482.8)/6639.9 CHARGE margin %",
      100 * (CHG_2025 - CHG_2026) / CHG_2025, 2.37, 0.005)

# --- §3.4 frequency seed spread ---------------------------------------------------
if SEED_FREQ_CSV.exists():
    import pandas as pd

    freq = pd.read_csv(SEED_FREQ_CSV)
    spread = freq.log_loss_dev.max() - freq.log_loss_dev.min()
    check("frequency dev log-loss seed spread", spread, 0.0101, 0.0001)
    check("0.9246-0.9145 = 0.0101 spread",
          freq.log_loss_dev.max() - freq.log_loss_dev.min(), 0.0101, 0.0001)
    gap = abs(FREQ_2026 - FREQ_2025)
    check("2026-2025 frequency gap", gap, 0.00055, 0.00001)
    check("0.0101/0.00055 = 18x", spread / gap, 18, 0.5)
else:
    results.append((False, f"FAIL  {SEED_FREQ_CSV} missing"))

# --- 1A: severity seed spread -----------------------------------------------------
if SEED_AMOUNT_CSV.exists():
    import pandas as pd

    amt = pd.read_csv(SEED_AMOUNT_CSV)
    sev_spread = amt.rmse_dev.max() - amt.rmse_dev.min()
    chg_spread = amt.charge_rmse_dev.max() - amt.charge_rmse_dev.min()
    sev_pct = 100 * sev_spread / SEV_2025
    chg_pct = 100 * chg_spread / CHG_2025

    control = amt.loc[amt.seed == 42]
    if len(control) == 1:
        check("1A control: seed 42 dev RMSE reproduces the notebook",
              float(control.rmse_dev.iloc[0]), 6469.6972025111745, 1e-6)
        check("1A control: seed 42 dev CHARGE RMSE reproduces the notebook",
              float(control.charge_rmse_dev.iloc[0]), 6482.842161889131, 1e-6)
    else:
        results.append((False, "FAIL  1A control: no seed-42 row"))

    print(
        f"\n1A measured: severity dev RMSE spread {sev_spread:.4f} ({sev_pct:.4f} % of "
        f"{SEV_2025}), margin/spread = {2.18 / sev_pct:.2f}x"
    )
    print(
        f"1A measured: CHARGE dev RMSE spread {chg_spread:.4f} ({chg_pct:.4f} % of "
        f"{CHG_2025}), margin/spread = {2.37 / chg_pct:.2f}x\n"
    )

    # Every figure §3.4 now quotes from this sweep, checked at the precision it prints.
    check("§3.4 severity spread (absolute)", sev_spread, 143.2, 0.05)
    check("§3.4 severity spread as % of 6613.8", sev_pct, 2.16, 0.005)
    check("§3.4 CHARGE spread (absolute)", chg_spread, 155.2, 0.05)
    check("§3.4 CHARGE spread as % of 6639.9", chg_pct, 2.34, 0.005)
    check("§3.4 severity margin/spread ratio", 2.18 / sev_pct, 1.01, 0.005)
    check("§3.4 CHARGE margin/spread ratio", 2.37 / chg_pct, 1.01, 0.005)

    # the claim that seed 42 is the best of the four, and the four-seed mean margins
    check("§3.4 seed 42 is best of four on severity",
          int((amt.rmse_dev < float(control.rmse_dev.iloc[0])).sum()), 0, 0)
    check("§3.4 seed 42 is best of four on CHARGE",
          int((amt.charge_rmse_dev < float(control.charge_rmse_dev.iloc[0])).sum()), 0, 0)
    check("§3.4 four-seed mean severity margin %",
          100 * (SEV_2025 - amt.rmse_dev.mean()) / SEV_2025, 0.65, 0.005)
    check("§3.4 four-seed mean CHARGE margin %",
          100 * (CHG_2025 - amt.charge_rmse_dev.mean()) / CHG_2025, 0.72, 0.005)

    # "the other three land within 0.4 % of the 2025 baseline"
    others = amt[amt.seed != 42]
    worst_other = max(
        100 * (SEV_2025 - float(others.rmse_dev.min())) / SEV_2025,
        100 * (CHG_2025 - float(others.charge_rmse_dev.min())) / CHG_2025,
    )
    ok = worst_other <= 0.4
    results.append((
        ok,
        f"{'PASS' if ok else 'FAIL'}  §3.4 other three seeds within 0.4 % of the 2025 "
        f"baselines: worst is {worst_other:.3f} %",
    ))
else:
    results.append((False, f"FAIL  {SEED_AMOUNT_CSV} missing -- run tools/seed_variance_amount.py"))

# --- the 2025 CHARGE baseline, from tools/replay_2025_charge.py --------------------
REPLAY_JSON = REPO / "data" / "ab_arms" / "replay_2025_charge.json"
if REPLAY_JSON.exists():
    replay = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    # the replay's own control: it must reproduce the 2025 notebook's severity RMSE
    results.append((
        bool(replay["control_passed"]),
        f"{'PASS' if replay['control_passed'] else 'FAIL'}  replay control: severity dev "
        f"RMSE {replay['severity_rmse_dev']:.6f} vs notebook "
        f"{replay['control_recorded_severity_rmse_dev']:.6f} "
        f"(delta {replay['control_delta']:.1e})",
    ))
    # and it must be the source of the article's 2025 CHARGE figure
    check("2025 CHARGE baseline comes from the replay",
          replay["charge_rmse_dev"], CHG_2025, 0.05)
    check("2025 severity baseline comes from the replay",
          replay["severity_rmse_dev"], SEV_2025, 0.05)
else:
    results.append((
        False,
        f"FAIL  {REPLAY_JSON} missing -- run tools/replay_2025_charge.py against the "
        "2025 artifacts (see its docstring)",
    ))

# --- section 3.4: the three-way binning ablation, order-corrected ------------------
CONST_SEV_DEV, CONST_CHG_DEV = 6617.646044495917, 6643.544969825402

def arm(name):
    path = REPO / "data" / "ab_arms" / name
    return pd.read_csv(path, comment="#") if path.exists() else None


def gap_ratio(a, b):
    """|mean difference| in multiples of the two arms' pooled four-seed spread."""
    pooled = ((a.max() - a.min()) + (b.max() - b.min())) / 2
    return abs(float(a.mean()) - float(b.mean())) / pooled


def disjoint(a, b):
    return bool(a.max() < b.min() or b.max() < a.min())


FREQ_OWN = {"ac": SEED_FREQ_CSV.name, "nc": "no_carving_frequency.csv",
            "ob": "matrix_frequency_full_optbinning.csv"}
FREQ_FIX = {"ac": "matrix_frequency_fixed_autocarver.csv",
            "nc": "matrix_frequency_fixed_nocarve.csv",
            "ob": "matrix_frequency_fixed_optbinning.csv"}
AMT_FIX = {"ac": "matrix_amount_fixed_autocarver.csv",
           "nc": "matrix_amount_fixed_nocarve.csv",
           "ob": "matrix_amount_fixed_optbinning.csv"}

own = {k: arm(v) for k, v in FREQ_OWN.items()}
fix = {k: arm(v) for k, v in FREQ_FIX.items()}
amt = {k: arm(v) for k, v in AMT_FIX.items()}

if all(v is not None for v in {**own, **fix, **amt}.values()):
    # frequency, each arm selecting its own 100 features
    check("§3.4 own features: AutoCarver mean", float(own["ac"].log_loss_dev.mean()), 0.9199, 0.0001)
    check("§3.4 own features: no carving mean", float(own["nc"].log_loss_dev.mean()), 0.9386, 0.0001)
    check("§3.4 own features: optbinning mean", float(own["ob"].log_loss_dev.mean()), 0.9660, 0.0001)
    check("§3.4 own features: AutoCarver vs no carving",
          gap_ratio(own["ac"].log_loss_dev, own["nc"].log_loss_dev), 2.45, 0.01)
    check("§3.4 own features: no carving vs optbinning",
          gap_ratio(own["nc"].log_loss_dev, own["ob"].log_loss_dev), 2.56, 0.01)
    check("§3.4 own features: AutoCarver vs optbinning",
          gap_ratio(own["ac"].log_loss_dev, own["ob"].log_loss_dev), 3.50, 0.01)

    # frequency, all arms on the same 100 features, fed in the selector's own order
    check("§3.4 same features: optbinning mean", float(fix["ob"].log_loss_dev.mean()), 0.9136, 0.0001)
    check("§3.4 same features: AutoCarver mean", float(fix["ac"].log_loss_dev.mean()), 0.9199, 0.0001)
    check("§3.4 same features: no carving mean", float(fix["nc"].log_loss_dev.mean()), 0.9334, 0.0001)
    check("§3.4 same features: AutoCarver vs no carving",
          gap_ratio(fix["ac"].log_loss_dev, fix["nc"].log_loss_dev), 1.59, 0.01)
    check("§3.4 same features: optbinning vs no carving",
          gap_ratio(fix["ob"].log_loss_dev, fix["nc"].log_loss_dev), 1.59, 0.01)
    check("§3.4 same features: the two binners draw",
          gap_ratio(fix["ac"].log_loss_dev, fix["ob"].log_loss_dev), 0.45, 0.01)

    # With the columns fed in the selector's own order, the AutoCarver fixed arm IS the
    # own-selection arm, so matrix_frequency_fixed_autocarver.csv carries copied rows
    # rather than a second run. This assertion therefore only proves the copy is faithful
    # -- it is NOT evidence that the two arms agree. That evidence is the two seeds re-run
    # by hand and recorded in that file's header (seed 42: 0.9199100734725816 own-selection
    # against 0.9199100734725815 fixed-set; seed 1: equal to all 16 digits).
    same = [f"{a:.13f}" == f"{b:.13f}" for a, b in
            zip(sorted(fix["ac"].log_loss_dev), sorted(own["ac"].log_loss_dev))]
    results.append((all(same),
                    f"{'PASS' if all(same) else 'FAIL'}  §3.4 bookkeeping: the copied "
                    f"fixed-set AutoCarver rows match their source on {sum(same)}/4 seeds "
                    f"(a copy check, not an independent reproduction)"))

    # both binners clear the unbinned arm with no overlap, on every seed
    for key, label in (("ac", "AutoCarver"), ("ob", "optbinning")):
        ok = disjoint(fix[key].log_loss_dev, fix["nc"].log_loss_dev)
        results.append((ok, f"{'PASS' if ok else 'FAIL'}  §3.4 {label} and no carving do not "
                            f"overlap on the shared feature set"))

    # severity: RMSE separates nothing, ranking separates the binners from no carving
    check("§3.4 severity RMSE: AutoCarver", float(amt["ac"].rmse_dev.mean()), 6570.7, 0.05)
    check("§3.4 severity RMSE: no carving", float(amt["nc"].rmse_dev.mean()), 6580.7, 0.05)
    check("§3.4 severity RMSE: optbinning", float(amt["ob"].rmse_dev.mean()), 6612.8, 0.05)
    check("§3.4 severity lift: AutoCarver", float(amt["ac"].top_decile_lift_dev.mean()), 3.35, 0.005)
    check("§3.4 severity lift: no carving", float(amt["nc"].top_decile_lift_dev.mean()), 2.43, 0.005)
    check("§3.4 severity lift: optbinning", float(amt["ob"].top_decile_lift_dev.mean()), 3.94, 0.005)
    check("§3.4 severity lift: optbinning vs no carving",
          gap_ratio(amt["ob"].top_decile_lift_dev, amt["nc"].top_decile_lift_dev), 2.41, 0.01)

    # the overfitting claim: every binned seed near 1.0, the unbinned arm's seed 7 at 0.31
    ratios = {k: (v.rmse_train / v.rmse_dev).round(2).tolist() for k, v in amt.items()}
    worst = min(min(v) for v in ratios.values())
    check("§3.4 worst train/dev RMSE ratio (the memorising seed)", worst, 0.31, 0.005)
    binned_ok = all(abs(r - 1.01) <= 0.06 for k in ("ac", "ob") for r in ratios[k])
    results.append((binned_ok, f"{'PASS' if binned_ok else 'FAIL'}  §3.4 every binned seed's "
                               f"train/dev RMSE ratio is near 1.0"))

    # optbinning's other configurations
    for label, name, claimed in (("rank-encoded ordinals", "matrix_frequency_ranks_optbinning.csv", 0.9571),
                                 ("bins capped at 3", "matrix_frequency_bins3_optbinning.csv", 0.9707)):
        frame = arm(name)
        if frame is not None:
            check(f"§3.4 optbinning {label}", float(frame.log_loss_dev.mean()), claimed, 0.0001)

    # the constant-predictor reference
    check("§3.4 constant-predictor dev RMSE", CONST_SEV_DEV, 6617.6, 0.05)
else:
    results.append((False, "FAIL  ablation arms missing -- run tools/ablation_matrix.py"))

# the carve-repeat spread quoted in §3.1, now measured rather than asserted
check("§3.1 carve-repeat spread %", 100 * (129.4 - 120.0) / 120.0, 7.8, 0.05)

for ok, line in results:
    print(line)
failed = sum(1 for ok, _ in results if not ok)
print(f"\n{len(results) - failed}/{len(results)} checks passed")
raise SystemExit(1 if failed else 0)
