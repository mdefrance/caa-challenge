"""Write Phase 1A's measured severity seed spread into ARTICLE.md (plan step 2.1).

Reads `data/ab_arms/seed_variance_amount.csv`, computes the dev-RMSE and dev-`CHARGE`
spreads as percentages of the 2025 baselines, picks template A / B / C by the plan's
decision rule, and performs the three edits: the seed table, the §3.4 claim, and the
Setup paragraph. Refuses to run if seed 42 does not reproduce the notebook to 1e-6.

Idempotent in the sense that it will fail loudly rather than double-apply: each edit
asserts its anchor text is present exactly once.

    uv run --no-sync python tools/apply_seed_spread_text.py [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
ARTICLE = REPO / "ARTICLE.md"
CSV = REPO / "data" / "ab_arms" / "seed_variance_amount.csv"

BASELINE_SEV, BASELINE_CHG = 6613.8, 6639.9
MARGIN_SEV, MARGIN_CHG = 2.18, 2.37
CONTROL_DEV, CONTROL_CHG = 6469.6972025111745, 6482.842161889131


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV).sort_values("seed", key=lambda s: s.map({42: 0, 1: 1, 7: 2, 2026: 3}))

    control = df.loc[df.seed == 42]
    assert len(control) == 1, "no seed-42 control row"
    assert abs(float(control.rmse_dev.iloc[0]) - CONTROL_DEV) < 1e-6, "control dev RMSE mismatch"
    assert abs(float(control.charge_rmse_dev.iloc[0]) - CONTROL_CHG) < 1e-6, "control CHARGE mismatch"

    sev_spread = float(df.rmse_dev.max() - df.rmse_dev.min())
    chg_spread = float(df.charge_rmse_dev.max() - df.charge_rmse_dev.min())
    sev_pct = 100 * sev_spread / BASELINE_SEV
    chg_pct = 100 * chg_spread / BASELINE_CHG
    r_sev = MARGIN_SEV / sev_pct
    r_chg = MARGIN_CHG / chg_pct
    worst = min(r_sev, r_chg)
    template = "A" if worst >= 3 else ("B" if worst >= 1 else "C")

    # How much of the headline is the reported seed being lucky? Rank it among its own
    # seeds, and report the margin averaged over all four rather than at the best one.
    mean_sev_pct = 100 * (BASELINE_SEV - float(df.rmse_dev.mean())) / BASELINE_SEV
    mean_chg_pct = 100 * (BASELINE_CHG - float(df.charge_rmse_dev.mean())) / BASELINE_CHG
    rank_sev = int((df.rmse_dev < float(control.rmse_dev.iloc[0])).sum()) + 1
    rank_chg = int((df.charge_rmse_dev < float(control.charge_rmse_dev.iloc[0])).sum()) + 1
    others = df[df.seed != 42]
    worst_other = max(
        100 * (BASELINE_SEV - float(others.rmse_dev.min())) / BASELINE_SEV,
        100 * (BASELINE_CHG - float(others.charge_rmse_dev.min())) / BASELINE_CHG,
    )
    print(
        f"seed 42 ranks {rank_sev}/4 on severity and {rank_chg}/4 on CHARGE (1 = best); "
        f"four-seed mean margins {mean_sev_pct:.2f} % / {mean_chg_pct:.2f} %; "
        f"the other three stay within {worst_other:.2f} % of the 2025 baselines"
    )

    print(f"severity dev RMSE spread {sev_spread:.4f} ({sev_pct:.3f} % of {BASELINE_SEV})")
    print(f"CHARGE   dev RMSE spread {chg_spread:.4f} ({chg_pct:.3f} % of {BASELINE_CHG})")
    print(f"margin/spread: severity {r_sev:.2f}x, CHARGE {r_chg:.2f}x -> template {template}")

    rows = []
    for _, r in df.iterrows():
        label = "42 *(the one reported)*" if r.seed == 42 else str(int(r.seed))
        rows.append(f"| {label} | {r.rmse_dev:.1f} | {r.charge_rmse_dev:.1f} |")
    table = (
        "Same check, severity side — 400 trials each, identical features:\n\n"
        "| Seed | Dev RMSE | Dev `CHARGE` RMSE |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )

    if template == "A":
        claim = (
            f"Dev RMSE moves by **{sev_spread:.1f} across those seeds ({sev_pct:.2f} % of the 2025 "
            f"figure)**, and end-to-end `CHARGE` by **{chg_spread:.1f} ({chg_pct:.2f} %)**. The "
            f"2.18 % and 2.37 % margins are {r_sev:.0f}× and {r_chg:.0f}× those spreads, so they "
            "stand — with the caveat that they are measured against an unseeded 2025 baseline and "
            "a narrower 2026 candidate set, so they belong to the *pipeline as re-run*, not to any "
            "single library change. **The 2026 pipeline wins end to end on the challenge's own "
            "metric, carried by the severity side, and the frequency side is a draw.**"
        )
        setup = (
            f"and the two that remain (severity 2.18 %, `CHARGE` 2.37 %) clear their own severity "
            f"spreads ({sev_pct:.2f} % and {chg_pct:.2f} %) by {r_sev:.0f}× and {r_chg:.0f}×"
        )
    elif template == "B":
        # At the bottom of the B band the margin is the *same size* as the noise, and
        # calling it "larger than search noise" would be indefensible even though the
        # plan's band technically allows it. Say what the ratio actually shows.
        if max(r_sev, r_chg) < 1.5:
            verdict = (
                f"are {r_sev:.2f}× and {r_chg:.2f}× those spreads — the margin and the noise are "
                "the same size. And it is worse than that for the claim: **seed 42, the one the "
                "notebook reports, is the best of the four on both metrics**, while the other "
                f"three all land within {worst_other:.1f} % of the 2025 baseline. Averaged over "
                f"the four seeds the margin is **{mean_sev_pct:.2f} %** on severity and "
                f"**{mean_chg_pct:.2f} %** on `CHARGE`, not 2.18 % and 2.37 %. **The frequency "
                "side is a draw, and on this evidence so is the severity side: the run we "
                "reported is the luckiest of four.**"
            )
            setup_tail = (
                f"and the two that looked like results (severity 2.18 %, `CHARGE` 2.37 %) are the "
                f"same size as their own seed spreads ({sev_pct:.2f} % and {chg_pct:.2f} %) and "
                f"rest on the best of four seeds — averaged over the four the margins are "
                f"{mean_sev_pct:.2f} % and {mean_chg_pct:.2f} %"
            )
        else:
            verdict = (
                f"are {r_sev:.1f}× and {r_chg:.1f}× those spreads: larger than search noise, but "
                "not by enough to call decisive on four seeds. We report them as *likely* "
                "improvements and stop short of claiming more. **The frequency side is a draw; the "
                "severity side probably, but not certainly, moved.**"
            )
            setup_tail = (
                f"and the two that remain (severity 2.18 %, `CHARGE` 2.37 %) exceed their own seed "
                f"spreads ({sev_pct:.2f} % and {chg_pct:.2f} %) by only {r_sev:.1f}× and "
                f"{r_chg:.1f}×, which is why §3.4 calls them likely rather than settled"
            )
        claim = (
            f"Dev RMSE moves by **{sev_spread:.1f} across those seeds ({sev_pct:.2f} % of the 2025 "
            f"figure)**, and end-to-end `CHARGE` by **{chg_spread:.1f} ({chg_pct:.2f} %)**. The "
            f"2.18 % and 2.37 % margins " + verdict
        )
        setup = setup_tail
    else:
        claim = (
            f"Dev RMSE moves by **{sev_spread:.1f} across those seeds ({sev_pct:.2f} % of the 2025 "
            f"figure)**, and end-to-end `CHARGE` by **{chg_spread:.1f} ({chg_pct:.2f} %)**. The "
            "2.18 % and 2.37 % margins fall *inside* that spread. Same verdict as frequency: a "
            "draw. **The honest scoreboard for a year of releases is therefore speed, not "
            "accuracy.**"
        )
        setup = (
            f"and the two that looked like results (severity 2.18 %, `CHARGE` 2.37 %) sit inside "
            f"their own seed spreads ({sev_pct:.2f} % and {chg_pct:.2f} %), so §3.4 calls them a "
            "draw too"
        )

    s = io.open(ARTICLE, encoding="utf-8").read()

    # --- 0. the end-to-end table's verdict column ------------------------------------
    # Under B and C the "2.18 % better" verdicts cannot stand unqualified: the table is
    # the most-read thing in the section, and a reader who stops there would take away
    # exactly the claim the seed sweep just undercut.
    old_head = "| Metric | 2025 (7.0.5) | 2026 (current) | |\n|---|---|---|---|"
    assert s.count(old_head) == 1, "end-to-end table header not found exactly once"
    s = s.replace(old_head, "| Metric | 2025 (7.0.5) | 2026 (current) | verdict |\n|---|---|---|---|")

    if template != "A":
        if template == "C":
            qualifier = "— **inside** the seed spread"
        elif max(r_sev, r_chg) < 1.5:
            qualifier = "— **the same size as** the seed spread"
        else:
            qualifier = "— but only {r:.1f}× the seed spread"
        for old, r in (
            ("| Severity — dev RMSE | 6613.8 | **6469.7** | ⬆ **2.18 % better** |", r_sev),
            (
                "| **`CHARGE` — dev RMSE** *(the challenge metric)* | 6639.9 | **6482.8** | ⬆ **2.37 % better** |",
                r_chg,
            ),
        ):
            assert s.count(old) == 1, f"table row not found exactly once: {old[:40]}"
            s = s.replace(old, old[:-1].rstrip() + " " + qualifier.format(r=r) + " |")

    if template != "A":
        old_lead = "**That top row is not a result, and it took a deliberate experiment to find out.**"
        assert s.count(old_lead) == 1, "top-row lead not found exactly once"
        s = s.replace(
            old_lead,
            "**That top row is not a result, and it took a deliberate experiment to find "
            "out.** Then the same experiment came for the other two.",
        )

    # --- 1. the seed table, after the frequency seed table's interpretation ----------
    anchor = (
        "single run, measure your seed spread before you interpret anything smaller than it.\n"
    )
    assert s.count(anchor) == 1, "seed-lesson anchor not found exactly once"
    s = s.replace(anchor, anchor + "\n" + table)

    # --- 2. the §3.4 claim ----------------------------------------------------------
    old_claim = (
        "The severity and `CHARGE` margins — 2.18 % and 2.37 % — sit two orders of magnitude above\n"
        "that noise floor, so those stand. **The 2026 pipeline wins end to end on the challenge's\n"
        "own metric, carried by the severity side, and the frequency side is a draw.**"
    )
    assert s.count(old_claim) == 1, "§3.4 claim anchor not found exactly once"
    s = s.replace(old_claim, claim)

    # --- 3. the Setup paragraph -----------------------------------------------------
    old_setup = (
        "Reproducible is not the same as representative. We measured the frequency search's\n"
        "own seed spread — same features, same 300 trials, four seeds — at **0.0101 dev log\n"
        "loss**, which is why §3.4 calls the frequency arms a draw rather than reading their\n"
        "0.00055 difference as a result. **No accuracy claim in this article rests on a\n"
        "margin smaller than that**, and the two that remain (severity 2.18 %, `CHARGE`\n"
        "2.37 %) clear it by two orders of magnitude."
    )
    assert s.count(old_setup) == 1, "Setup paragraph anchor not found exactly once"
    new_setup = (
        "Reproducible is not the same as representative. We measured both searches' own seed "
        "spread over four seeds — **0.0101 dev log loss** on frequency\n"
        "(`data/ab_arms/seed_variance.csv`) and the severity figures in §3.4\n"
        "(`data/ab_arms/seed_variance_amount.csv`) — which is why §3.4 calls the frequency arms a "
        "draw rather than reading their 0.00055 difference as a result. **Every accuracy figure "
        "in this article is stated against its own measured seed spread**, " + setup + "."
    )
    s = s.replace(old_setup, new_setup)

    if args.dry_run:
        print("\n--- dry run, nothing written ---\n")
        print(table)
        print(claim)
        print()
        print(new_setup)
        return

    io.open(ARTICLE, "w", encoding="utf-8").write(s)
    print(f"\napplied template {template} to {ARTICLE}")


if __name__ == "__main__":
    main()
