"""How much does the seeded Optuna search itself move the *severity* dev metric?

The frequency counterpart is `tools/seed_variance.py`, and this is the same harness
pointed at `src/amount_model_2026.ipynb`: carve and select once, then run the identical
400-trial search under several TPE seeds. Seed 42 is the control -- it must reproduce the
notebook's dev RMSE 6469.6972025111745 and dev CHARGE RMSE 6482.842161889131, or this
harness is measuring something else.

The question it answers: §3.4 of the article reports the 2026 severity arm as 2.18 %
better than 2025, and the end-to-end CHARGE arm as 2.37 % better. Those margins were
never checked against the severity search's own seed spread -- only the frequency
search's was measured. If the margins sit inside that spread, they are not results.
"""

import contextlib
import io as _io
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# repository root, so the script runs from anywhere; CAA_REPO overrides it
REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "amount_model_2026.ipynb")
OUT = os.path.join(REPO, "data", "ab_arms", "seed_variance_amount.csv")

SEEDS = [42, 1, 7, 2026]
N_TRIALS = 400

# the notebook's stored outputs, for the control check
SHIPPED_DEV_RMSE = 6469.6972025111745
SHIPPED_DEV_CHARGE = 6482.842161889131

# the 2025 baselines the article's margins are measured against
BASELINE_SEV = 6613.8
BASELINE_CHG = 6639.9
MARGIN_SEV_PCT = 2.18
MARGIN_CHG_PCT = 2.37


def carve_and_select():
    """Exec the notebook's code cells up to (not including) the Optuna search."""
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "import optuna" in src:  # stop before the search; run the selector cell itself
            break
        exec(compile(src, f"<cell {i}>", "exec"), g)
    return g


def charge_rmse(frame, target, xgb, selected_features, rmse):
    """Cell 14's end-to-end CHARGE score: expected count x predicted amount."""
    true_charge = (
        frame["CHARGE"] if "CHARGE" in frame.columns else target.loc[frame.index, "CHARGE"]
    )
    pred_charge = frame["pred_sum"] * xgb.predict(frame[selected_features])
    return rmse(true_charge, pred_charge)


def main():
    g = carve_and_select()
    best_features = g["best_features"]
    print(
        f"carved + selected: {len(best_features)} features on {g['x_train'].shape[0]} rows",
        flush=True,
    )

    import optuna
    from sklearn.metrics import root_mean_squared_error
    from objectives import get_best_regression_model, get_regression_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_train, y_dev = g["y_train"], g["y_dev"]
    w_train, w_dev = g["w_train"], g["w_dev"]
    target = g["target"]

    # tweedie requires a non-negative target -- copied from code cell 11
    y_transform = lambda u: u.where(u >= 0, 0)

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        objective = get_regression_objective(
            x_train[best_features],
            y_transform(y_train),
            x_dev[best_features],
            y_transform(y_dev),
            objective="reg:tweedie",
            w_train=w_train,
            w_dev=w_dev,
        )
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=N_TRIALS)

        # reuse the notebook's own reporting path so the numbers are comparable
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            selected_features, xgb = get_best_regression_model(
                study.best_params,
                x_train[best_features],
                y_transform(y_train),
                x_dev[best_features],
                y_transform(y_dev),
                w_train=w_train,
                w_dev=w_dev,
                train_on_full=False,
            )
        text = buf.getvalue()
        train = float(re.search(r"RMSE Train:\s*([0-9.]+)", text).group(1))
        dev = float(re.search(r"RMSE Dev:\s*([0-9.]+)", text).group(1))

        chg_train = charge_rmse(
            x_train, target, xgb, selected_features, root_mean_squared_error
        )
        chg_dev = charge_rmse(
            x_dev, target, xgb, selected_features, root_mean_squared_error
        )

        rows.append(
            {
                "seed": seed,
                "rmse_train": train,
                "rmse_dev": dev,
                "charge_rmse_train": chg_train,
                "charge_rmse_dev": chg_dev,
                "max_depth": study.best_params.get("max_depth"),
                "n_estimators": study.best_params.get("n_estimators"),
                "minutes": round((time.perf_counter() - t0) / 60, 1),
            }
        )

        note = ""
        if seed == 42:
            ok = (
                abs(dev - SHIPPED_DEV_RMSE) < 1e-6
                and abs(chg_dev - SHIPPED_DEV_CHARGE) < 1e-6
            )
            note = (
                "  <-- control, matches notebook to 1e-6"
                if ok
                else (
                    f"  <-- CONTROL MISMATCH (notebook dev {SHIPPED_DEV_RMSE} "
                    f"charge {SHIPPED_DEV_CHARGE})"
                )
            )
        print(
            f"seed {seed:5d}: dev RMSE {dev:.10f}  dev CHARGE {chg_dev:.10f}  "
            f"train RMSE {train:.10f}  train CHARGE {chg_train:.10f}  "
            f"depth {rows[-1]['max_depth']}  {rows[-1]['minutes']} min{note}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\n" + df.to_string(index=False))

    for col, baseline, margin, label in (
        ("rmse_dev", BASELINE_SEV, MARGIN_SEV_PCT, "severity dev RMSE"),
        ("charge_rmse_dev", BASELINE_CHG, MARGIN_CHG_PCT, "CHARGE dev RMSE"),
    ):
        lo, hi = df[col].min(), df[col].max()
        spread = hi - lo
        pct = 100 * spread / baseline
        print(
            f"\n{label}: min {lo:.6f} | max {hi:.6f} | spread {spread:.6f} "
            f"| std {df[col].std():.6f}"
        )
        print(
            f"  spread as % of the 2025 baseline ({baseline}): {pct:.4f} %"
            f"  -- the article's margin is {margin} %"
        )
        if pct > 0:
            print(f"  margin / spread = {margin / pct:.2f}x")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
