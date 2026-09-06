"""How much does the seeded Optuna search itself move the frequency dev metric?

Carves and selects once — the shipped ClassificationSelector 100 — then runs the identical
300-trial search under several TPE seeds. Seed 42 is the control: it must reproduce the
notebook's 0.9199100734725815 exactly, or this harness is measuring something else.

The question it answers is not really about Run 19. Section 3.5 reports that the 2026
frequency arm is worse than 2025 by 0.00055 *because the run is seeded*. If the spread
across seeds is an order of magnitude larger than that, the honest word is
"indistinguishable".
"""

import contextlib
import io as _io
import json
import os
import re
import sys
from pathlib import Path
import time

import pandas as pd

# repository root, so the script runs from anywhere; CAA_REPO overrides it
REPO = os.environ.get("CAA_REPO") or str(Path(__file__).resolve().parents[1])
NB = os.path.join(REPO, "src", "frequency_model_2026.ipynb")
OUT = os.path.join(REPO, "data", "ab_arms", "seed_variance.csv")

SEEDS = [42, 1, 7, 2026]
N_TRIALS = 300
SHIPPED_DEV = 0.9199100734725815


def carve_and_select():
    nb = json.load(open(NB, encoding="utf-8"))
    os.chdir(os.path.join(REPO, "src"))
    sys.path.insert(0, os.getcwd())
    g = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if "optuna" in src:  # stop before the search; run the selector cell itself
            break
        exec(compile(src, f"<cell {i}>", "exec"), g)
    return g


def main():
    g = carve_and_select()
    best_features = g["best_features"]
    print(
        f"carved + selected: {len(best_features)} features on {g['x_train'].shape[0]} rows",
        flush=True,
    )

    import optuna
    from utils.objectives import get_best_multiclass_model, get_multiclass_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_train, y_dev = g["y_ord_train"], g["y_ord_dev"]
    w_train, w_dev = g["w_train"], g["w_dev"]

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        objective = get_multiclass_objective(
            x_train[best_features],
            y_train,
            x_dev[best_features],
            y_dev,
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
            get_best_multiclass_model(
                study.best_params,
                x_train[best_features],
                y_train,
                x_dev[best_features],
                y_dev,
                w_train=w_train,
                w_dev=w_dev,
                train_on_full=False,
            )
        text = buf.getvalue()
        train = float(re.search(r"Log Loss Train:\s*([0-9.]+)", text).group(1))
        dev = float(re.search(r"Log Loss Dev:\s*([0-9.]+)", text).group(1))
        rows.append(
            {
                "seed": seed,
                "log_loss_train": train,
                "log_loss_dev": dev,
                "max_depth": study.best_params.get("max_depth"),
                "n_estimators": study.best_params.get("n_estimators"),
                "minutes": round((time.perf_counter() - t0) / 60, 1),
            }
        )
        note = ""
        if seed == 42:
            note = (
                "  <-- control, matches notebook"
                if abs(dev - SHIPPED_DEV) < 1e-12
                else f"  <-- CONTROL MISMATCH (notebook {SHIPPED_DEV})"
            )
        print(
            f"seed {seed:5d}: dev {dev:.10f}  train {train:.10f}  "
            f"depth {rows[-1]['max_depth']}  {rows[-1]['minutes']} min{note}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    spread = df.log_loss_dev.max() - df.log_loss_dev.min()
    print("\n" + df.to_string(index=False))
    print(
        f"\ndev log loss: min {df.log_loss_dev.min():.6f} | max {df.log_loss_dev.max():.6f} "
        f"| spread {spread:.6f} | std {df.log_loss_dev.std():.6f}"
    )
    print("2025 baseline 0.91936 — gap to the shipped arm is 0.00055")
    print(f"seed spread is {spread / 0.00055:.1f}x that gap")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
