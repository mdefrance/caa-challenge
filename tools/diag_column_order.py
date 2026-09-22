"""Does the ORDER of identical columns change the dev metric?

The fixed-feature arms model exactly the 100 features AutoCarver's own pipeline selected,
so the AutoCarver fixed arm ought to reproduce the own-selection arm. It does not: 0.9216
against 0.9199 on the four-seed mean, and 0.9247 against 0.9199 on seed 42.

The suspect is column order. The search tunes `colsample_bytree` and `colsample_bylevel`,
so XGBoost draws a subset of columns per tree and per level, and that draw depends on the
order the columns sit in the frame. The own-selection arm feeds them in the selector's
ranking order; the fixed arm feeds them grouped by type. Same values, different order.

This fits ONE fixed hyperparameter set (no Optuna) to the same carved frame under several
column permutations. If order is irrelevant, every permutation returns the same log loss.

    uv run --no-sync python tools/diag_column_order.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO / "tools"))
OUT = REPO / "data" / "ab_arms" / "column_order_sensitivity.json"

# one representative configuration, in the family the tuner actually explores
PARAMS = dict(
    objective="multi:softprob",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=1,
    subsample=0.8,
    colsample_bytree=0.5,   # < 1, so the column draw is in play
    colsample_bylevel=0.7,
    random_state=42,
)


def main() -> int:
    import ablation_matrix as M

    g = M.notebook_prefix(M.SPEC["frequency"]["notebook"], M.SPEC["frequency"]["stop"])
    import numpy as np
    from sklearn.metrics import log_loss
    from xgboost import XGBClassifier

    shared = json.loads(
        (REPO / "data" / "ab_arms" / "fixed_features_frequency.json").read_text(encoding="utf-8")
    )["selected"]
    cats = [c for c in g["categorical_columns"] if c in shared]
    nums = [c for c in g["numerical_columns"] if c in shared]
    ordinal_columns = g["ordinal_columns"]
    keys = [c for c in list(ordinal_columns) if c in shared]
    ords = ({k: ordinal_columns[k] for k in keys}
            if isinstance(ordinal_columns, dict) else keys)

    train, dev, _, _ = M.build_autocarver(
        g["x_train"], g["x_dev"], g["y_ord_train"], g["y_ord_dev"],
        cats, ords, nums, "frequency",
    )
    columns = [c for c in train.columns if c in dev.columns]
    print(f"carved frame: {len(columns)} columns", flush=True)

    y_train, y_dev = g["y_ord_train"], g["y_ord_dev"]
    w_train, w_dev = g["w_train"], g["w_dev"]

    def score(order, label):
        model = XGBClassifier(num_class=3, device="cuda", n_jobs=-1, **PARAMS)
        model.fit(train[order], y_train, sample_weight=w_train)
        proba = model.predict_proba(dev[order])
        value = float(log_loss(y_dev, proba, sample_weight=w_dev))
        print(f"  {label:34} {value:.10f}", flush=True)
        return value

    rng = np.random.default_rng(0)
    orders = {
        "as built (type-grouped)": list(columns),
        "reversed": list(reversed(columns)),
        "alphabetical": sorted(columns),
        "shuffled A": list(rng.permutation(columns)),
        "shuffled B": list(rng.permutation(columns)),
    }
    print("weighted dev log loss, identical features and hyperparameters:")
    results = {label: score(order, label) for label, order in orders.items()}

    values = list(results.values())
    spread = max(values) - min(values)
    print(f"\nspread across orderings: {spread:.6f}")
    print("AutoCarver four-seed spread, for scale: 0.010086")
    print(f"=> column order alone moves the metric by {spread / 0.010086:.0%} of the seed spread")

    OUT.write_text(json.dumps({"params": PARAMS, "log_loss_by_order": results,
                               "spread": spread}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
