"""Is the severity comparison a draw, or is dev RMSE simply blind?

Every severity arm lands within a few RMSE of predicting the training mean, so "the arms
are equivalent" and "the metric cannot tell them apart" both fit the evidence. This tells
them apart, cheaply: it fits **one** model per arm at a **fixed** configuration -- no
Optuna, no per-seed search -- and scores each on metrics that do have dynamic range on a
heavy-tailed target.

* Spearman correlation between prediction and truth: pure ranking, indifferent to scale.
* Top-decile lift: mean actual amount among the 10 % highest predictions, over the overall
  mean. A model with no signal scores 1.0.
* Tweedie deviance at a fixed power, the loss family the models are actually trained on.
  The tuned power differs per arm, so a tuned one would not be comparable across arms.

The constant predictor is included as the floor: Spearman 0, lift 1.0, by construction.

Because the configuration is fixed rather than tuned, these numbers are **not** comparable
to the tuned RMSEs in the article's tables. They answer one question only: does any metric
separate the arms when RMSE does not?

    uv run --no-sync python tools/severity_metric_dynamic_range.py
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CAA_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO / "tools"))
OUT = REPO / "data" / "ab_arms" / "severity_metric_dynamic_range.json"

# one configuration for every arm, in the family the severity models are trained in
FIXED_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.7,
    max_depth=1,
    n_estimators=300,
    learning_rate=0.1,
    random_state=42,
)


def main() -> int:
    import ablation_matrix as M

    g = M.notebook_prefix(M.SPEC["amount"]["notebook"], M.SPEC["amount"]["stop"])
    import numpy as np
    import pandas as pd
    from sklearn.metrics import root_mean_squared_error
    from xgboost import XGBRegressor
    from AutoCarver import Features

    cats = list(g["categorical_columns"])
    ordinal_columns = g["ordinal_columns"]
    nums = list(g["numerical_columns"])
    x_train, x_dev = g["x_train"], g["x_dev"]
    y_train, y_dev = g["y_train"], g["y_dev"]
    ords = list(ordinal_columns)

    class Args:  # the optbinning defaults the article's arm used
        optb_max_bins = 5
        optb_ordinals = "categorical"
        optb_bin_numericals = "no"

    builders = {
        "AutoCarver": lambda: M.build_autocarver(
            x_train, x_dev, y_train, y_dev, cats, ordinal_columns, nums, "amount"
        )[:3],
        "no carving": lambda: M.build_nocarve(x_train, x_dev, cats, ordinal_columns, nums),
        "optbinning": lambda: M.build_optbinning(
            x_train, x_dev, np.asarray(y_train, dtype=float), cats, ordinal_columns, nums, Args()
        ),
    }

    merged = pd.read_csv(REPO / "data" / "frequency_2026.csv", low_memory=False)
    merged.set_index("ID", inplace=True)

    results = {"fixed_params": FIXED_PARAMS, "arms": {}}
    for name, build in builders.items():
        t0 = time.perf_counter()
        train_b, dev_b, meta = build()
        if name == "no carving":
            features = Features(categoricals=cats) + Features(numericals=ords + nums)
        else:
            features = Features(categoricals=list(train_b.columns))
        best = M.run_selector("amount", features, train_b, y_train, M.SPEC["amount"]["n_best"])

        carried = [c for c in merged.columns if c not in train_b.columns]
        train_j = train_b.join(merged[carried])
        dev_j = dev_b.join(merged[carried])
        w_train = (train_j["pred_sum"] - train_j["FREQ"]).abs()

        model = XGBRegressor(device="cuda", n_jobs=-1, **FIXED_PARAMS)
        model.fit(train_j[best], y_train.where(y_train >= 0, 0), sample_weight=w_train)
        predicted = model.predict(dev_j[best])

        row = {
            "rmse_dev": float(root_mean_squared_error(y_dev, predicted)),
            **M.severity_extras(y_dev, predicted),
            "n_modelled_features": len(best),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        results["arms"][name] = row
        print(
            f"{name:12} rmse {row['rmse_dev']:9.2f} | spearman {row['spearman_dev']:+.4f} | "
            f"top-decile lift {row['top_decile_lift_dev']:.3f} | tweedie deviance "
            f"{row['tweedie_deviance_dev']:.2f}",
            flush=True,
        )

    constant = float(y_train.mean())
    flat = np.full(len(y_dev), constant)
    results["constant_predictor"] = {
        "value": constant,
        "rmse_dev": float(root_mean_squared_error(y_dev, flat)),
        # a flat prediction has no ranking at all; spearman is undefined, lift is 1 by
        # construction, so only the deviance is meaningful as a floor
        "tweedie_deviance_dev": M.severity_extras(y_dev, flat + 1e-9)["tweedie_deviance_dev"],
    }
    print(
        f"{'constant':12} rmse {results['constant_predictor']['rmse_dev']:9.2f} | "
        f"spearman   0.0000 | top-decile lift 1.000 | tweedie deviance "
        f"{results['constant_predictor']['tweedie_deviance_dev']:.2f}",
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
