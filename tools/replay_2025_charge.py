"""Replay the 2025 pipeline to score its end-to-end `CHARGE` on the dev set.

Why this exists
---------------
The 2025 notebooks only ever computed `CHARGE` on the submission sample, where there are
no labels -- so the era we actually won with has no end-to-end dev score of its own. The
article quotes one (6639.9). This script is where that number comes from.

It **re-fits nothing**. It rebuilds the 2025 split, applies the saved 2025 carver, joins
the saved 2025 frequency hand-off, loads the saved 2025 XGBoost severity model, and
scores. The control is the severity RMSE: it must reproduce the 2025 notebook's own
recorded `RMSE Dev: 6613.817574532793`. If that matches, the `CHARGE` figure computed
alongside it comes from the same models the notebook used.

Artifacts
---------
The 2025 model artifacts are not in this repository -- they are gitignored build outputs
of the original working repo, and one of them (the target carver) is 56 MB. Point
`--legacy` at that checkout. Everything else (data, code) resolves from there too, so the
replay runs in the 2025 environment rather than against a migrated artifact.

Run it with the 2025 interpreter, not this repo's::

    ../caa-challenge-frequency/.venv-705/Scripts/python.exe \
        tools/replay_2025_charge.py --legacy ../caa-challenge-frequency

AutoCarver 7.0.5 wrote these carvers and the current line does not load them, which is
the whole reason the legacy virtualenv is required.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# the 2025 notebook's own recorded severity dev RMSE -- the control for this replay
RECORDED_SEVERITY_DEV_RMSE = 6613.817574532793
CONTROL_TOLERANCE = 1e-4

FREQ_VERSION = "012"  # frequency hand-off the 2025 severity notebook consumed
AMOUNT_CARVER_VERSION = "018"  # carver + best_features the 2025 severity model was fit on
AMOUNT_MODEL_VERSION = "019"  # the saved severity XGBoost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        default="../caa-challenge-frequency",
        help="checkout holding the 2025 model artifacts, data and helper modules",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the JSON record (default: data/ab_arms/replay_2025_charge.json "
        "next to this script's repository)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parents[1]
    legacy = Path(args.legacy).resolve()
    out_path = Path(args.out) if args.out else here / "data" / "ab_arms" / "replay_2025_charge.json"

    legacy_src = legacy / "src"
    if not (legacy_src / "model" / "amount" / f"{AMOUNT_MODEL_VERSION}_xgboost.json").exists():
        print(f"ERROR: 2025 artifacts not found under {legacy_src / 'model'}", file=sys.stderr)
        print("Point --legacy at the checkout that holds them.", file=sys.stderr)
        return 2

    # the 2025 helpers live in src/utils/ and were imported as top-level modules when the
    # notebooks ran; the data toolkit resolves its own data directory, so pin it explicitly
    sys.path.insert(0, str(legacy_src / "utils"))
    os.environ.setdefault("CAA_DATA_DIR", str(legacy / "data"))
    os.chdir(legacy_src)  # so "model/..." and "../data/..." resolve as they did in 2025

    import pandas as pd
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    from AutoCarver import BinaryCarver, ContinuousCarver
    from data_toolkit import Processor

    data_path = "../data/"
    target_col = "CM"

    # --- the 2025 severity notebook, cell by cell -------------------------------------
    # cell 0: load and join
    data = pd.read_csv(data_path + "train_input_Z61KlZo.csv", low_memory=False)
    data.set_index("ID", inplace=True)
    target = pd.read_csv(data_path + "train_output_DzPxaPY.csv", low_memory=False)
    target.set_index("ID", inplace=True)
    data = data.join(target.drop("ANNEE_ASSURANCE", axis=1))
    print("data", data.shape, flush=True)

    # cell 1: the split is stratified on the *discretized* claim amount, so the 2025
    # target carver has to be loaded to reproduce it
    target_carver = BinaryCarver.load("model/cm_carver_freq_tschuprowt.json")
    y_freq = target_carver.transform(data)[target_col]
    x_train, x_dev, y_train, y_dev = train_test_split(
        data, data[target_col], test_size=0.2, random_state=42, stratify=y_freq
    )
    print("split", x_train.shape, x_dev.shape, flush=True)

    # cell 4: feature engineering
    proc = Processor()
    x_train = proc.fit_transform(x_train)
    x_dev = proc.transform(x_dev)

    # cells 8 + 13: the saved carver, applied -- not re-fitted
    carver = ContinuousCarver.load(f"model/amount/{AMOUNT_CARVER_VERSION}_carver.json")
    x_train = carver.transform(x_train)
    x_dev = carver.transform(x_dev)
    print("carved", x_train.shape, x_dev.shape, flush=True)

    # cell 17: the frequency model's hand-off, as saved in 2025
    merged = pd.read_csv(data_path + f"frequency_{FREQ_VERSION}.csv", low_memory=False)
    merged.set_index("ID", inplace=True)
    x_train = x_train.join(merged[[c for c in merged.columns if c not in x_train.columns]])
    x_dev = x_dev.join(merged[[c for c in merged.columns if c not in x_dev.columns]])

    # cell 25 + 46: the saved feature list and the saved model
    with open(
        f"model/amount/{AMOUNT_CARVER_VERSION}_best_features.json", "r", encoding="utf-8"
    ) as handle:
        best_features = json.load(handle)
    xgb = XGBRegressor()
    xgb.load_model(f"model/amount/{AMOUNT_MODEL_VERSION}_xgboost.json")

    # the saved booster carries the exact columns it was fit on, which is more reliable
    # than re-deriving them from hyper-parameters we no longer have
    selected_features = xgb.get_booster().feature_names
    if selected_features is None:
        selected_features = best_features
    missing = [c for c in selected_features if c not in x_dev.columns]
    if missing:
        print(f"ERROR: {len(missing)} model features absent from the replayed frame:", file=sys.stderr)
        print(missing[:10], file=sys.stderr)
        return 3
    print(f"model features: {len(selected_features)} (best_features list: {len(best_features)})", flush=True)

    def numeric_view(frame):
        """Model inputs as numbers, whatever dtype pandas happened to infer.

        The 2025 feature engineering leans on `.replace()`, which silently downcast
        object to float on the pandas of the day and no longer does. The columns still
        hold the same numbers -- they just arrive as `object` now, which XGBoost rejects.
        Coercing them back is safe precisely because the control below reproduces the
        2025 notebook's severity RMSE; if the coercion changed a value, it would not.
        """
        block = frame[selected_features]
        objects = block.columns[block.dtypes == object]
        if len(objects):
            block = block.copy()
            for column in objects:
                block[column] = pd.to_numeric(block[column], errors="raise")
        return block, len(objects)

    # --- scoring ----------------------------------------------------------------------
    results = {}
    for label, frame, y_true in (("train", x_train, y_train), ("dev", x_dev, y_dev)):
        block, n_coerced = numeric_view(frame)
        if label == "dev":
            print(f"coerced {n_coerced} object-dtype model columns back to numeric", flush=True)
        severity_pred = xgb.predict(block)
        # severity RMSE, scored exactly as objectives.get_best_regression_model prints it:
        # against the raw amount, unweighted
        results[f"severity_rmse_{label}"] = float(root_mean_squared_error(y_true, severity_pred))
        # end-to-end CHARGE = expected claim count x predicted amount, the same
        # construction the 2026 notebook uses
        true_charge = target.loc[frame.index, "CHARGE"]
        pred_charge = frame["pred_sum"].to_numpy() * severity_pred
        results[f"charge_rmse_{label}"] = float(root_mean_squared_error(true_charge, pred_charge))

    for key, value in results.items():
        print(f"{key}: {value}")

    # --- control ----------------------------------------------------------------------
    delta = abs(results["severity_rmse_dev"] - RECORDED_SEVERITY_DEV_RMSE)
    ok = delta < CONTROL_TOLERANCE
    print(
        f"\ncontrol: severity dev RMSE {results['severity_rmse_dev']:.6f} vs the 2025 "
        f"notebook's recorded {RECORDED_SEVERITY_DEV_RMSE:.6f} -- delta {delta:.2e} "
        f"({'MATCH' if ok else 'MISMATCH'})"
    )
    if not ok:
        print(
            "The replay does not reproduce the 2025 notebook, so its CHARGE figure is not "
            "the 2025 model's. Do not quote it.",
            file=sys.stderr,
        )

    results["control_recorded_severity_rmse_dev"] = RECORDED_SEVERITY_DEV_RMSE
    results["control_delta"] = delta
    results["control_passed"] = ok
    results["n_model_features"] = len(selected_features)
    results["artifacts"] = {
        "target_carver": "model/cm_carver_freq_tschuprowt.json",
        "amount_carver": f"model/amount/{AMOUNT_CARVER_VERSION}_carver.json",
        "amount_model": f"model/amount/{AMOUNT_MODEL_VERSION}_xgboost.json",
        "frequency_handoff": f"data/frequency_{FREQ_VERSION}.csv",
        "legacy_checkout": str(legacy),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
