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
The 2025 model artifacts are not in this repository (one of them, the target carver, is
56 MB). They are published as the Kaggle dataset
https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025-solution, built by
`tools/build_2025_solution.py`. Download it and point `--artifacts` at the folder; the
challenge CSVs and `Incendies.csv` are read from this repository's `data/` (or `--data`),
and the 2025 feature engineering from `tools/legacy_2025/data_toolkit.py`.

Run it with the 2025 interpreter (`requirements-705.txt`), not this repo's::

    .venv-705/Scripts/python.exe tools/replay_2025_charge.py --artifacts path/to/solution

`--legacy` instead reads everything from the original working checkout, which is how the
published figure was first produced::

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
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--artifacts",
        default=None,
        help="folder holding the published 2025 solution (Kaggle dataset "
        "mariodefrance/caa-challenge-2025-solution)",
    )
    source.add_argument(
        "--legacy",
        default=None,
        help="original working checkout holding the 2025 model artifacts, data and helpers "
        "(default when --artifacts is not given: ../caa-challenge-frequency)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="folder with the challenge CSVs and Incendies.csv (default: this repository's "
        "data/ with --artifacts, <legacy>/data with --legacy)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the JSON record (default: data/ab_arms/replay_2025_charge.json "
        "next to this script's repository)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parents[1]
    out_path = Path(args.out) if args.out else here / "data" / "ab_arms" / "replay_2025_charge.json"

    if args.artifacts:
        artifacts = Path(args.artifacts).resolve()
        target_carver_path = artifacts / "cm_carver_freq_tschuprowt.json"
        amount_carver_path = artifacts / f"{AMOUNT_CARVER_VERSION}_carver.json"
        best_features_path = artifacts / f"{AMOUNT_CARVER_VERSION}_best_features.json"
        model_path = artifacts / f"{AMOUNT_MODEL_VERSION}_xgboost.json"
        handoff_path = artifacts / "frequency_2025_handoff.csv.gz"
        helper_dir = here / "tools" / "legacy_2025"
        data_dir = Path(args.data).resolve() if args.data else here / "data"
        source_label = str(artifacts)
    else:
        legacy = Path(args.legacy or "../caa-challenge-frequency").resolve()
        legacy_model = legacy / "src" / "model"
        target_carver_path = legacy_model / "cm_carver_freq_tschuprowt.json"
        amount_carver_path = legacy_model / "amount" / f"{AMOUNT_CARVER_VERSION}_carver.json"
        best_features_path = legacy_model / "amount" / f"{AMOUNT_CARVER_VERSION}_best_features.json"
        model_path = legacy_model / "amount" / f"{AMOUNT_MODEL_VERSION}_xgboost.json"
        handoff_path = legacy / "data" / f"frequency_{FREQ_VERSION}.csv"
        helper_dir = legacy / "src" / "utils"
        data_dir = Path(args.data).resolve() if args.data else legacy / "data"
        source_label = str(legacy)

    required = (target_carver_path, amount_carver_path, best_features_path, model_path,
                handoff_path, helper_dir / "data_toolkit.py",
                data_dir / "train_input_Z61KlZo.csv", data_dir / "train_output_DzPxaPY.csv",
                data_dir / "Incendies.csv")
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        print("ERROR: missing inputs:\n  " + "\n  ".join(missing_files), file=sys.stderr)
        return 2

    # the 2025 helpers were imported as top-level modules when the notebooks ran; the data
    # toolkit resolves its own data directory, so pin it explicitly
    sys.path.insert(0, str(helper_dir))
    os.environ["CAA_DATA_DIR"] = str(data_dir)

    import pandas as pd
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    from AutoCarver import BinaryCarver, ContinuousCarver
    from data_toolkit import Processor

    target_col = "CM"

    # --- the 2025 severity notebook, cell by cell -------------------------------------
    # cell 0: load and join
    data = pd.read_csv(data_dir / "train_input_Z61KlZo.csv", low_memory=False)
    data.set_index("ID", inplace=True)
    target = pd.read_csv(data_dir / "train_output_DzPxaPY.csv", low_memory=False)
    target.set_index("ID", inplace=True)
    data = data.join(target.drop("ANNEE_ASSURANCE", axis=1))
    print("data", data.shape, flush=True)

    # cell 1: the split is stratified on the *discretized* claim amount, so the 2025
    # target carver has to be loaded to reproduce it
    target_carver = BinaryCarver.load(str(target_carver_path))
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
    carver = ContinuousCarver.load(str(amount_carver_path))
    x_train = carver.transform(x_train)
    x_dev = carver.transform(x_dev)
    print("carved", x_train.shape, x_dev.shape, flush=True)

    # cell 17: the frequency model's hand-off, as saved in 2025 (the published copy keeps
    # only ID, pred_sum and the columns the saved severity model reads, which gives the
    # model identical inputs: every other hand-off column is either dropped by this join
    # or never selected)
    merged = pd.read_csv(handoff_path, low_memory=False)
    merged.set_index("ID", inplace=True)
    x_train = x_train.join(merged[[c for c in merged.columns if c not in x_train.columns]])
    x_dev = x_dev.join(merged[[c for c in merged.columns if c not in x_dev.columns]])

    # cell 25 + 46: the saved feature list and the saved model
    with open(best_features_path, "r", encoding="utf-8") as handle:
        best_features = json.load(handle)
    xgb = XGBRegressor()
    xgb.load_model(str(model_path))

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
        "target_carver": target_carver_path.name,
        "amount_carver": amount_carver_path.name,
        "amount_model": model_path.name,
        "frequency_handoff": handoff_path.name,
        "source": "--artifacts" if args.artifacts else "--legacy",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
