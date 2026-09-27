"""Assemble the Kaggle dataset that publishes the 2025 solution.

The article's 2025 `CHARGE` baseline is scored by `tools/replay_2025_charge.py` from the
saved 2025 models, which only ever lived in the original working checkout. This script
copies the files the replay needs into `kaggle/solution-2025/`, beside the tracked
`dataset-metadata.json`, so they can be pushed as
https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025-solution.

The one transformation is the frequency hand-off. The 2025 file (`frequency_012.csv`) is
2.4 GB because it carries every carved column; the replay only reads `pred_sum` and the
columns the saved severity model was fitted on. The published copy keeps `ID`, `pred_sum`
and those columns, copied as text (no numeric parsing, so every value keeps its exact text), gzipped.

Run from the repository root with any Python::

    python tools/build_2025_solution.py --legacy ../caa-challenge-frequency
"""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# the four saved 2025 artifacts the severity replay loads, copied unchanged
COPIED = {
    "cm_carver_freq_tschuprowt.json": "src/model/cm_carver_freq_tschuprowt.json",
    "018_carver.json": "src/model/amount/018_carver.json",
    "018_best_features.json": "src/model/amount/018_best_features.json",
    "019_xgboost.json": "src/model/amount/019_xgboost.json",
}
HANDOFF_SOURCE = "data/frequency_012.csv"
HANDOFF_NAME = "frequency_2025_handoff.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", default="../caa-challenge-frequency")
    parser.add_argument("--out", default=str(HERE / "kaggle" / "solution-2025"))
    args = parser.parse_args()

    legacy = Path(args.legacy).resolve()
    out = Path(args.out).resolve()
    if not (out / "dataset-metadata.json").exists():
        print(f"ERROR: {out / 'dataset-metadata.json'} missing", file=sys.stderr)
        return 2

    for name, relative in COPIED.items():
        shutil.copyfile(legacy / relative, out / name)

    # columns the saved severity model reads, straight from the booster's own record
    model = json.loads((legacy / COPIED["019_xgboost.json"]).read_text(encoding="utf-8"))
    model_features = set(model["learner"]["feature_names"])

    source = legacy / HANDOFF_SOURCE
    csv.field_size_limit(2**31 - 1)
    with source.open("r", encoding="utf-8", newline="") as raw, \
            gzip.open(out / HANDOFF_NAME, "wt", encoding="utf-8", newline="") as slim:
        reader = csv.reader(raw)
        header = next(reader)
        keep = [i for i, column in enumerate(header)
                if column in ("ID", "pred_sum") or column in model_features]
        writer = csv.writer(slim, lineterminator="\n")
        writer.writerow([header[i] for i in keep])
        rows = 0
        for row in reader:
            writer.writerow([row[i] for i in keep])
            rows += 1
    print(f"{HANDOFF_NAME}: {len(keep)} of {len(header)} columns, {rows} rows")

    for path in sorted(out.iterdir()):
        if path.name == "dataset-metadata.json":
            continue
        print(f"{path.name:36s} {path.stat().st_size / 1e6:9.2f} MB  sha256 {sha256(path)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
