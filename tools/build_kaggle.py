"""Build the Kaggle copies of the 2026 notebooks and their two helper modules from ``src/``.

The Kaggle notebooks are never edited on Kaggle. They are generated here from the repository
notebooks, pushed, and thrown away, so the two cannot drift: ``kaggle/`` commits only each
kernel's ``kernel-metadata.json``, and this script writes the code file beside it (gitignored).

Every way the Kaggle copy differs from the repository is listed in ``NOTEBOOKS[...]["edits"]``
below, each as an exact string that must occur **exactly once** in the source notebook. If a
repository notebook changes underneath one of them, the build fails instead of pushing a
half-patched copy.

Usage, from the repo root::

    .venv/Scripts/python.exe tools/build_kaggle.py          # writes kaggle/*/ code files
    kaggle kernels push -p kaggle/data-toolkit               # utility scripts first:
    kaggle kernels push -p kaggle/objectives                 # the notebooks import them
    kaggle kernels push -p kaggle/frequency
    kaggle kernels push -p kaggle/amount
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KAGGLE = ROOT / "kaggle"
REPO_URL = "https://github.com/mdefrance/caa-challenge/blob/main/src/"
KAGGLE_DATA = "/kaggle/input/datasets/mariodefrance/caa-challenge-2025/"

# Kaggle's image ships a numpy older than AutoCarver's floor, so installing upgrades numpy
# underneath a kernel that has already imported it, and the kernel must restart. Kaggle's batch
# runner treats that restart as DeadKernelError, so batch runs rely on the notebook's package
# requirement `AutoCarver==7.7.3`, installed before the kernel starts (see kaggle/README.md);
# this cell then finds AutoCarver present and only prints versions. It still installs and
# restarts in an interactive session without that setting.
INSTALL_CELL = '''import importlib.metadata as md
import subprocess
import sys

REQ = ["AutoCarver>=7.7.3"]

try:
    autocarver_version = md.version("AutoCarver")
except md.PackageNotFoundError:
    autocarver_version = None

if autocarver_version is None:
    # the install upgrades numpy; that only breaks a kernel that has already imported it
    numpy_loaded = "numpy" in sys.modules
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQ], check=True)
    if numpy_loaded:
        print("installed -- numpy was already loaded, so the kernel restarts: run from the top")
        import IPython

        IPython.Application.instance().kernel.do_shutdown(restart=True)
    else:
        print("installed AutoCarver", md.version("AutoCarver"), "| numpy", md.version("numpy"))
else:
    print("AutoCarver", autocarver_version, "| numpy", md.version("numpy"), "| scipy", md.version("scipy"))'''

BANNER = """> **Kaggle copy, generated. Do not edit it here.** Built from
> [`src/{src}`]({url}{src}) by `tools/build_kaggle.py`.
>
> To fit a free Kaggle CPU session it runs **{trials} Optuna trials instead of {full}** and
> carves with **`N_JOBS = 1` instead of 6**, so its numbers will **not** match the article's.
> The article's numbers are the stored outputs of the repository notebook, which is seeded and
> reproduces them exactly at full budget.
>
> Run the first cell. If it restarts the kernel, run from the top again.{extra}"""

AMOUNT_EXTRA = """
>
> It reads the frequency hand-off (`frequency_2026.csv`, `oos_frequency_2026.csv`, the `CM`
> target carver) from the dataset. Those are the article's own seed-42 outputs, so this
> notebook runs on its own and does not consume the frequency notebook's reduced-budget run."""

HANDOFF_NOTE = "  # for inspection only: caa-amount-model reads the article's copy from the dataset"

NOTEBOOKS = {
    "frequency": {
        "src": "frequency_model_2026.ipynb",
        "code_file": "caa-frequency-model.ipynb",
        "trials": 30,
        "full": 300,
        "extra": "",
        "edits": [
            ('data_path = "../data/"', f'data_path = "{KAGGLE_DATA}"'),
            ("N_JOBS = 6\n", "N_JOBS = 1  # 6 in the article's run\n"),
            ("N_TRIALS = 300\n", "N_TRIALS = 30  # 300 in the article's run\n"),
            ("output_path = data_path\n", 'output_path = "/kaggle/working/"\n'),
            (
                'merged.to_csv(output_path + f"frequency_{FREQ_VERSION}.csv")',
                'merged.to_csv(output_path + f"frequency_{FREQ_VERSION}.csv")' + HANDOFF_NOTE,
            ),
            (
                '# pred.to_csv(output_path + "predictions/2026_pred.csv", index=False)',
                'pred.to_csv(output_path + "2026_pred.csv", index=False)',
            ),
            (
                'oos.to_csv(output_path + f"oos_frequency_{FREQ_VERSION}.csv", index=False)',
                'oos.to_csv(output_path + f"oos_frequency_{FREQ_VERSION}.csv", index=False)'
                + HANDOFF_NOTE,
            ),
            (
                'TARGET_CARVER_PATH = Path("model/cm_carver_freq_tschuprowt_2026.json")',
                'TARGET_CARVER_PATH = Path(output_path + "cm_carver_freq_tschuprowt_2026.json")',
            ),
            (
                "target_carver.save(TARGET_CARVER_PATH, light_mode=True)",
                "target_carver.save(TARGET_CARVER_PATH, light_mode=True)" + HANDOFF_NOTE,
            ),
        ],
    },
    "amount": {
        "src": "amount_model_2026.ipynb",
        "code_file": "caa-amount-model.ipynb",
        "trials": 30,
        "full": 400,
        "extra": AMOUNT_EXTRA,
        "edits": [
            ('data_path = "../data/"', f'data_path = "{KAGGLE_DATA}"'),
            (
                'BinaryCarver.load(Path("model/cm_carver_freq_tschuprowt_2026.json"))',
                'BinaryCarver.load(Path(data_path + "cm_carver_freq_tschuprowt_2026.json"))',
            ),
            ("N_JOBS = 6\n", "N_JOBS = 1  # 6 in the article's run\n"),
            ("N_TRIALS = 400\n", "N_TRIALS = 30  # 400 in the article's run\n"),
            (
                '# pred.to_csv("predictions/2026_amount_pred.csv", index=False)',
                'pred.to_csv("/kaggle/working/2026_amount_pred.csv", index=False)',
            ),
        ],
    },
}

# utility scripts: copied verbatim, Kaggle imports them under the module name of the file
SCRIPTS = {"data-toolkit": "data_toolkit.py", "objectives": "objectives.py"}


def apply_edits(nb: nbformat.NotebookNode, edits: list[tuple[str, str]], name: str) -> None:
    for old, new in edits:
        hits = [c for c in nb.cells if old in c.source]
        count = sum(c.source.count(old) for c in hits)
        if count != 1:
            raise SystemExit(f"{name}: expected exactly one {old!r}, found {count}")
        hits[0].source = hits[0].source.replace(old, new)


def build_notebook(key: str, spec: dict) -> Path:
    nb = nbformat.read(SRC / spec["src"], as_version=4)
    apply_edits(nb, spec["edits"], spec["src"])
    for cell in nb.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    banner = BANNER.format(
        src=spec["src"], url=REPO_URL, trials=spec["trials"], full=spec["full"],
        extra=spec["extra"],
    )
    nb.cells[1:1] = [nbformat.v4.new_markdown_cell(banner), nbformat.v4.new_code_cell(INSTALL_CELL)]
    for cell in nb.cells[1:3]:
        cell.pop("id", None)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    nb.nbformat_minor = 4
    out = KAGGLE / key / spec["code_file"]
    nbformat.write(nb, out)
    return out


def check_metadata(folder: Path, code_file: str) -> None:
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta["code_file"] != code_file:
        raise SystemExit(f"{folder.name}: metadata code_file {meta['code_file']!r} != {code_file!r}")
    if meta.get("is_private", True):
        raise SystemExit(f"{folder.name}: is_private must be false, readers cannot run it otherwise")


def main() -> None:
    for key, spec in NOTEBOOKS.items():
        check_metadata(KAGGLE / key, spec["code_file"])
        print("wrote", build_notebook(key, spec).relative_to(ROOT))
    for key, module in SCRIPTS.items():
        code_file = f"{key}.py"
        check_metadata(KAGGLE / key, code_file)
        shutil.copyfile(SRC / module, KAGGLE / key / code_file)
        print("wrote", (KAGGLE / key / code_file).relative_to(ROOT))


if __name__ == "__main__":
    main()
