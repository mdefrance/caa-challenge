"""Scrub local filesystem paths out of saved notebook outputs.

Warning tracebacks captured by ``nbconvert --execute --inplace`` carry the absolute
paths of the machine that ran them (home directory, temp kernel files, venv). Those
are noise in a public repository and leak a username. This rewrites them to stable
placeholders **in outputs only** — cell sources are never touched, and no output is
deleted, so the recorded numbers stay exactly as they were measured.

Run from the repository root::

    python tools/scrub_notebook_paths.py            # scrub every notebook under src/
    python tools/scrub_notebook_paths.py --check    # report only, exit 1 if any hit
    python tools/scrub_notebook_paths.py src/a.ipynb src/b.ipynb
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]

# Order matters: the most specific pattern wins. Each is matched against both the
# raw form and the JSON-escaped (doubled-backslash) form that ends up in outputs.
SUBSTITUTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(re.escape(str(REPO_ROOT)) + r"[\\/]?", re.IGNORECASE), "<repo>/"),
    (
        re.compile(
            r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+[\\/]AppData[\\/]Local[\\/]Temp[\\/]?"
        ),
        "<tmp>/",
    ),
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+[\\/]?"), "<home>/"),
    (re.compile(r"/(?:home|Users)/[^/\s\"']+/"), "<home>/"),
]

OUTPUT_TEXT_KEYS = ("text", "evalue", "ename")
DATA_MIME_KEYS = ("text/plain", "text/html", "text/markdown", "application/json")


def scrub_text(text: str) -> tuple[str, int]:
    """Returns the scrubbed text and the number of substitutions made."""
    total = 0
    for pattern, replacement in SUBSTITUTIONS:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


def scrub_value(value):
    """Recursively scrubs strings inside notebook output values."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        out, total = [], 0
        for item in value:
            scrubbed, n = scrub_value(item)
            out.append(scrubbed)
            total += n
        return out, total
    return value, 0


def scrub_notebook(path: Path, write: bool) -> int:
    notebook = nbformat.read(path, as_version=4)
    total = 0

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            for key in OUTPUT_TEXT_KEYS:
                if key in output:
                    output[key], n = scrub_value(output[key])
                    total += n
            if "traceback" in output:
                output["traceback"], n = scrub_value(output["traceback"])
                total += n
            for key in DATA_MIME_KEYS:
                if key in output.get("data", {}):
                    output["data"][key], n = scrub_value(output["data"][key])
                    total += n

    if total and write:
        nbformat.write(notebook, path)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="report only, do not rewrite"
    )
    args = parser.parse_args()

    targets = [p.resolve() for p in args.notebooks] or sorted(
        (REPO_ROOT / "src").glob("*.ipynb")
    )
    grand_total = 0

    for path in targets:
        count = scrub_notebook(path, write=not args.check)
        grand_total += count
        verb = "found" if args.check else "scrubbed"
        label = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        print(f"{label}: {verb} {count}")

    if args.check and grand_total:
        print(f"\n{grand_total} local path(s) still present", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
