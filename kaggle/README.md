# Kaggle kernels

The four Kaggle kernels behind the article's "run it without installing anything" links.
Only each kernel's `kernel-metadata.json` is committed. The code beside it is **generated**
from `src/` by `tools/build_kaggle.py` and gitignored, so the Kaggle copies cannot drift from
the repository notebooks. Never edit them on Kaggle; change `src/`, rebuild, push.

| Folder | Kaggle | Built from |
|---|---|---|
| `frequency/` | [caa-frequency-model](https://www.kaggle.com/code/mariodefrance/caa-frequency-model) | `src/frequency_model_2026.ipynb` |
| `amount/` | [caa-amount-model](https://www.kaggle.com/code/mariodefrance/caa-amount-model) | `src/amount_model_2026.ipynb` |
| `data-toolkit/` | utility script `data_toolkit` | `src/data_toolkit.py`, verbatim |
| `objectives/` | utility script `objectives` | `src/objectives.py`, verbatim |

Every difference between a Kaggle notebook and its source is listed in the `edits` lists inside
`tools/build_kaggle.py`: the dataset path, a reduced budget (30 Optuna trials, `N_JOBS = 1`),
outputs written to `/kaggle/working/`, a banner saying so, and an install-then-restart first
cell. Each edit must match exactly once, so a change in `src/` that breaks one fails the build.

```bash
.venv/Scripts/python.exe tools/build_kaggle.py
kaggle kernels push -p kaggle/data-toolkit
kaggle kernels push -p kaggle/objectives
kaggle kernels push -p kaggle/frequency
kaggle kernels push -p kaggle/amount
```

Push the utility scripts first, since both notebooks import them.

**A pushed notebook's own run always fails, and that is expected.** AutoCarver needs a newer
numpy than Kaggle's image, so installing it forces a kernel restart, and Kaggle's batch runner
treats that restart as `DeadKernelError`. So the push only delivers the code. To publish a run,
open each notebook in the Kaggle editor, run the first cell (it restarts the kernel), then
**Run All**, then **Save Version → Quick Save**. Quick Save keeps the interactive outputs.

All four must stay **public** (`"is_private": false`, which the build checks); a public
notebook that imports a
private utility script fails for every reader but its owner. Keep each `title` as it is,
because Kaggle derives the URL slug from the title and the article links those URLs.
