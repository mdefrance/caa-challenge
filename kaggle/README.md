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

**Each notebook needs one setting the push cannot carry:** the Kaggle package requirement
`AutoCarver==7.7.3` (notebook settings, requirements; 7.8.0 works too, but 7.7.3 is what the
article measured). Kaggle installs it before the kernel starts, which is the only way past the
numpy trap in a batch run: AutoCarver needs a newer numpy than the image, and installing it
inside a running kernel forces a restart that Kaggle's runner treats as `DeadKernelError`. Two traps in that field, both hit on 2026-09-22:

- **No quotes.** The line is split on spaces with no shell, so `"AutoCarver>=7.7.3"` reaches
  pip with the quotes in it and fails as an invalid requirement.
- **No `>` or `<`.** The install step does go through a shell, so `AutoCarver>=7.7.3` redirects
  pip's output into a file named `=7.7.3` and installs whatever AutoCarver is newest. Use `==`.

`kernel-metadata.json` has no field for it, so check the setting survives each push. The
notebook's own install cell stays as the fallback for interactive sessions.

All four must stay **public** (`"is_private": false`, which the build checks); a public
notebook that imports a private utility script fails for every reader but its owner. Keep each
`title` as it is, because Kaggle derives the URL slug from the title and the article links
those URLs.

## Datasets

Each dataset folder tracks only its `dataset-metadata.json`; the files beside it are built or
downloaded, and gitignored.

| Folder | Kaggle | Contents |
|---|---|---|
| `caa-challenge-2025/` | [caa-challenge-2025](https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025) | the challenge files, `Incendies.csv` and the 2026 hand-off. Only the metadata is kept here; update the description with `kaggle datasets metadata mariodefrance/caa-challenge-2025 --update -p kaggle/caa-challenge-2025` |
| `solution-2025/` | [caa-challenge-2025-solution](https://www.kaggle.com/datasets/mariodefrance/caa-challenge-2025-solution) | the saved 2025 models the replay needs, built by `tools/build_2025_solution.py` from the original working checkout |

A new version of a dataset replaces all of its files, so never push `caa-challenge-2025/` from
local copies: that would re-upload ~1.9 GB and could swap mirror files for non-identical ones.
