# Superseded: optbinning arms that binned the numerical features

Measured 2026-09-07/08. In these runs `BinningProcess` was given every column, including
the 120 numericals, while the AutoCarver arm followed the pipeline and carved only the
qualitative features, leaving numericals to the selector.

That compares two different scopes, not two binners. The reference benchmark
(`AutoCarver/docs/source/examples/Comparison/comparison_notebook.ipynb`) is explicit that
the protocol must be "identical for every library", so these runs are kept only as a
record and no number in `ARTICLE.md` should cite them.

Replaced by runs with `--optb-bin-numericals no`, where numericals pass through raw in
both arms.
