# Superseded: fixed-feature arms built in the wrong column order

Measured 2026-09-08/10. These arms model the right 100 (or 200) features, but the harness
rebuilt the frame grouped by type -- categoricals, then ordinals, then numericals -- instead
of keeping the shared list's own order, which is the order the pipeline's selector ranked
them in and the order the own-selection arms feed to the model.

Position is not neutral. The search tunes `colsample_bytree` and `colsample_bylevel`, so
XGBoost draws columns per tree and per level *by position*. Same features in a different
order is a different model. Everything stayed deterministic -- each arm reproduces itself
exactly -- but the fixed-set arms could not be compared seed-by-seed against the
own-selection arms, and the AutoCarver fixed arm returned 0.9216 where it should have
reproduced 0.9199.

Between-arm comparisons within this set were still fair, since all arms shared the same
(wrong) order. Replaced by runs that feed the shared list in its own order.
