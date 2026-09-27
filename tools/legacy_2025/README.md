# The 2025 feature engineering, verbatim

`data_toolkit.py` is the helper module the 2025 notebooks imported, copied byte for byte from
the original working repository. `tools/replay_2025_charge.py` needs it, because the saved
2025 severity model was fitted on features built by this version, and today's
`src/data_toolkit.py` (renamed helpers, flat layout) does not rebuild them identically.

Do not edit it. It only runs under the 2025 environment (`requirements-705.txt`).
