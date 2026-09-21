# P4 evaluator utilities

This directory is the normalized utility view required by the delivery
layout. The evaluator's original sibling modules are also retained one
level above so the frozen imports and hashes remain unchanged.

Utility modules: `audit_batch.py`, `continuous_scoring.py`, `evaluate_pre_continuous_0_100.py`, `rescore_continuous.py`, `rescore_existing.py`, `rescore_v2_geometric.py`, `run_all.py`, `validate_synthetic.py`.

Learned trackers/checkpoints and shared server checkouts referenced by a
frozen config are external runtime dependencies; they are intentionally
not copied into this artifact package.
