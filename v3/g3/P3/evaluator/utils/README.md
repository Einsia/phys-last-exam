# P3 evaluator utilities

This directory is the normalized utility view required by the delivery
layout. The evaluator's original sibling modules are also retained one
level above so the frozen imports and hashes remain unchanged.

Utility modules: `audit_formal_artifacts.py`, `rescore_continuous.py`, `rescore_v2_geometric.py`, `run_batch.py`, `run_batch_pre_continuous_aggregate_20260826.py`, `test_rescore_continuous.py`, `track_worker_p3.py`, `validate_synthetic.py`, `write_human_sanity_review.py`.

Learned trackers/checkpoints and shared server checkouts referenced by a
frozen config are external runtime dependencies; they are intentionally
not copied into this artifact package.
