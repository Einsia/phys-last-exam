# P11 evaluator utilities

This directory is the normalized utility view required by the delivery
layout. The evaluator's original sibling modules are also retained one
level above so the frozen imports and hashes remain unchanged.

Utility modules: `audit_continuous_results.py`, `compare_batches.py`, `continuous_scoring.py`, `rescore_continuous.py`, `run_batch.py`, `test_continuous_scoring.py`, `validate_synthetic.py`.

Learned trackers/checkpoints and shared server checkouts referenced by a
frozen config are external runtime dependencies; they are intentionally
not copied into this artifact package.
