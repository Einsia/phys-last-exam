# P9 evaluator utilities

This directory is the normalized utility view required by the delivery
layout. The evaluator's original sibling modules are also retained one
level above so the frozen imports and hashes remain unchanged.

Utility modules: `continuous_scoring.py`, `evaluate_pre_continuous_0_100.py`, `inspect_inputs.py`, `inspect_videos.py`, `make_sanity_sheet.py`, `rescore_continuous.py`, `rescore_v2_geometric.py`, `run_batch.py`, `track_worker_p9.py`, `validate_metrics.py`.

Learned trackers/checkpoints and shared server checkouts referenced by a
frozen config are external runtime dependencies; they are intentionally
not copied into this artifact package.
