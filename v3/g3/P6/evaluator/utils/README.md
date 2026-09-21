# P6 evaluator utilities

This directory is the normalized utility view required by the delivery
layout. The evaluator's original sibling modules are also retained one
level above so the frozen imports and hashes remain unchanged.

Utility modules: `audit_results.py`, `batch_evaluate.py`, `batch_evaluate_pre_future_continuous_20260826.py`, `build_dense_audit_sheets.py`, `build_group_audit_sheets.py`, `evaluate_pre_future_continuous_20260826.py`, `p6_make_contact.py`, `p6_overlay_sheet.py`, `p6_probe_diff.py`, `p6_probe_hough.py`, `probe_visibility.py`, `rescore_continuous.py`, `rescore_continuous_pre_future_batch_20260826.py`, `rescore_v2_geometric.py`, `summarize_results.py`, `test_continuous_scoring.py`, `test_future_continuous_integration.py`.

Learned trackers/checkpoints and shared server checkouts referenced by a
frozen config are external runtime dependencies; they are intentionally
not copied into this artifact package.
