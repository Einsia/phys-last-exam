# V3 evaluator source tree

The runnable entry points and task materials live here. Start with the repository
[README](../README.md). The top-level `unified_evaluators/` package implements the
V3 consistency gate, scoring contract, and backend runner; `g1/` through `g9/`
contain the forty task evaluators and their first-frame materials; `scripts/`
contains batch execution, reporting, and audit tools.

`SCORING_V3.md` is the scoring specification and `OUTPUT_FORMAT_V2.md` documents
the public JSON contract. Generated results, videos, caches, virtual environments,
and model checkpoints are intentionally kept outside Git.
