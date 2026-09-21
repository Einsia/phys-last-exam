# P12 evaluator

This evaluator follows the P38 interface: `evaluate.py` evaluates one video and `batch.py` evaluates all direct `.mp4` inputs using explicit metadata associations. It records PTS, source previews, motion diagnostics, task-named intermediate videos, masks/tracks placeholders, and a result JSON. Semantic extraction is conservative and returns `metric: null` when object identity or geometry cannot be validated; it never substitutes prompt text for measurements.

## Single video

```bash
python videos/g7/P12/evaluator/evaluate.py \
  --video_path videos/g7/P12/continuation.mp4 \
  --image_path videos/g7/P12/first_frame.png \
  --video_prompt_file videos/g7/P12/video.txt \
  --model minimax_h3 --sample_id sample_00 \
  --output videos/g7/P12/eval_results/minimax_h3/result_sample_00.json
```

## Batch

```bash
python videos/g7/P12/evaluator/batch.py minimax_h3
```

`--device`, `--seed`, `--margin`, and `--config` are accepted. Unknown metadata is written as `null`; no sample is numbered from directory order. `P47` is intentionally excluded from this set.
