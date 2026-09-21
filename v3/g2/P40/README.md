<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P40 Sand Piles

M1 and M2 are `[0,1]` scores where higher is better. M1 is
`clip(1-|alpha_small/alpha_large-1|,0,1)` and M2 is
`clip(1-max_side_asymmetry_deg/90,0,1)`. Raw slope angles and residuals remain
in `verbose.measurements`. The evaluator uses OpenCV contours and
line fits, and requires two separated piles, visible ground, and stable profiles.

`evaluator/evaluate.py` is a standalone implementation. It imports no code from
other tasks or from the parent repository. Install its complete dependency set with
`python -m pip install -r evaluator/requirements.txt`.

Single video:

```text
python P40/evaluator/evaluate.py --video VIDEO.mp4 --task_id P40 \
  --output P40/eval_results/json/sample_00.json
```

The JSON contains `task_id`, `video_path`, `image_path`, `seed`, `model`, nested
`metrics.M1/M2` objects (`extract_success` and `metric`), and
the intermediate measurements/debug paths under `verbose`.

Batch evaluation:

```text
bash scripts/run_eval.sh minimax_h3
```

The batch script writes standard JSON results, debug plots/source overlays, and a
CSV summary under `P40/eval_results`.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.
