<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P21b evaluator

This task is evaluated by the standalone `evaluator/evaluate.py`; it imports no
code from another task. M1 and M2 are `[0,1]` scores where higher is better.
Because M1 specifies only a direction, it is `1` when
`water_height_change_norm < 0` and `0` otherwise. M2 is
`clip(1-stone_bottom_gap_norm,0,1)`. Raw changes and gaps remain in
`verbose.measurements`. Measurements and quality warnings are retained when a generated video
violates the physical condition.

Run `bash scripts/run_eval.sh minimax_h3` to evaluate every standard sample in
this task directory, or
pass a video/directory and output directory as the first two arguments. Results
are written to `eval_results/json`, `eval_results/debug`, and `results.csv`.

M2 is emitted when a compact, independently segmented stone is tracked. A
measured stone that does not reach the floor receives a lower score rather than
an extraction failure. An ice-plus-stone bounding box cannot satisfy M2. Run
`bash scripts/run_eval.sh --annotate` to mark `vessel`, `ice`, `stone_seed`, and
`floor` in `evaluator/roi.json`; the overlay shows separate stone and ice boxes
and both bottom trajectories.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.

The final frame is checked independently for residual ice. If residual ice is
detected, the stone/ice-completion M2 score is forced to `0`; the detection box,
confidence and warning remain in the JSON for audit.
