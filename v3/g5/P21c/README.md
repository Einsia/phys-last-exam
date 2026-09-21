<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P21c evaluator

The self-contained evaluator tracks the liquid surface and an ice-texture
component. M1 and M2 are `[0,1]` scores where higher is better. M1 is `1` when
`water_height_change_norm > 0` and `0` otherwise because the physical metric
specifies direction but no target magnitude. M2 is
`clip(1-solid_evidence_ratio,0,1)`. M2 has `extract_success: false` only when
the initial ice evidence is unavailable; visible residual ice is a measurable
physical failure and receives a low score. Raw values remain in
`verbose.measurements`. `plot.png`, `measurements.json`, and
`overlay.mp4` are generated for every sample under `eval_results/debug`.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.
The final frame is checked independently for residual ice; if detected, M2 is
forced to `0` and the detection evidence is retained in the JSON.

Use `bash scripts/run_eval.sh minimax_h3` for batch evaluation or invoke the standalone
Python evaluator with `--video ... --output ...`.
