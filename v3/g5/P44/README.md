<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P44 evaluator

This task has an independent bright-chain skeleton extractor and catenary fit.
M1 and M2 are `[0,1]` scores where higher is better. M1 is
`clip(1-fit_rmse_norm,0,1)` and M2 is
`clip(1-endpoint_height_error_norm,0,1)`. Raw fit and endpoint errors remain in
`verbose.measurements`. M2 uses independently detected left/right support points
when available (falling back to the fitted endpoint estimate). Every sample
includes the raw skeleton, fit parameters, support points, `plot.png`, and an
annotated `overlay.mp4` under `eval_results/debug`.

Run `bash scripts/run_eval.sh minimax_h3` to evaluate standard samples, or use the
standalone evaluator's `--video/--output` interface for one sample.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.
