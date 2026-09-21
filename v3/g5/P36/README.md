<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P36 evaluator

The standalone evaluator detects the gray magnet block and white control block,
tracks both with OpenCV Lucas-Kanade optical flow, and measures displacement
onset and fitted speed. M1 and M2 are `[0,1]` scores where higher is better:
`M1=clip(1-1/(t_mag/t_ctrl),0,1)` and
`M2=clip(1-v_mag/v_ctrl,0,1)`. The raw ratios remain in
`verbose.measurements`. If the two blocks cannot be separated, the
sample is marked `extract_success: false` with a specific failure reason.

`eval_results/debug/<sample>/plot.png` contains both displacement curves,
onset markers, fit residuals and sample counts; `overlay.mp4` shows the ROI,
tracked points and onset labels. Run `bash scripts/run_eval.sh minimax_h3`; it
evaluates the standard samples under `output_videos/minimax_h3`.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.
