<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P47 evaluator（V2）

独立拟合两泡外轮廓和有符号隔膜圆弧，计算同帧三半径关系；投影或曲率不可辨时返回不可测。

当前仅实现 M1；M2 保留 null 占位。原始物理量保存在 `metric`，纯物理分、识别分和最终分分别保存。完整公式、参数来源和总分口径见 [SCORING_V2.md](../../SCORING_V2.md)。历史 `evaluator.md` 中旧评分和旧路径说明由该规范更新。

## 批量评测

从 v2 根目录运行：

```bash
bash g9/P47/scripts/run_eval.sh minimax_h3 --device cpu 
```

使用 `metadata_v2.json` 显式关联视频、首帧、prompt 和样本 ID；不按文件遍历顺序配对。未知 seed 保留 null。`--input_dir`、`--metadata` 和 `--output_dir` 可覆盖默认值。

## 文件

- [prompt.txt](prompt.txt)：实际视频生成 prompt。
- [首帧](first_frames/provided/first_frame_01.png) 与 [首帧 prompt](first_frames/provided/prompt.txt)。来源生成器未知，使用 provided 分类。
- [视频](output_videos/minimax_h3/sample_00.mp4)：源视频内容的无损复制。
- [单视频入口](evaluator/evaluate.py) 与 [依赖](evaluator/requirements.txt)。`--help` 列出元数据、设备和阈值参数。
- [JSON 结果](eval_results/minimax_h3/result_sample_00.json)。
- [数值代入与评分](eval_results/minimax_h3/debug_sample_00/scoring_calculation.md)。
- [测量可视化目录](eval_results/minimax_h3/debug_sample_00)：源帧对齐视频、实际轨迹/轮廓、物理量曲线和时间戳。

可测但物理错误仍为 `extract_success=true`；无法可靠测量为 false/null；没有定义的 M2 为 null/null。公开结果严格使用统一 JSON 字段，评分细节位于 `verbose.M1.scoring` 与 `verbose.M1._scoring_summary`。总分只对已定义指标等权平均；本题总分等于 M1 最终分，M2 不计入分母。

本次重跑不生成新视频。保留原始 `continuation.mp4`、`first_frame.png`、`video.txt` 和旧 metadata，历史结果备份位于 v2 的 `work/g8_g9_v2_20260910/before/`。
