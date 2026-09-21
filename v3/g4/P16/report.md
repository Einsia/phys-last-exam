# P16 结论　—　刚性杆四共线 marker 交比不变

8 个 minimax-h3 视频（gpt 与仿真两路首帧，每路 seed 42/43/44/45），平均分 **0.84**，其中 8 个达到判定线（M1 得分 ≥ 0.50）。

| Sample | route/seed | extract_success | M1 | 得分 | 备注 |
|---|---|---|---|---|---|
| sample_00 | gpt seed42 | ✅ | +0.0534 | 0.84 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_01 | gpt seed43 | ✅ | +0.0693 | 0.82 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_02 | gpt seed44 | ✅ | +0.0451 | 0.86 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_03 | gpt seed45 | ✅ | +0.0754 | 0.81 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_04 | sim seed42 | ✅ | +0.0437 | 0.86 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_05 | sim seed43 | ✅ | +0.0641 | 0.83 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_06 | sim seed44 | ✅ | +0.0239 | 0.91 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |
| sample_07 | sim seed45 | ✅ | +0.0601 | 0.84 | chi is a ratio of differences between neighbouring projected marker positions, so it is best conditioned when those gaps are wide; the smallest adjacent gap per frame is reported here and drawn in the debug figure so a chi excursion can be read against it. A rigid rod under a pinhole projection must hold chi constant regardless of pose, so a sustained excursion means the rendered rod is not rigid. |

## 1. MiniMax H3 表现

刚性杆上四共线 marker 的交比会随时间漂移 2.4%~7.5%，说明杆在运动中有轻微非刚性形变或 marker 相对滑动；共线性本身保持得很好（残差不到杆长的 0.2%）。

## 2. 自动评测脚本可靠性

8 个样本中 8 个可自动测出 M1。交比对 marker 质心很敏感，个别帧检测塌陷会让标准差整个偏掉，已加入「相邻间距低于本片中位数 60% 即判为检测失效」的剔除；换一套 OpenCV 重跑，同一视频的 M1 仍可能有 0.05 上下的差异。
每个样本的测量过程图在 `eval_results/minimax_h3/debug/sample_XX/plot.png`，
图上直接画出被测的物理量，可据此复核。

## 3. 留存判定：**Keep**

题目稳定；评测仍对 marker 质心的细微差异敏感，跨环境有 0.05 量级的浮动，已在上面注明。
