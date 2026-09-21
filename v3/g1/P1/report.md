# P1 结论　—　钢球自由下落

8 个 minimax-h3 视频（gpt 与仿真两路首帧，每路 seed 42/43/44/45），平均分 **0.26**，其中 0 个达到判定线（M1 得分 ≥ 0.50）。

| Sample | route/seed | extract_success | M1 | 得分 | 备注 |
|---|---|---|---|---|---|
| sample_00 | gpt seed42 | ✅ | +1.0173 | 0.24 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_01 | gpt seed43 | ❌ | - | 0.00 | ball tracked for only 4 consecutive frames |
| sample_02 | gpt seed44 | ✅ | +0.3592 | 0.30 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_03 | gpt seed45 | ✅ | +0.4748 | 0.28 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_04 | sim seed42 | ✅ | +0.4266 | 0.31 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_05 | sim seed43 | ✅ | +0.1942 | 0.37 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_06 | sim seed44 | ✅ | +0.5703 | 0.27 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |
| sample_07 | sim seed45 | ✅ | +0.5601 | 0.29 | a large value means the velocity increments are not consistent, i.e. the fall is not uniformly accelerated; CV grows without bound as the mean increment approaches zero, which is what a constant-velocity fall produces |

## 1. MiniMax H3 表现

8 个视频里球都在画面内，但没有一个是加速下落：等时间隔速度增量的变异系数 0.19~1.02，位移比也停在 1:1.1:1.2 附近而不是 1:3:5，也就是模型把自由落体渲染成了近似匀速。gpt seed43 在第 4 帧后球就脱离画面，无法评测。

## 2. 自动评测脚本可靠性

8 个样本中 7 个可自动测出 M1。唯一测不出的 gpt seed43 是视频本身的问题：球在第 4 帧后离开画面，失败理由里给出了实测的可追踪帧数，并且照样出了过程图。
每个样本的测量过程图在 `eval_results/minimax_h3/debug/sample_XX/plot.png`，
图上直接画出被测的物理量，可据此复核。

## 3. 留存判定：**Keep**

题目和评测都稳定，指标清楚地把「匀速下落」和「匀加速下落」分开了，正是这道题该做的事。
