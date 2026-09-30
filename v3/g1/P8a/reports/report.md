# P8a 结论　—　同摆长 5° vs 15° 小角等时性

8 个 minimax-h3 视频（gpt 与仿真两路首帧，每路 seed 42/43/44/45），平均分 **0.78**，其中 8 个达到判定线（M1 得分 ≥ 0.50）。

| Sample | route/seed | extract_success | M1 | 得分 | 备注 |
|---|---|---|---|---|---|
| sample_00 | gpt seed42 | ✅ | +0.0002 | 0.88 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_01 | gpt seed43 | ✅ | -0.1460 | 0.64 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_02 | gpt seed44 | ✅ | -0.0043 | 0.94 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_03 | gpt seed45 | ✅ | -0.0594 | 0.68 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_04 | sim seed42 | ✅ | +0.0113 | 0.90 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_05 | sim seed43 | ✅ | -0.0948 | 0.61 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_06 | sim seed44 | ✅ | -0.0270 | 0.78 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |
| sample_07 | sim seed45 | ✅ | -0.0363 | 0.77 | the two amplitudes are reported alongside: the point of the task is that the ratio stays at 1 even though they differ |

## 1. MiniMax H3 表现

小角等时性大体成立，8 个里 7 个两摆周期比偏差在 10% 以内，seed42 只差 0.02%。剩下的偏差主要来自摆动幅度衰减导致的周期漂移。

## 2. 自动评测脚本可靠性

8 个样本中 8 个可自动测出 M1。自动结果与逐帧读图的人工判断一致。
每个样本的测量过程图在 `eval_results/minimax_h3/debug/sample_XX/plot.png`，
图上直接画出被测的物理量，可据此复核。

## 3. 留存判定：**Keep**

续写 prompt 已重写（原版有一半视频左摆全程不动），重出后 8/8 可测。
