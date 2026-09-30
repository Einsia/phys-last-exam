# P9 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`evaluate.py` 是 V3 入口，当前原始物理提取器为 `evaluate_raw_legacy.py`。`measure_backend.py` 保留旧 V2 评分复算接口，不参与 V3 测量。旧 `evaluator/utils/` 已合并到当前目录，同名脚本只保留一份。

物理测量：

- M1：跟踪两个摆锤与悬点，以峰值间隔估计周期，测量摆长，计算 abs((T1/T2)^2/(L1/L2)-1)。
- M2：分别计算短摆和长摆周期的变异系数，再汇总纯物理分。

配置文件冻结几何、阈值、初始化种子及权重；审计、连续重评分、历史批处理和合成回归文件仍保留在本目录。历史批处理输出格式与 V3 不同，运行 V3 使用题目 `scripts/run_eval.sh`。

当前 V3 还需要与 source_video_path 匹配的坐标缓存及外部运行资产，具体见题目 README。
