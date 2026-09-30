# P6 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`evaluate.py` 是 V3 入口，当前原始物理提取器为 `evaluate_raw_legacy.py`。`measure_backend.py` 保留旧 V2 评分复算接口，不参与 V3 测量。旧 `evaluator/utils/` 已合并到当前目录，同名脚本只保留一份。

物理测量：

- M1：测量球心累计位移、标记转角和球半径，计算积分形式的 v/(omega*R)-1。
- M2：由球心速度与 omega*R 的差估计瞬时接触点速度残差。

配置文件冻结几何、阈值、初始化种子及权重；审计、连续重评分、历史批处理和合成回归文件仍保留在本目录。历史批处理输出格式与 V3 不同，运行 V3 使用题目 `scripts/run_eval.sh`。
