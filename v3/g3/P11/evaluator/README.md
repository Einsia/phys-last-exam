# P11 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`evaluate.py` 是 V3 入口，当前原始物理提取器为 `evaluate_raw_legacy.py`。`measure_backend.py` 保留旧 V2 评分复算接口，不参与 V3 测量。旧 `evaluator/utils/` 已合并到当前目录，同名脚本只保留一份。

物理测量：

- M1：从时序亮度差分提取光束，独立拟合液面，测量光线与法线夹角，计算 sin(theta_i)/sin(theta_t)-n_water。
- M2：计算入射光、折射光与液面交点的距离差，以冻结的容器宽度归一化。

配置文件冻结几何、阈值、初始化种子及权重；审计、连续重评分、历史批处理和合成回归文件仍保留在本目录。历史批处理输出格式与 V3 不同，运行 V3 使用题目 `scripts/run_eval.sh`。
