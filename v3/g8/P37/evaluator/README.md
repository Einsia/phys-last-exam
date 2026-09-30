# P37 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`evaluate.py` 是 V3 入口，`measure_backend.py` 是门控通过后的物理后端，`batch.py` 是调用本题公开入口的 manifest 批量适配器；当前文档统一使用 `scripts/run_eval.sh`。

公共提取实现在 `v3/refined_evaluators/`；`utils/` 仅保留命名空间占位。参数、默认模型位置及阈值以物理后端的 `--help` 为准。
