# P34 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`evaluate.py` 是 V3 入口，`measure_backend.py` 是门控通过后的物理后端，`batch.py` 是调用本题公开入口的 manifest 批量适配器；当前文档统一使用 `scripts/run_eval.sh`。

P34 的分割、几何测量、模型加载、媒体和诊断实现在 `utils/`，模型位置与阈值参数可查看 `measure_backend.py --help`。
