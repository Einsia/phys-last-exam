# P8c 评测代码

当前运行入口和输入准备见 [题目 README](../README.md)，结果语义见 [SCORING_V3.md](../../../SCORING_V3.md)。以下命令在题目根目录执行：

```bash
python evaluator/evaluate.py --help
bash scripts/run_eval.sh minimax_h3 --help
```

`p8c_pendulum.py` 是本题物理提取实现，供 `g7/evaluator/evaluate.py` 统一解码和分发使用。公共 CV helper 位于 `v3/shared/g7_cv_common.py`。

`evaluate_raw_legacy.py` 是旧物理入口；`measure_backend.py` 和 `batch.py` 是旧 evaluator_common 接口，源码仓库不附带该历史包。它们不能代替 V3 入口；`utils/` 仅保留旧命名空间占位。
