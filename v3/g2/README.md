# G2 — V3 评测目录

本组题目：[P13](P13/README.md), [P19](P19/README.md), [P28](P28/README.md), [P40](P40/README.md), [P42](P42/README.md)。各题使用 `data/metadata.json` 明确关联输入，路径以题目根目录为基准。

从 `v3/` 目录运行，先按 [公共环境说明](../README.md#运行环境) 配置 Python、模型和视频：

```bash
export EVALUATOR_PYTHON="$(command -v python)"
bash g2/scripts/run_eval.sh minimax_h3
```

默认输出位于 `v3/results/g2/<task>/eval_results/<model>/`；分组日志位于 `g2/eval_run_logs/`。可用 `--tasks`、`--samples` 进一步筛选登记样本。当前分数按 [SCORING_V3.md](../SCORING_V3.md) 的一致性 15% / 物理 85% 规则计算。
