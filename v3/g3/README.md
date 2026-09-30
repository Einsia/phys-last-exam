# G3 — V3 评测目录

本组题目：[P11](P11/README.md), [P3](P3/README.md), [P4](P4/README.md), [P6](P6/README.md), [P9](P9/README.md)。各题使用 `data/metadata.json` 明确关联输入，路径以题目根目录为基准。

从 `v3/` 目录运行，先按 [公共环境说明](../README.md#运行环境) 配置 Python、模型和视频：

```bash
export EVALUATOR_PYTHON="$(command -v python)"
bash g3/scripts/run_eval.sh minimax_h3
```

默认输出位于 `v3/results/g3/<task>/eval_results/<model>/`；分组日志位于 `g3/eval_run_logs/`。可用 `--tasks`、`--samples` 进一步筛选登记样本。当前分数按 [SCORING_V3.md](../SCORING_V3.md) 的一致性 15% / 物理 85% 规则计算。

`data/manifest.json` 是历史包清单，`tools/` 保留题包审计和构建工具。`scripts/evaluate_v2.py` 是文件名保留为 v2 的当前批量包装器，执行 V3 门控与物理流程；不支持旧 `--verify` 或已有分数复算模式。逐题 `evaluator/utils/` 已合并到 `evaluator/`。P3/P9 的坐标缓存及源视频别名需单独恢复。
