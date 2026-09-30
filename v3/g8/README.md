# G8 — V3 评测目录

本组题目：[P34](P34/README.md), [P37](P37/README.md), [P38](P38/README.md), [P39](P39/README.md), [P41](P41/README.md)。各题使用 `data/metadata.json` 明确关联输入，路径以题目根目录为基准。

从 `v3/` 目录运行，先按 [公共环境说明](../README.md#运行环境) 配置 Python、模型和视频：

```bash
export EVALUATOR_PYTHON="$(command -v python)"
bash g8/scripts/run_eval.sh minimax_h3
```

默认输出位于 `v3/results/g8/<task>/eval_results/<model>/`；分组日志位于 `g8/eval_run_logs/`。可用 `--tasks`、`--samples` 进一步筛选登记样本。当前分数按 [SCORING_V3.md](../SCORING_V3.md) 的一致性 15% / 物理 85% 规则计算。

评测说明保存在每题 `docs/evaluator.md`，首帧标注保存在 `annotations/`（仅有标注的题目）。这些题的公开结果只定义 M1，M2 保留 null；所需分割/跟踪模型和视频缓存见题目 README。历史 V2 规则只用于解释旧结果。
