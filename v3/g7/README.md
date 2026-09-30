# G7 — V3 评测目录

本组题目：[P10](P10/README.md), [P12](P12/README.md), [P27](P27/README.md), [P7](P7/README.md), [P8c](P8c/README.md)。各题使用 `data/metadata.json` 明确关联输入，路径以题目根目录为基准。

从 `v3/` 目录运行，先按 [公共环境说明](../README.md#运行环境) 配置 Python、模型和视频：

```bash
export EVALUATOR_PYTHON="$(command -v python)"
bash g7/scripts/run_eval.sh minimax_h3
```

默认输出位于 `v3/results/g7/<task>/eval_results/<model>/`；分组日志位于 `g7/eval_run_logs/`。可用 `--tasks`、`--samples` 进一步筛选登记样本。当前分数按 [SCORING_V3.md](../SCORING_V3.md) 的一致性 15% / 物理 85% 规则计算。

`data/tasks.json` 保存题目参数；`data/manifest.json`、`data/first_frame_manifest.json` 保存历史批次和首帧清单。`evaluator/` 保留共享解码、分发、合同、物理判定和历史结果工具，各题专属模块归于各自的 `evaluator/`。公共 CV helper 位于 `../shared/g7_cv_common.py`。

`tools/blender_scene_builder.py` 是离线场景生成工具。`scripts/evaluate_v2.py` 实际是当前 V3 批量包装器；旧复算和 legacy 脚本不作为默认运行入口。
