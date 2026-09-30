# G1–G9 统一评测输出

> 本文说明历史 V2 结果。V3 当前入口已采用一致性门控；运行方式和输出语义见 [README.md](README.md) 与 [SCORING_V3.md](SCORING_V3.md)。

每题的 `evaluator/evaluate.py` 是单视频入口，`scripts/run_eval.sh` 是批量入口。
结果保存在 `<组>/<题>/eval_results/<模型>/result_sample_xx.json`。
公开 `video_path` 使用 `<model>/videos/sample_xx.mp4`，公开 `image_path` 使用
`first_frames/...`；实际磁盘路径和输入哈希仍由 `data/metadata.json` 与 `verbose.M1._provenance` 追溯。

```bash
# 在 v2 目录执行，重跑全部已登记视频。
.venv/bin/python scripts/run_all_eval.py --workers 3
# 全量批次结束后生成验证记录和报告。
.venv/bin/python scripts/complete_all_eval.py
# 按组或按题运行。
bash g1/scripts/run_eval.sh minimax_h3
bash g1/P1/scripts/run_eval.sh minimax_h3
# 单视频；清单中的元数据会自动关联。
.venv/bin/python g1/P1/evaluator/evaluate.py \
  --video g1/P1/output_videos/minimax_h3/sample_00.mp4 \
  --output g1/P1/eval_results/minimax_h3/result_sample_00.json
```

## 字段和评分

公开 `result_sample_xx.json` 只有以下八个顶层字段：
`task_id`、`video_path`、`image_path`、`video_prompt`、`model`、`seed`、`metrics`、`verbose`。
`metrics.M1/M2` 严格只有 `extract_success` 和 `metric` 两个字段。`extract_success=false`
或指标未定义时，`metric` 必须为 null；未记录的模型或种子保留 null，不从样本编号推断。

`metrics.M1/M2.metric` 是 **0–1 的最终指标分，包含识别奖励**，等于该项的 `proxy_score`；
可测时必须是有限数值。原始指标移至 `verbose.M1/M2.raw_metric`，完整测量与诊断保留在
`raw_measurement`；它们可以是数值、布尔判断或多个子项组成的对象，不限制在 0–1。
`verbose.M1/M2.scoring` 分别保存纯物理分、识别分、最终分及公式；总分和覆盖状态位于
`verbose.M1._scoring_summary`，其中 `metric_semantics=normalized_final_score` 明确公开字段含义。
残差的纯物理分为 `1/(1+abs(e)/a)`，固定尺度 a 沿用原实现；方向、比例、覆盖率沿用各题原映射。
可可靠测量的指标最终分为 `0.15+0.85*physics_score`。

`verbose.M1/M2.scoring.proxy_formula` 展示分项公式及可测、不可测、未定义三种情况，
`proxy_calculation` 展示该样本的实际数值代入；总分的实际数值代入位于
`verbose.M1._scoring_summary.calculation`，其中 `formula` 的 `S_M1/S_M2` 分别指分项 `proxy_score`。
识别奖励本身为 0.15，在每项纯物理分汇总后加一次。

| 指标状态 | extract_success | metric / physics_score | recognition_score / proxy_score |
|---|---|---|---|
| 可测 | true | 最终指标分 / 纯物理分 | 0.15 / 最终分 |
| 已定义但不可测 | false | null / null | 0 / 0 |
| 题目未定义 | null | null / null | null / null |

例如 P1/sample_00 的 M1：原始误差为 `1.0173066326807456`，固定尺度 `a=0.1`，
纯物理分为 `0.08950094546567824`，公开 `metric=0.2260758036458265`。
归一化与识别奖励只计算一次；重新整理 JSON 时从 `raw_metric` 恢复原始量，不对最终分再次归一化。

总分只对**已定义**指标等权计算。M1、M2 均定义时，总分为各自最终分的固定等权平均；
一项提取失败仍保留其权重，贡献为零。题目只定义 M1 时，总分等于 M1 最终分，M2 保留 null。
这一规则适用于 G8/G9，也适用于没有辅助指标的 P5。

多个辅助量合并为 M2 的内部子项。所需子项缺失会使 M2 不可测，不丢弃缺失项来补分；
内部几何平均、方向和覆盖率保持原定义，识别分在 M2 汇总后加一次。
P12 仍采用已测光线残差与覆盖率的特殊组合。

`_scoring_summary.weights` 明确列出权重，`_scoring_summary.score` 是总分。
`complete/partial/unavailable` 表示已定义指标的测量覆盖情况，不是物理合格判定。
`overall_proxy_valid=false` 不会把部分可测的视频总分抹掉。

## 可复核的过程

每个已定义指标的 `verbose` 包含测量原理、步骤、原始量、归一化细节和 `evidence` 文件清单。
原始后端结果另存于 `raw_result.json`，保留数值、跟踪与有效性诊断。
图像或视频证据展示实际测量几何；无法测量时明确记录失败原因。
`scoring_calculation.md` 展示分项与总分计算。

`provenance` 记录运行时间、输入哈希、后端命令、缓存情况及运行错误。
G3 的 P3/P9 使用原视频关联的坐标缓存重新计算物理量并重新生成证据；P3 额外核对原缓存的
视频和配置哈希，P9 核对原视频目录关联与冻结首帧种子，并记录本次哈希。
G8/G9 使用其已有缓存校验规则，P47 重新运行逐帧轮廓测量。
这些运行不能解读为重新执行了全部神经网络推理。

G3 的 P4 使用原有 `--no-cotracker` CPU 选项重新运行 color/bgsub 两个后端，阈值不变。
其他需要 SAM2 容器分割的旧提取器可在 CPU 上使用原检查点。
G1–G7 使用本地 `.runtime/opencv4` 中的 OpenCV 4.11 二进制依赖，避免 OpenCV 5 返回数组形状变化；
G8/G9 保留此前验证的运行环境。主运行环境为 Python 3.12。

历史结果和代码位于 `work/g1_g9_v2_20260911/before`，输入与备份哈希见同目录 `input_manifest.json`。
历史文件中的旧分数、旧路径和旧覆盖状态只供追溯，当前消费者应读取标准 `result_sample_xx.json`。
