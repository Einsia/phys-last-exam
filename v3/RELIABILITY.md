# V3 evaluator reliability workflow

这套扩展把“视频看起来正常”“物理量可测”“物理约束满足”分开记录。它不改变 V3 原有连续分数；新增字段只描述测量证据和判断覆盖率。

## 任务协议

`reliability/protocol.py` 为每个任务冻结适用条件、必须事件、需要测量的量、物理约束和证据不足条件。当前先为 P9、P13、P21、P48 提供完整协议，其余任务使用通用的 `evidence_insufficient` 规则并保留原任务指标。

自动程序遵守一条硬规则：**提取失败不能单独证明事件没有发生。** 只有后端给出 `event_status`、`task_status`、`condition_status` 或明确的事件/条件失败证据，才允许产生 `task_failed`。

## 四种测量状态

每个 V3 结果新增 `measurement_status`，每个指标块新增同名字段：

| 状态 | 含义 |
|---|---|
| `task_failed` | 有明确证据表明事件未完成或实验条件无效 |
| `physics_pass` | 关键量可测，且达到该任务冻结的物理阈值 |
| `physics_fail` | 关键量可测，但违反冻结的物理约束 |
| `evidence_insufficient` | 证据不足，无法支持物理判断 |

旧的 `score_status`（`complete`、`partial`、`unavailable`、`consistency_rejected` 等）继续保留，用于兼容既有排行榜。新增的 `measurement_coverage` 是可可靠测量指标数除以已定义指标数；报告同时给出可判断覆盖率、可判断样本的通过率和全部视频的经验证通过比例。

P21 和 P48 的试运行阈值分别为 0.65 和 0.50。它们来自控制集的第一轮校准：当前提取器在已知正确控制上的分数约为 0.70 和 0.56。控制集很小，因此阈值在独立校准集扩大前标记为 provisional，不把它当成人工真值。

## 真值控制集

生成控制素材：

```bash
python v3/scripts/build_control_videos.py \
  --output v3_evaluator/reliability_controls
```

当前控制集包含 16 个 MP4：

- P9 双摆：正确、5% 周期偏差、25% 周期偏差、深色外观扰动；
- P13 反射：正确、5° 偏差、25° 偏差、深色外观扰动；
- P21 浮冰融化：正确、未融化、未完成、深色外观扰动；
- P48 液滴合并：正确、10% 体积偏差、30% 体积偏差、深色外观扰动。

验证编码和真值：

```bash
python v3/scripts/validate_control_videos.py \
  v3_evaluator/reliability_controls/manifest.json
```

运行现有提取器并生成校准报告：

```bash
python v3/scripts/run_control_extractors.py
python v3/scripts/summarize_control_runs.py
```

结果在 `v3_evaluator/reliability_controls/calibration_report.md`，原始真值在 `manifest.json`，提取器运行记录在 `extractor_runs3/extractor_report.json`。控制视频不进入模型排行榜。

## 迁移已有 V3 结果

不重跑视频即可把四种状态回填到现有结果索引：

```bash
PYTHONPATH=v3 python v3/scripts/backfill_reliability_status.py
```

输出：

- `v3_evaluator/reliability_status/status_augmented_results.json`：逐视频状态；
- `v3_evaluator/reliability_status/status_summary.json`：覆盖率和分组统计；
- `v3_evaluator/reliability_status/status_report.md`：可读报告。

## 真实生成视频的 VLM 对照

`compare_vlm_evaluator.py` 使用现有本地 Qwen3-VL-8B-Instruct 权重，在同一分层样本上分别运行：

1. 直接 VLM：只给任务现象和视频帧，不给出详细公式；
2. 物理规则 VLM：给出适用条件、事件、测量量、物理约束和证据不足规则；
3. V3：读取现有测量结果和新的四状态迁移结果。

样本按 `model × group × score_status` 轮询抽取，默认 240 条，脚本可恢复：

```bash
PYTHONPATH=v3 python v3/scripts/compare_vlm_evaluator.py \
  --count 240 \
  --output-dir v3_evaluator/reliability_vlm \
  --model v3/.models/Qwen3-VL-8B-Instruct \
  --device cuda:0
```

输出 `sample_manifest.json`、逐条追加的 `results.jsonl`、`summary.json` 和 `report.md`。对照指标包括 direct/rule 一致率、分数 Spearman、可测覆盖率、与 V3 状态的一致率和模型排序。VLM 是比较对象，不能替代人工复核或控制真值；报告会明确这一限制。
