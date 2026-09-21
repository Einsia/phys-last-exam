# G8/G9 评测与输出规范（2026-09-11）

> 本文说明历史 V2 评分。V3 当前评分为一致性 15% + 物理 85%，默认通过阈值 0.3；请参见 [SCORING_V3.md](SCORING_V3.md)。

版本：`physical-bench-proxy-v2-recognition015-arithmetic-g8g9-defined-metrics`。

按用户提供的 G3/G7 V2 规则扩展到 G8/G9。原始物理测量、纯物理分、识别分和最终分分别保存。G3/G7 各题的参数不套用到其他题。

## 指标与尺度

| 题目 | `verbose.M1.raw_metric` 原始物理量 | 纯物理分 q | 固定参数来源 |
| --- | --- | --- | --- |
| P34 | 两个有符号偏转、偏转和、反向标记 | 原反向抵消分 × 最小运动分 | 保留原 `min_deflection_deg=5`，无新增残差尺度 |
| P37 | 开环/闭环高度比及各自归一化高度 | `clip((1-height_ratio)/0.2,0,1)` | 原 `margin=0.2` |
| P38 | 两板完整周期数及次数比 | `N_slotted/(N_solid+N_slotted)`；开槽板为 0 次时为 0 | 保留原次数映射；相同正次数得 0.5 |
| P39 | 时序匹配 F1 | F1 | 原事件时间容差 0.5 s；它不是残差尺度 a |
| P41 | `abs(beta_water-0.5)+abs(beta_sand)` | `1/(1+abs(e)/1.0)` | 原 `error_half_score=1.0` |
| P43 | `N_COM_cross-N_tip`，单位为帧 | 对同一对事件的秒差使用 `1/(1+abs(delta_t)/0.5)` | 原 `time_error_half_score_sec=0.5`；不把帧数误当秒 |
| P47 | 同帧三半径误差的 PTS 加权中位数 | `1/(1+abs(e)/1.0)` | 原题目文档中的 `error_at_zero=1.0`；先前没有可运行实现 |
| P49 | `abs((v1/v2)/(r1/r2)^2-1)` | `1/(1+abs(e)/1.0)` | 保留原 `error_at_zero=1.0` 的数值，改作半分尺度 |

`error_at_zero` 为兼容旧 CLI 保留的参数名，V2 中不再代表截零边界。全部参数固定；未按本批视频调参，未做人类分数校准。有限残差连续扣分，不设置硬截断。P34/P37/P38/P39 保留原非残差映射。

## 每项识别分及总分

可可靠测量：`verbose.M*.scoring.physics_score=q`，`recognition_score=0.15`，
`proxy_score=0.15+0.85*q`。公开 `metrics.M*.metric` 保存该 0–1 最终分（包含识别奖励）；
原始量和分层评分在 verbose 中保留，`metrics.M*` 仍只有 `extract_success` 和 `metric` 两个字段。

不可测：`extract_success=false`，`metric=null`，`physics_score=null`，识别分和最终分均为 0，`proxy_valid=false`。检测到物体不能代替完成物理量测量；可测的错误物理行为仍获得识别分。

G8/G9 现有任务 evaluator 文档明确仅实现 M1。本次不从其他题目的 M2 定义推造辅助指标。M2 保持 `extract_success=null`、`metric=null`；`verbose.M2` 展示未定义原因，其 `scoring` 中的分数值均为 null。

按用户 2026-09-11 确认，**总分只对已定义指标等权平均**。G8/G9 当前仅定义 M1，因此
`verbose.M1._scoring_summary.weights={"M1":1.0,"M2":0.0}`，总分等于 M1 最终分，最高总分为 1。
`defined_metrics` 和 `not_applicable_metrics` 在 `_scoring_summary` 中明确区分未设置与提取失败。

已定义但不可测的指标仍计入分母并贡献 0。例如同时定义 M1/M2 而 M2 提取失败时，总分仍为 `0.5*M1.proxy_score+0.5*0`；不会丢弃失败项补分。

`score_status` 和 `overall_proxy_valid` 按**已定义指标**判断：全部已定义指标可测为 complete/true；部分可测为 partial/false；无可测指标为 unavailable/false。当前 M1 可测且 M2 未设置的样本为 complete。任何复合指标先完成内部物理分汇总，再加一次识别分。

本次只更新总分聚合、覆盖状态和对应报告；原始量、纯物理分、识别分及分项最终分保持不变。上版聚合记录保存在 `aggregation_history`，修改前文件备份于 `work/g8_g9_defined_metrics_20260911/before/`。

## JSON 与测量证据

每个结果包含 `task_id`、`video_path`、`image_path`、实际 `video_prompt`、`model`、`seed`、`metrics.M1/M2` 和 `verbose.M1/M2`。未知 seed 为 null。

公开结果的 `video_path` 使用 `minimax_h3/videos/sample_00.mp4`，`image_path` 使用
`first_frames/provided/first_frame_01.png`；实际磁盘输入仍由 `metadata_v2.json` 与
`verbose.M1._provenance` 追溯。首帧生成器没有可靠记录，因此放在 `first_frames/provided/`，不冒称 GPT、仿真或实拍来源。

`score_details` 和原提取阶段 `calculation.md` 中的旧 `score/metric` 表述对应纯物理分 q；加入识别分及已定义指标聚合后的数值以 `verbose.M*.scoring`、`verbose.M1._scoring_summary` 和 `scoring_calculation.md` 为准。

`verbose.M1` 包含测量原理、过程、物理量、窗口、阈值、失败原因、运行方式、曲线和图像/视频证据引用。`raw_measurement` 保存提取器测量；`normalization_source.previous_metrics` 保存本次归一化前提取器的指标字段。历史交付结果另完整备份于 `work/g8_g9_v2_20260910/before/`，对应哈希见 `input_manifest.json`。

P34/P37/P38/P39/P41/P43/P49 在核验视频、标注、模型及掩码哈希后复用既有 SAM/CoTracker 缓存，重新解码、测量、评分和导出可视化。`extraction_run` 记录缓存使用情况。没有声称本次重新运行了缓存中的神经网络推理，也没有声称识别准确率提高。

P47 新增 OpenCV/SciPy 轮廓测量后端。圆模型独立拟合，不使用 Laplace 关系约束曲率；分割产物为实际观测边缘掩码。可可靠辨认的反向弯曲正常计分；可证明平直的隔膜以明确的发散极限计 q=0，不能把普通 NaN/Inf 当作这一极限。存在两条可解析的膜面边界、投影不明确或曲率精度不足时返回 false/null，并保存实际边缘和候选拟合。此后端未实现完整的任意视角三维恢复。

## 运行

从 v2 目录批量重跑全部八题：

```bash
.venv/bin/python scripts/run_g8_g9_eval.py --model minimax_h3 --device cpu --reuse_masks --reuse_tracks
```

每题入口：`bash g8/P37/scripts/run_eval.sh minimax_h3 --device cpu --reuse_masks --reuse_tracks`。P34 只支持 `--reuse_masks`；P47 为逐帧轮廓测量。总入口处理这两个差异。

默认优先读 `output_videos/<model>` 与 `metadata_v2.json`，显式 `--input_dir` 保留原目录输入功能。新增视频必须在对应 manifest 登记首帧、prompt、样本 ID；六个使用审核标注的旧后端还需提供匹配的 `annotation`，不能把当前视频标注用于另一个视频。

依赖见各题 `evaluator/requirements.txt`。运行环境由 `EVALUATOR_PYTHON` / `P34_PYTHON` 指定；可用 `EVALUATOR_PYTHON` / `P34_PYTHON` 指定环境。本地 DINO/SAM 权重位于 `g8/P34/models/`；CoTracker 默认寻找 v2 的 `shared_models/`、`shared_vendor/`，其次使用原项目的同名资源。也可显式传 `--cotracker_checkpoint` 并安装依赖中的官方 CoTracker 包。没有下载、生成或替换源视频。

测试与全量验证：

```bash
.venv/bin/python -m unittest discover -s tests/refined -v
.venv/bin/python -m unittest discover -s g8/P34/evaluator/tests -v
.venv/bin/python scripts/validate_g8_g9.py
```
