# G1–G9 目录重构说明

本次重构把“题目资产、题目评测器、全局运行时、历史资料”分开。题目元数据中的路径仍以题目根目录为基准解析，所以把 `metadata.json` 放进 `data/` 不会改变 `video_path`、`image_path` 等字段的语义。

## 统一题目树

```text
Pxx/
├── README.md
├── data/
│   ├── metadata.json             # 唯一正式样本清单（v2 内容为准）
│   └── samples.csv              # 仅 G1/G4/G6 的旧样本/route 索引
├── prompts/
│   ├── video.txt                 # 正式视频续写 prompt
│   ├── video_simulation.txt      # 有区别的 simulation prompt（可选）
│   └── index.txt                 # 仅在题目根 prompt 是索引说明时使用
├── first_frames/
│   ├── gpt/
│   ├── simulation/
│   └── provided/                 # 外部提供、无法归入生成器的首帧
├── annotations/                  # 审核过的首帧标注（可选）
├── evaluator/                    # 单题评测入口、后端、题目配置和测试
├── scripts/                      # 题目级运行/标注脚本
├── reports/                      # 历史报告（可选）
├── docs/                         # 题目评测说明（可选）
├── output_videos/<model>/        # 外部生成输入
└── eval_results/<model>/         # 历史结果或显式选定的题目内输出
```

`README.md`、`evaluator/`、`scripts/`、`output_videos/` 和 `eval_results/` 是所有题目都可以使用的固定位置；没有对应内容的可选目录不创建。

## 公共实现和元数据

- G1、G4、G6 原先每题各自复制的 `evaluator/utils/physeval` 已合并到 `v3/shared/physeval/`。它包含视频解码、轨迹/拟合、液体/摆、任务注册、结果 schema、可视化和 task evaluator。所有 12 个 `measure_backend.py` 已改为导入 `shared.physeval`。P1 中对不完整测量诊断的修正版作为公共版本保留。
- G7 的通用 OpenCV helper 已移到 `v3/shared/g7_cv_common.py`；P7/P8c 的题目模块从这里导入。
- 每题的 `metadata_v2.json` 已成为 `data/metadata.json`。同时存在旧 `metadata.json` 的 G7/G8/G9 题目保留旧对象在顶层 `legacy_metadata`，正式读取只看 `samples`。旧路径不参与评测。
- G8/G9 的 annotation 路径已改为 `annotations/first_frame_annotations.json`，运行时仍按题目根目录解析。

## 文件归档结果

- G1/G4/G6：根目录 `prompt.txt` → `prompts/video.txt`；有区别的 `prompt_sim.txt` → `prompts/video_simulation.txt`；`report.md` → `reports/report.md`；`samples.csv` → `data/samples.csv`。
- G2/G5/G7/G8/G9：根目录 prompt 和重复的 `video.txt` 合并为 `prompts/video.txt`。G7 P10/P12/P27 中不相同的旧 `video.txt` 作为 `prompts/video_legacy.txt` 保留。重复的根目录首帧和 `first_frame.txt` 删除；唯一的 G2 P19/P40/P42 根首帧归档到 `first_frames/provided/`；G5 P36/P44 的 simulation 首帧归档到 `first_frames/simulation/`。
- G8/G9 的 `first_frame_annotations.json` → `annotations/first_frame_annotations.json`；`evaluator.md` → `docs/evaluator.md`。
- G3 P3/P4/P6/P9 根 `prompt.txt` 与 `video_prompts/` 重复，已删除；P11 的根 prompt 是批次索引说明，保留为 `prompts/index.txt`。每题 `evaluator/utils/` 中与 `evaluator/` 相同的文件已删除，缺失或不同的 README 已合并到 `evaluator/README.md`。
- G3 的全局 `evaluate_v2.py` → `scripts/evaluate_v2.py`，全局 `manifest.json` → `data/manifest.json`；`tools/` 的审计和构建脚本保持在全局工具目录。

## G7 全局和题目评测器

`g7/evaluator/` 下保留的是跨题运行时：`contract.py`（公开结果合同）、`evaluate.py`（解码/分发）、`physics.py`（物理判定）、`evaluate_one.py`/`run_batch.py`（单样本/批量适配）、`task_entrypoint.py`、`result_schema.py`、`renormalize_results.py`、`validate_results.py`、`validate_schema.py`、`test_proxy.py` 和依赖文件。它们不属于某一道题，不拆到题目目录。

原 `g7/evaluator/tasks/` 中的题目脚本已拆到：

```text
g7/P7/evaluator/p7_rotational.py
g7/P8c/evaluator/p8c_pendulum.py
g7/P10/evaluator/p10_mechanics.py
g7/P12/evaluator/p12_optics.py
g7/P27/evaluator/p27_thermal.py
```

全局 `blender_scene_builder.py` 是离线场景/输入生成器，归档到 `g7/tools/`；`evaluate_v2.py` 是统一评分 wrapper，归档到 `g7/scripts/`；`first_frame_manifest.json`、`manifest.json`、`tasks.json` 分别是首帧资产清单、批次样本清单、题目参数清单，归档到 `g7/data/`。全局运行时代码已改用这些新路径。

## 入口和路径兼容

`v3/scripts/run_all_eval.py`、统一 runtime、G8/P34 batch、G1–G9 准备脚本都读取 `Pxx/data/metadata.json`。G1/G4/G6 后端读取 `data/samples.csv` 和 `prompts/`；G2/G5 后端的首帧 fallback 读取 `first_frames/`；G7 运行时读取 `g7/data/` 并从题目 evaluator 模块分发。没有视频文件的源码包仍可检查目录和元数据，实际评测需提供 metadata 登记的外部 MP4。

当前 V3 批量入口默认把新结果写入 `v3/results/<group>/<task>/eval_results/<model>/`。题目内的 `eval_results/` 可保留历史测量、坐标缓存及运行日志，不表示默认新结果会写回该目录。逐题 README 已统一说明工作目录、解释器、样本关联与 V3 参数；文件级概览见 [SCHEMA_G1_G9.md](SCHEMA_G1_G9.md)。
