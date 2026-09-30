# G1–G9 目录重构说明

标准首帧统一存放在每道题的根目录，已有的首帧生成提示词归入本题的 `prompts/first_frame.txt`。分组和题目级 `data/`、题目级 `reports/`、所有题目级 `first_frames/` 已按要求删除；G5 分组级报告保留。

## 题目目录

```text
Pxx/
├── README.md
├── first_frame.png               # 每题选定的标准首帧
├── prompts/
│   ├── first_frame.txt            # 已有的首帧生成提示词；G3 未提供
│   ├── video.txt                  # 视频续写提示词；G3 使用逐样本文件
│   ├── video_simulation.txt       # 有差异时保留
│   ├── video_legacy.txt           # 有差异时保留
│   └── index.txt                  # 原有批次说明（可选）
├── video_prompts/                 # G3/G7 原有逐样本视频提示词
├── annotations/                  # 首帧几何标注（可选）
├── evaluator/                    # 单题入口、物理后端、配置、依赖和原有测试
├── scripts/                      # 单题运行/标注脚本
├── docs/                         # 题目评测说明（可选）
├── output_videos/<model>/         # 外部输入的运行路径约定
└── eval_results/<model>/          # 历史结果、缓存或显式选择的题目内输出
```

可选目录只在有实际内容时保留。视频、权重和完整评测结果未随源码交付，运行目录不表示已有这些资产。

## 标准首帧的选择和提示词

- G1/G2/G4/G5/G6/G7：原 `first_frames/gpt/gpt_01.png` 移为 `first_frame.png`，同目录 `prompt.txt` 移为 `prompts/first_frame.txt`。
- G3：P3/P4/P6/P9 选择各自的 `Pxx_gpt_01_modern.png`；P11 选择第一张 `P11_gpt_01_30deg.png`。五题均未提供生成提示词原文，本次仅迁移图片。原始生成提示词未随仓库交付，不创建占位文件。
- G8/G9：原 `first_frames/provided/first_frame_01.png` 移为 `first_frame.png`，同目录 `prompt.txt` 移为 `prompts/first_frame.txt`。

40 张标准首帧和 35 份现有生成提示词按原始字节迁移。后续删除了所有题目的 `first_frames/` 目录，其中其余图片、首帧清单、来源记录和仿真生成代码一并删除。根目录 `first_frame.png` 与 `prompts/first_frame.txt` 保留，生成首帧与视频续写的提示词分别保存。

## 公共代码和评测目录

G1/G4/G6 共用 `v3/shared/physeval/`，12 个题目后端均从该处加载。G7 的通用 OpenCV helper 位于 `v3/shared/g7_cv_common.py`；各题物理模块分别位于：

```text
g7/P7/evaluator/p7_rotational.py
g7/P8c/evaluator/p8c_pendulum.py
g7/P10/evaluator/p10_mechanics.py
g7/P12/evaluator/p12_optics.py
g7/P27/evaluator/p27_thermal.py
```

`g7/evaluator/` 保留跨题解码/分发、物理判定、结果合同、历史批量适配和校验工具。`g7/tools/blender_scene_builder.py` 是离线生成器，`g7/scripts/evaluate_v2.py` 是评测包装器。G3 的全局工具在 `g3/tools/`，评测包装器在 `g3/scripts/`；题目原 `evaluator/utils/` 的重复文件已合并。

G8/G9 的几何标注归档到 `annotations/first_frame_annotations.json`，说明文档位于 `docs/`。G2/G5 的人工 ROI 配置仍由 `evaluator/roi.json` 承载。

## 当前运行路径

统一单视频入口在没有登记 metadata 时读取本题 `first_frame.png` 和已有 `prompts/video.txt`，显式 `--image`、`--video_prompt_file` 或 `--video_prompt` 可覆盖。G3 使用逐样本 `video_prompts/`，需要显式选择实际续写提示词；P3/P9 还需要原视频对应的坐标缓存和配置。

G2 的首帧标注脚本和 G2/G5 的后端默认首帧已适配根目录文件；G8/G9 manifest 批量适配器的标准首帧 fallback、外部 all_test 准备脚本中的本地首帧路径也已更新。

原 `run_all_eval.py`、题目/分组 `run_eval.sh` 和入口生成器仍依赖已删除的 metadata 清单，尚需适配新的批量输入方式。G7 物理分发器仍依赖已删除的 `g7/data/tasks.json`。本次首帧迁移没有重建这些清单。

评分语义见 [SCORING_V3.md](SCORING_V3.md)，每组一道典型题目的实际文件树见 [SCHEMA_G1_G9.md](SCHEMA_G1_G9.md)。
