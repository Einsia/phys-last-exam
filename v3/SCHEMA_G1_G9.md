# VDM-Bench V3：G1–G9 新目录与代码文件 Schema（README 已核对）

本文件记录 data/reports 删除、标准首帧迁移及题目 first_frames 目录清理后的目录体系。核对日期：2026-10-01；本次清理基于 Git 版本 7619d43。

每个分组仅展开一道典型题目，同时完整保留该分组的 scripts、evaluator、tools、分组 reports 等公共目录。下面的目录树根据当前本地文件直接生成，叶子节点是实际文件；其他题目仅列题号。MP4、权重、坐标缓存和生成结果未随源码交付，另列运行路径约定。

文件归属和调用关系以代码为依据：文件名中的 v2/legacy 不能单独说明它是否处于当前执行链，实际调用以 unified_evaluators/runtime.py 为准。

## 公共路径契约与文件分类

| 分类 | 固定位置 | 用途 |
| --- | --- | --- |
| 标准首帧 | gX/Pxx/first_frame.png | 所有 40 题均有选定首帧。G1–G7 选第一张 GPT 图片，G8/G9 选原 provided 首帧。 |
| 首帧生成提示词 | Pxx/prompts/first_frame.txt | 35 题已有原文并已迁移；G3 五题未提供，不创建占位文件。 |
| 视频续写提示词 | Pxx/prompts/video.txt、video_simulation.txt、video_legacy.txt；G3/G7 的 video_prompts/ | 题目文本和逐样本文本分别保留，按输入视频实际来源选择。 |
| 首帧标注 | Pxx/annotations/first_frame_annotations.json；evaluator/roi.json | 审核过的几何标注或人工 ROI 配置。 |
| 单题代码 | Pxx/evaluator/ | 当前入口、物理后端、题目配置、依赖和原有辅助脚本/测试。 |
| 运行脚本 | Pxx/scripts/、gX/scripts/、v3/scripts/ | 单题/分组/公共脚本；原 metadata 批量方式尚需适配删除清单后的输入。 |
| 说明与分组报告 | Pxx/docs/；g5/reports/ | 说明文档与分组级历史报告。题目级 reports/first_frames 和全部分组/题目 data 已删除。 |
| 公共测量代码 | v3/shared/、unified_evaluators/、refined_evaluators/ | 公共测量、统一门控和结果合同。 |

统一单视频入口在没有 metadata 清单时读取根目录标准首帧及已有视频续写提示词，支持显式覆盖。G3/P3/P9 的缓存及 G8/G9 的标注/模型仍需准备；G7 分发器的旧 tasks.json 依赖和原批量输入方式尚需适配。当前总分仍为一致性 15% 与纯物理分 85%。

## V3 级共享代码树

```text
v3/
├── refined_evaluators/
│   ├── __init__.py
│   ├── batch.py
│   ├── bubbles.py
│   ├── common.py
│   ├── definitions.py
│   ├── export.py
│   ├── media.py
│   ├── numeric.py
│   ├── output.py
│   ├── resources.py
│   ├── runtime.py
│   ├── scoring.py
│   ├── tasks.py
│   └── vision.py
├── reliability/
│   ├── __init__.py
│   ├── protocol.py
│   ├── status.py
│   └── vlm_compare.py
├── scripts/
│   ├── all_test_common.py
│   ├── audit_all_test.py
│   ├── backfill_reliability_status.py
│   ├── build_control_videos.py
│   ├── compare_vlm_evaluator.py
│   ├── finalize_all_eval.py
│   ├── install_unified_entries.py
│   ├── prepare_all_test_annotations.py
│   ├── prepare_all_test_g3.py
│   ├── prepare_g1_g9.py
│   ├── report_all_test.py
│   ├── run_all_eval.py
│   ├── run_all_test.py
│   ├── run_control_extractors.py
│   ├── smoke_consistency.py
│   ├── summarize_control_runs.py
│   └── validate_control_videos.py
├── shared/
│   ├── physeval/
│   │   ├── tasks/
│   │   │   ├── __init__.py
│   │   │   ├── p1.py
│   │   │   ├── p14.py
│   │   │   ├── p16.py
│   │   │   ├── p18.py
│   │   │   ├── p2.py
│   │   │   ├── p20.py
│   │   │   ├── p21.py
│   │   │   ├── p45.py
│   │   │   ├── p48.py
│   │   │   ├── p5.py
│   │   │   ├── p8a.py
│   │   │   └── p8b.py
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── fitting.py
│   │   ├── liquid.py
│   │   ├── pendulum.py
│   │   ├── registry.py
│   │   ├── schema.py
│   │   ├── track.py
│   │   ├── video.py
│   │   └── viz.py
│   ├── __init__.py
│   └── g7_cv_common.py
├── tests/
│   ├── refined/
│   │   ├── test_bubbles.py
│   │   ├── test_integration_contract.py
│   │   ├── test_numeric.py
│   │   └── test_scoring_v2.py
│   ├── __init__.py
│   ├── test_batch_schema_paths.py
│   ├── test_consistency_v3.py
│   ├── test_reliability_status.py
│   └── test_unified_contract.py
├── unified_evaluators/
│   ├── __init__.py
│   ├── consistency.py
│   ├── contract.py
│   ├── execute_backend.py
│   ├── physics.py
│   └── runtime.py
├── OUTPUT_FORMAT_V2.md
├── README.md
├── RELIABILITY.md
├── RESTRUCTURE_NOTES.md
├── SCHEMA_G1_G9.md
├── SCORING_V2.md
└── SCORING_V3.md
```

| 文件或目录 | 职责 |
| --- | --- |
| unified_evaluators/runtime.py | 40 个题目共用的输入加载、门控、后端调度、调试和发布流程；支持根目录标准首帧。 |
| unified_evaluators/consistency.py | 本地/HTTP VLM 一致性判断、帧采样、参数和模型接口。 |
| unified_evaluators/contract.py、physics.py | 统一结果结构与 G1–G9 物理测量映射；G3/G7 的指标转换在 physics.py。 |
| unified_evaluators/execute_backend.py | 物理子进程环境与 OpenCV 兼容包装。 |
| shared/physeval/context.py、schema.py、registry.py | 评测上下文、原始测量结果和任务注册；共享任务规则位于 tasks/。 |
| shared/physeval/video.py、track.py、fitting.py、pendulum.py、liquid.py、viz.py | 视频解码、目标轨迹、拟合、摆/液面分析和可视化。 |
| shared/g7_cv_common.py | G7 公共图像检测和几何辅助，不属于某一题的资产。 |
| refined_evaluators/ | G8/G9 共用物理提取、数值计算、模型资源、媒体导出、评分、manifest 批处理。P34 还保留自己的 evaluator/utils 实现。 |
| reliability/ | 测量状态分类、控制样本协议与 VLM 对照工具。 |
| scripts/run_all_eval.py、install_unified_entries.py | 原登记样本批量执行和入口生成工具；仍依赖已删除的 metadata，尚需适配。 |
| scripts/ 中其他文件 | 控制视频构建、可靠性校准、准备、审计和报告辅助工具；不是题目单视频入口。 |
| tests/ | 公共回归测试；包含原 metadata 目录解析、评分、一致性及可靠性检查。单题原有测试仍保留在 evaluator/。 |

## G1：共享轨迹与物理测量库；典型题 P1 — 钢球自由下落

本组题目：P1, P2, P5, P8a。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P1，并保留本组全部公共文件。

```text
g1/
├── P1/
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P1/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P1/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前单题入口 | P1/evaluator/evaluate.py | 调用 unified_evaluators/runtime.py，执行 V3 门控与统一输出。 |
| 物理适配 | P1/evaluator/measure_backend.py | 构建 Context；读取 prompts/video.txt，调用 shared.physeval 的 P1 规则。 |
| 输入提示词 | P1/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 依赖 | P1/evaluator/requirements.txt | 本题物理测量依赖；历史题目报告已删除。 |
| 单题/整组运行 | P1/scripts/run_eval.sh、scripts/run_eval.sh | 各自调用公共 run_all_eval.py，选择 P1 或 G1。 |

其余 P2/P5/P8a 使用相同公共库；辅助量按题目规则不同。P5 的公开 M2 未定义。

## G2：题目独立 OpenCV 后端与首帧生成代码；典型题 P19 — 连通器液面

本组题目：P13, P19, P28, P40, P42。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P19，并保留本组全部公共文件。

```text
g2/
├── P19/
│   ├── evaluator/
│   │   ├── utils/
│   │   │   └── README.md
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   ├── annotate_roi.py
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P19/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P19/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前评测代码 | P19/evaluator/evaluate.py、measure_backend.py | 前者是 V3 门控入口，后者测量左右液面及末段运动。 |
| ROI 标注 | P19/scripts/annotate_roi.py | 直接读取根目录 first_frame.png 或显式指定图片；生成 evaluator/roi.json。run_eval.sh 不支持旧 --annotate 模式。 |
| 输入提示词 | P19/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 运行脚本 | P19/scripts/run_eval.sh、scripts/run_eval.sh | 单题/整组的 V3 批量入口。 |

P13/P28 根首帧因与 first_frames 重复已去除；P19/P40/P42 的独立根首帧归到 provided/。本组每题保留自己的 OpenCV 测量后端。

## G3：原始物理提取、冻结配置与历史审计文件合并；典型题 P3 — 30° 与 60° 斜抛

本组题目：P11, P3, P4, P6, P9。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P3，并保留本组全部公共文件。

```text
g3/
├── P3/
│   ├── evaluator/
│   │   ├── audit_formal_artifacts.py
│   │   ├── config.yaml
│   │   ├── config_v2_geometric.json
│   │   ├── evaluate.py
│   │   ├── evaluate_raw_legacy.py
│   │   ├── measure_backend.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── rescore_continuous.py
│   │   ├── rescore_v2_geometric.py
│   │   ├── run_batch.py
│   │   ├── run_batch_pre_continuous_aggregate_20260826.py
│   │   ├── synthetic_validation.json
│   │   ├── test_rescore_continuous.py
│   │   ├── track_worker_p3.py
│   │   ├── validate_synthetic.py
│   │   └── write_human_sanity_review.py
│   ├── scripts/
│   │   ├── run_eval.sh
│   │   └── run_eval_legacy.sh
│   ├── video_prompts/
│   │   ├── P3__formal_v1__P3_gpt_01_modern_seed42.txt
│   │   ├── P3__formal_v1__P3_gpt_01_modern_seed43.txt
│   │   ├── P3__formal_v1__P3_gpt_01_modern_seed44.txt
│   │   ├── P3__formal_v1__P3_gpt_01_modern_seed45.txt
│   │   ├── P3__formal_v1__P3_gpt_02_classic_seed42.txt
│   │   ├── P3__formal_v1__P3_gpt_02_classic_seed43.txt
│   │   ├── P3__formal_v1__P3_gpt_02_classic_seed44.txt
│   │   ├── P3__formal_v1__P3_gpt_02_classic_seed45.txt
│   │   ├── P3__formal_v1__P3_gpt_03_industrial_seed42.txt
│   │   ├── P3__formal_v1__P3_gpt_03_industrial_seed43.txt
│   │   ├── P3__formal_v1__P3_gpt_03_industrial_seed44.txt
│   │   ├── P3__formal_v1__P3_gpt_03_industrial_seed45.txt
│   │   ├── P3__formal_v1__P3_sim_01_seed42.txt
│   │   ├── P3__formal_v1__P3_sim_01_seed43.txt
│   │   ├── P3__formal_v1__P3_sim_01_seed44.txt
│   │   ├── P3__formal_v1__P3_sim_01_seed45.txt
│   │   ├── P3__formal_v1__P3_sim_02_seed42.txt
│   │   ├── P3__formal_v1__P3_sim_02_seed43.txt
│   │   ├── P3__formal_v1__P3_sim_02_seed44.txt
│   │   ├── P3__formal_v1__P3_sim_02_seed45.txt
│   │   ├── P3__formal_v1__P3_sim_03_seed42.txt
│   │   ├── P3__formal_v1__P3_sim_03_seed43.txt
│   │   ├── P3__formal_v1__P3_sim_03_seed44.txt
│   │   └── P3__formal_v1__P3_sim_03_seed45.txt
│   ├── first_frame.png
│   └── README.md
├── scripts/
│   ├── evaluate_v2.py
│   └── run_eval.sh
├── tools/
│   ├── audit_document_only_package.py
│   └── build_document_only_package.py
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P3/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | 原文未提供 | 本次不迁移、不创建占位文件。 |
| 当前入口与测量器 | P3/evaluator/evaluate.py、evaluate_raw_legacy.py | V3 门控后实际调用 evaluate_raw_legacy.py；文件名含 legacy 但处于当前物理执行链。 |
| 历史评分适配 | P3/evaluator/measure_backend.py | 旧评分 replay 入口依赖旧 scoring_v2 包，当前 V3 不调用。 |
| 跟踪与配置 | P3/evaluator/track_worker_p3.py、config.yaml、config_v2_geometric.json | 冻结跟踪种子/几何配置和旧几何评分设置。当前 P3 用已关联坐标缓存重新计算物理量。 |
| 历史批量与重评分 | P3/evaluator/run_batch.py、run_batch_pre_continuous_aggregate_20260826.py、rescore_continuous.py、rescore_v2_geometric.py | 保留的原始批处理与旧分数迁移，不替代当前 scripts/run_eval.sh。 |
| 审计与原有回归 | P3/evaluator/audit_formal_artifacts.py、write_human_sanity_review.py、validate_synthetic.py、test_rescore_continuous.py、synthetic_validation.json | 审核、人工检查、合成回归代码及历史验证记录；没有再复制到 utils/。 |
| 逐样本视频提示词 | P3/video_prompts/*.txt | 保留已有视频续写提示词；首帧来源清单已随 first_frames 目录删除。 |
| 全局工具 | tools/audit_document_only_package.py、build_document_only_package.py | 历史 document-only 审计/构建工具，需要外部交付输入；分组 data 清单已删除。 |
| 当前批量入口 | P3/scripts/run_eval.sh、scripts/run_eval.sh、scripts/evaluate_v2.py | 文件名 evaluate_v2 保留，但该包装器调用当前 V3 run_all_eval；旧 --verify 已移除。 |

P4/P6/P9/P11 也已合并 evaluator/utils。P9 同样需要坐标缓存；P4 使用已有 CPU 提取选项；P11/prompts/index.txt 是批次说明，不是生成 prompt。G3 的多样本 video_prompts/ 保持现有形式。

## G4：共享库与独立仿真提示词；典型题 P14 — 点光源与影线共点

本组题目：P14, P16, P18, P20, P21。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P14，并保留本组全部公共文件。

```text
g4/
├── P14/
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   ├── video.txt
│   │   └── video_simulation.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P14/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P14/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前入口与后端 | P14/evaluator/evaluate.py、measure_backend.py | V3 入口调公共 runtime；测量后端调用 shared/physeval 的 P14 共点规则。 |
| 输入提示词 | P14/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 分路线提示词 | P14/prompts/video.txt、video_simulation.txt | 两份 prompt 内容不同，因此分别保留；根 prompt_sim.txt 已归档。 |
| 依赖与运行 | P14/evaluator/requirements.txt、P14/scripts/run_eval.sh、scripts/run_eval.sh | 物理依赖、单题和整组运行。 |

P16/P18/P20/P21 复用同一套 physeval；P21 也有独立 simulation prompt。首帧路线及数量以各题文件和 metadata 为准。

## G5：独立 OpenCV 测量、ROI 与来源缺口记录；典型题 P36 — 磁体与对照物运动

本组题目：P21b, P21c, P23, P36, P44。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P36，并保留本组全部公共文件。

```text
g5/
├── P36/
│   ├── evaluator/
│   │   ├── utils/
│   │   │   └── README.md
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   ├── annotate_roi.py
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
├── reports/
│   └── report.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P36/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P36/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前评测实现 | P36/evaluator/evaluate.py、measure_backend.py | V3 门控入口与磁体/对照物运动测量后端。 |
| 输入提示词 | P36/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| ROI 与运行 | P36/scripts/annotate_roi.py、run_eval.sh、scripts/run_eval.sh | 标注 evaluator/roi.json；单题/整组运行使用公共批量器。 |
| 辅助说明 | P36/evaluator/utils/README.md | 保留的辅助目录说明。 |
| 分组报告 | reports/report.md | 分组历史汇总移入分组 reports，未归到某一道题。 |

P44 也有保留的 simulation 图和来源缺口说明；P21b/P21c/P23 的 simulation 目录仅有说明文件，没有可执行 renderer。

## G6：共享测量库与体积守恒测量；典型题 P48 — 液滴合并体积守恒

本组题目：P45, P48, P8b。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P48，并保留本组全部公共文件。

```text
g6/
├── P48/
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P48/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P48/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前入口与共享后端 | P48/evaluator/evaluate.py、measure_backend.py | V3 入口；共享 physeval 的 P48 液滴体积规则。 |
| 输入提示词 | P48/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 运行与依赖 | P48/evaluator/requirements.txt、P48/scripts/run_eval.sh、scripts/run_eval.sh | 题目物理依赖及单题/分组运行。 |

P45/P8b 同样使用公共 physeval。P45 当前只有 GPT 首帧，因此无需人为增加 simulation 资产。

## G7：题目模块拆分与分组分发运行时；典型题 P7 — 实心球与圆环滚动

本组题目：P10, P12, P27, P7, P8c。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P7，并保留本组全部公共文件。

```text
g7/
├── evaluator/
│   ├── contract.py
│   ├── evaluate.py
│   ├── evaluate_one.py
│   ├── physics.py
│   ├── renormalize_results.py
│   ├── requirements.txt
│   ├── result_schema.py
│   ├── run_batch.py
│   ├── task_entrypoint.py
│   ├── test_proxy.py
│   ├── validate_results.py
│   └── validate_schema.py
├── P7/
│   ├── evaluator/
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── README.md
│   │   ├── batch.py
│   │   ├── evaluate.py
│   │   ├── evaluate_p7.py
│   │   ├── evaluate_raw_legacy.py
│   │   ├── measure_backend.py
│   │   ├── p7_rotational.py
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   ├── run_eval.sh
│   │   └── run_eval_legacy.sh
│   ├── video_prompts/
│   │   ├── result_sample_00.txt
│   │   ├── result_sample_01.txt
│   │   ├── result_sample_02.txt
│   │   ├── result_sample_03.txt
│   │   ├── result_sample_04.txt
│   │   ├── result_sample_05.txt
│   │   ├── result_sample_06.txt
│   │   ├── result_sample_07.txt
│   │   ├── result_sample_08.txt
│   │   ├── result_sample_09.txt
│   │   ├── result_sample_10.txt
│   │   ├── result_sample_11.txt
│   │   ├── result_sample_12.txt
│   │   ├── result_sample_13.txt
│   │   ├── result_sample_14.txt
│   │   ├── result_sample_15.txt
│   │   ├── result_sample_16.txt
│   │   ├── result_sample_17.txt
│   │   ├── result_sample_18.txt
│   │   ├── result_sample_19.txt
│   │   ├── result_sample_20.txt
│   │   ├── result_sample_21.txt
│   │   ├── result_sample_22.txt
│   │   └── result_sample_23.txt
│   ├── first_frame.png
│   └── README.md
├── scripts/
│   ├── evaluate_v2.py
│   ├── run_eval.sh
│   └── run_eval_legacy.sh
├── tools/
│   └── blender_scene_builder.py
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P7/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P7/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前题目入口 | P7/evaluator/evaluate.py | 调用统一 V3 runtime；一致性通过后，runtime 调分组 evaluator/evaluate.py。 |
| 本题物理模块 | P7/evaluator/p7_rotational.py | 从原 evaluator/tasks 拆出，现由分组解码器调用；共用 shared/g7_cv_common.py。 |
| 保留的旧接口 | P7/evaluator/evaluate_raw_legacy.py、measure_backend.py、batch.py | 旧 task adapter / evaluator_common 入口；不作为当前运行方法，部分依赖未随源码交付。 |
| 另一条提取方案 | P7/evaluator/evaluate_p7.py | 保留的 SAM3/CoTracker 方案，需额外模型环境，当前默认分发不调用。 |
| 输入提示词 | P7/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 运行脚本 | P7/scripts/run_eval.sh、scripts/run_eval.sh、scripts/evaluate_v2.py | 当前单题/分组批量入口；带 legacy 的脚本只作历史保留。 |
| 全局生成器 | tools/blender_scene_builder.py | 离线场景/输入生成工具，不放进题目 evaluator。 |

其余四题分别归档为 P8c/evaluator/p8c_pendulum.py、P10/evaluator/p10_mechanics.py、P12/evaluator/p12_optics.py、P27/evaluator/p27_thermal.py。P10/P12/P27 的旧 prompt 有内容差异，保存在 prompts/video_legacy.txt。

### G7 全局 evaluator 的职责与已删除清单依赖

| 保留文件 | 用途 |
| --- | --- |
| evaluator/evaluate.py | 统一解码并向各题 evaluator 模块分发，是当前 V3 的 G7 物理后端入口。 |
| evaluator/physics.py | 物理条件判定与失败原因。 |
| evaluator/contract.py | G7 原公开结果合同与代理评分构建，用于历史单样本/批量适配。 |
| evaluator/evaluate_one.py、run_batch.py | 原历史批次适配，依赖已删除的 g7/data 清单；当前需适配输入方式。 |
| evaluator/task_entrypoint.py | 旧固定题号 CLI 适配器，供保留的 evaluate_raw_legacy.py 使用。 |
| evaluator/result_schema.py、renormalize_results.py | 历史结果 schema、结果归一化与格式迁移。 |
| evaluator/validate_results.py、validate_schema.py、test_proxy.py | 历史输出校验与原有代理评分回归。 |
| evaluator/requirements.txt | G7 共享物理运行时的基础依赖。 |

## G8：标注、分割跟踪与共享 refined 后端；典型题 P37 — 完整环与开口环升高

本组题目：P34, P37, P38, P39, P41。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P37，并保留本组全部公共文件。

```text
g8/
├── P37/
│   ├── annotations/
│   │   └── first_frame_annotations.json
│   ├── docs/
│   │   └── evaluator.md
│   ├── evaluator/
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── README.md
│   │   ├── batch.py
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   ├── README.md
│   │   ├── requirements-tested.txt
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P37/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P37/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前题目入口 | P37/evaluator/evaluate.py、measure_backend.py | V3 单视频门控入口；物理适配导入 v3/refined_evaluators/runtime.py。 |
| manifest 批量适配 | P37/evaluator/batch.py | 调用公共 manifest 批处理并运行本题公开入口；可使用外部 manifest；原 scripts/run_eval.sh 的样本清单已删除。 |
| 标注与首帧 | P37/annotations/first_frame_annotations.json、first_frame.png、prompts/first_frame.txt | 标注与图像分开；prompts/first_frame.txt 是首帧生成文本，不是视频续写文本。 |
| 输入提示词 | P37/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 说明与依赖 | P37/docs/evaluator.md、evaluator/README.md、requirements.txt、requirements-tested.txt | 保留历史方法说明及依赖记录；当前评分以 SCORING_V3.md 为准。 |
| 运行与占位 | P37/scripts/run_eval.sh、scripts/run_eval.sh、P37/evaluator/utils/README.md、__init__.py | 标准批量入口；本题 utils 只是保留的命名空间，实际公共提取代码在 refined_evaluators。 |

P38/P39/P41 采用相同公共后端布局。P34 保留题目自己的 evaluator/utils/{common,media,models,measurement,debug}.py 和原有测试；该题没有首帧 annotation 文件。

## G9：终端速度/几何测量与独立气泡题例外；典型题 P49 — 大小球终端速度与半径平方律

本组题目：P43, P47, P49。标准首帧位于各题根目录；已删除题目样本清单。此处只展开 P49，并保留本组全部公共文件。

```text
g9/
├── P49/
│   ├── annotations/
│   │   └── first_frame_annotations.json
│   ├── docs/
│   │   └── evaluator.md
│   ├── evaluator/
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── README.md
│   │   ├── batch.py
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   ├── README.md
│   │   ├── requirements-tested.txt
│   │   └── requirements.txt
│   ├── prompts/
│   │   ├── first_frame.txt
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   ├── first_frame.png
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 标准首帧 | P49/first_frame.png | 原选定图片移至题目根目录，内容未修改。 |
| 首帧生成提示词 | P49/prompts/first_frame.txt | 原生成提示词逐字节迁移，与视频续写提示词分开。 |
| 当前入口与后端 | P49/evaluator/evaluate.py、measure_backend.py | V3 门控入口与 refined_evaluators 物理适配。 |
| 输入提示词 | P49/prompts/；video_prompts/（有文件时） | 首帧生成文本与视频续写文本分别保存；metadata 和 samples.csv 已删除。 |
| 两种提示词 | P49/prompts/video.txt、prompts/first_frame.txt | 分别是视频续写文本与首帧生成文本。 |
| 说明与依赖 | P49/docs/evaluator.md、evaluator/README.md、requirements.txt、requirements-tested.txt | 历史测量说明与依赖清单；当前入口与计分按新 README。 |
| 批量与命名空间 | P49/evaluator/batch.py、scripts/run_eval.sh、scripts/run_eval.sh、evaluator/utils/README.md、__init__.py | 备用 manifest 适配、当前单题/分组运行与保留命名空间。 |

P43 布局与 P49 接近。P47 没有 first_frame_annotations.json，物理测量使用 refined_evaluators/bubbles.py 的独立轮廓几何流程；同样保留 V3 一致性门控。

## 运行路径与 README 的适配结果

MP4、模型权重和完整测量缓存未随源码交付。以下是运行路径约定，目录不一定已存在：

```text
v3/
├── gX/Pxx/
│   ├── first_frame.png
│   ├── output_videos/<model>/video.mp4
│   └── eval_results/
│       ├── run_logs/
│       └── debug/<source_video_stem>/tracks_raw.npz  # G3/P3、P9 需准备的缓存
└── results/gX/Pxx/eval_results/<model>/
    ├── result_<sample_id>.json
    └── debug/ 或 debug_<sample_id>/
```

40 道题的 README 已使用根目录标准首帧，并将生成首帧的提示词和视频续写提示词分开说明。G3 五题没有原始首帧生成提示词，本次仅迁移图片。已删除 metadata 的登记样本批量命令不再列作可直接运行的示例；G7 的旧任务清单依赖明确注明尚需适配。

本次删除了剩余 31 个题目级 first_frames 目录及其中 232 个文件，全部 40 题均已无该目录。40 张根目录标准首帧与 35 份生成提示词的内容保持一致，题目 README 链接和典型题文件树已同步清理。前次公共回归运行 36 项，34 项通过、2 项跳过；本次仅删除目录和更新文档，未执行完整 GPU 视频评测。
