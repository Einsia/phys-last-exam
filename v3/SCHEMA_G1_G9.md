# VDM-Bench V3：G1–G9 新目录与代码文件 Schema（README 已核对）

本文件用于审核已经落地的目录重构。核对日期：2026-09-30；仓库：vdmbench/v3；本次重构基于 Git 版本 c5cf5e7，以下描述重构后的文件体系。

每个分组仅展开一道典型题目，同时完整保留该分组的 scripts、data、evaluator、tools、reports 等公共目录。下面的目录树根据当前本地文件直接生成，叶子节点是实际文件；其他题目仅列题号。MP4、权重、坐标缓存和生成结果未随源码交付，另列运行路径约定。

文件归属和调用关系以代码为依据：文件名中的 v2/legacy 不能单独说明它是否处于当前执行链，实际调用以 unified_evaluators/runtime.py 为准。

## 公共路径契约与文件分类

| 分类 | 固定位置或保留形式 | 用途与解析基准 |
| --- | --- | --- |
| 正式样本关联 | gX/Pxx/data/metadata.json | 唯一正式 metadata；samples 供当前评测读取。旧对象合并在 legacy_metadata，仅作归档。 |
| 题目提示词 | Pxx/prompts/video.txt；有差异时保留 video_simulation.txt / video_legacy.txt | 题目层文本归档；实际样本 prompt 优先来自 metadata 的 video_prompt。 |
| 逐样本提示词 | G3/G7 的 Pxx/video_prompts/ | 保留已有逐样本文本，不把它们当成一份可覆盖所有样本的题目 prompt。 |
| 首帧与生成来源 | Pxx/first_frames/gpt/、simulation/、provided/ | GPT、仿真、外部提供首帧分别归档；只有实际有文件的目录才出现在树中。real/.gitkeep 是空位标记。 |
| 首帧标注 | Pxx/annotations/first_frame_annotations.json | 仅有标注的题目存在；metadata 的 annotation 指向这里。G2/G5 的 ROI 配置仍由 evaluator/roi.json 承载。 |
| 单题代码 | Pxx/evaluator/ | V3 单题入口、物理后端或适配器、题目配置、依赖，以及原有辅助脚本与测试。 |
| 单题/分组运行脚本 | Pxx/scripts/run_eval.sh；gX/scripts/run_eval.sh | 分别选择一题或整组的登记样本；公共执行器位于 v3/scripts/run_all_eval.py。 |
| 报告与说明 | Pxx/reports/、Pxx/docs/；分组 reports/ | 保存历史测量报告和评测说明；当前运行方法和结果语义以 README / SCORING_V3.md 为准。 |
| 公共测量库 | v3/shared/physeval/；v3/shared/g7_cv_common.py | G1/G4/G6 共用测量库；G7 的 P7/P8c 共用 CV helper。 |
| 路径基准 | metadata 的 video_path / source_video_path / image_path / annotation | 以 Pxx/ 为基准，不以 data/ 为基准。外部显式 manifest 的相对路径按其所在目录解释。 |

通用单题入口读取 data/metadata.json，检查输入哈希，再运行一致性门控；通过后调用本组物理后端。G1/G4/G6 的辅助 M2/M3/… 合并到公开 M2；G8/G9 公开只定义 M1。当前 V3 总分为一致性 15% 与纯物理分 85%，不会把历史 V2 识别奖励再加一次。

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
| unified_evaluators/runtime.py | 40 个题目共用的 metadata 加载、门控、后端调度、调试和发布流程。 |
| unified_evaluators/consistency.py | 本地/HTTP VLM 一致性判断、帧采样、参数和模型接口。 |
| unified_evaluators/contract.py、physics.py | 统一结果结构与 G1–G9 物理测量映射；G3/G7 的指标转换在 physics.py。 |
| unified_evaluators/execute_backend.py | 物理子进程环境与 OpenCV 兼容包装。 |
| shared/physeval/context.py、schema.py、registry.py | 评测上下文、原始测量结果和任务注册；共享任务规则位于 tasks/。 |
| shared/physeval/video.py、track.py、fitting.py、pendulum.py、liquid.py、viz.py | 视频解码、目标轨迹、拟合、摆/液面分析和可视化。 |
| shared/g7_cv_common.py | G7 公共图像检测和几何辅助，不属于某一题的资产。 |
| refined_evaluators/ | G8/G9 共用物理提取、数值计算、模型资源、媒体导出、评分、manifest 批处理。P34 还保留自己的 evaluator/utils 实现。 |
| reliability/ | 测量状态分类、控制样本协议与 VLM 对照工具。 |
| scripts/run_all_eval.py、install_unified_entries.py | 登记样本批量执行，以及生成单题/分组入口的维护工具；均已适配 data/metadata.json。 |
| scripts/ 中其他文件 | 控制视频构建、可靠性校准、准备、审计和报告辅助工具；不是题目单视频入口。 |
| tests/ | 公共回归测试；test_batch_schema_paths.py 验证移入 data/ 后的题目根目录解析与入口生成。单题原有测试仍保留在 evaluator/。 |

## G1：共享轨迹与物理测量库；典型题 P1 — 钢球自由下落

本组题目：P1, P2, P5, P8a。正式 metadata 登记 32 个样本关联。此处只展开 P1，并保留本组全部公共文件。

```text
g1/
├── P1/
│   ├── data/
│   │   ├── metadata.json
│   │   └── samples.csv
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   ├── gpt_02.png
│   │   │   ├── gpt_03.png
│   │   │   └── prompt.txt
│   │   └── simulation/
│   │       ├── sim_01.png
│   │       ├── sim_02.png
│   │       └── sim_03.png
│   ├── prompts/
│   │   └── video.txt
│   ├── reports/
│   │   └── report.md
│   ├── scripts/
│   │   └── run_eval.sh
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前单题入口 | P1/evaluator/evaluate.py | 调用 unified_evaluators/runtime.py，执行 V3 门控与统一输出。 |
| 物理适配 | P1/evaluator/measure_backend.py | 构建 Context；读取 data/samples.csv、prompts/video.txt，调用 shared.physeval 的 P1 规则。 |
| 样本与 prompt | P1/data/metadata.json、data/samples.csv、prompts/video.txt | 正式关联与旧 route/seed 索引分开；生成首帧的 prompt 在 first_frames/gpt/prompt.txt。 |
| 首帧资产 | P1/first_frames/gpt/*.png、simulation/*.png | 树中逐文件列出；GPT/仿真来源保持独立。 |
| 报告与依赖 | P1/reports/report.md、evaluator/requirements.txt | 历史测量报告与物理运行依赖。 |
| 单题/整组运行 | P1/scripts/run_eval.sh、scripts/run_eval.sh | 各自调用公共 run_all_eval.py，选择 P1 或 G1。 |

其余 P2/P5/P8a 使用相同公共库；辅助量按题目规则不同。P5 的公开 M2 未定义。

## G2：题目独立 OpenCV 后端与首帧生成代码；典型题 P19 — 连通器液面

本组题目：P13, P19, P28, P40, P42。正式 metadata 登记 7 个样本关联。此处只展开 P19，并保留本组全部公共文件。

```text
g2/
├── P19/
│   ├── data/
│   │   └── metadata.json
│   ├── evaluator/
│   │   ├── utils/
│   │   │   └── README.md
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   ├── gpt_02.png
│   │   │   ├── gpt_03.png
│   │   │   └── prompt.txt
│   │   ├── provided/
│   │   │   └── first_frame.png
│   │   ├── real/
│   │   │   └── .gitkeep
│   │   └── simulation/
│   │       ├── code/
│   │       │   ├── __init__.py
│   │       │   ├── builders_induction_surface.py
│   │       │   ├── builders_mech_optics.py
│   │       │   ├── builders_thermal_em.py
│   │       │   ├── common.py
│   │       │   ├── render.py
│   │       │   ├── render_all.py
│   │       │   ├── requirements.txt
│   │       │   └── tasks.json
│   │       └── sim_01.png
│   ├── prompts/
│   │   └── video.txt
│   ├── scripts/
│   │   ├── annotate_roi.py
│   │   └── run_eval.sh
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前评测代码 | P19/evaluator/evaluate.py、measure_backend.py | 前者是 V3 门控入口，后者测量左右液面及末段运动。 |
| ROI 标注 | P19/scripts/annotate_roi.py | 直接读取 first_frames 下的图；生成 evaluator/roi.json。run_eval.sh 不支持旧 --annotate 模式。 |
| 渲染入口 | P19/first_frames/simulation/code/render.py、render_all.py | 按生成配置生成单场景或批次仿真素材；不属于评测输出目录。 |
| 渲染实现 | code/common.py、builders_induction_surface.py、builders_mech_optics.py、builders_thermal_em.py、__init__.py | 本题保留的仿真 builder 与公共生成辅助。 |
| 生成配置/依赖 | code/tasks.json、requirements.txt | 首帧生成的参数与依赖，和正式评测 metadata 作用不同。 |
| 输入与占位 | P19/data/metadata.json、prompts/video.txt、first_frames/provided/first_frame.png、real/.gitkeep | provided 保留独立图像；real/.gitkeep 仅占位。evaluator/utils/README.md 也只是辅助位置说明。 |
| 运行脚本 | P19/scripts/run_eval.sh、scripts/run_eval.sh | 单题/整组的 V3 批量入口。 |

P13/P28 根首帧因与 first_frames 重复已去除；P19/P40/P42 的独立根首帧归到 provided/。本组每题保留自己的 OpenCV 测量后端。

## G3：原始物理提取、冻结配置与历史审计文件合并；典型题 P3 — 30° 与 60° 斜抛

本组题目：P11, P3, P4, P6, P9。正式 metadata 登记 168 个样本关联。此处只展开 P3，并保留本组全部公共文件。

```text
g3/
├── data/
│   └── manifest.json
├── P3/
│   ├── data/
│   │   └── metadata.json
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
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── P3_gpt_01_modern.png
│   │   │   ├── P3_gpt_02_classic.png
│   │   │   ├── P3_gpt_03_industrial.png
│   │   │   └── source_manifest.json
│   │   ├── simulation/
│   │   │   ├── code/
│   │   │   │   ├── simulation/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── common.py
│   │   │   │   ├── p3_sop_variants.py
│   │   │   │   └── tasks_all.json
│   │   │   ├── P3_sim_01.json
│   │   │   ├── P3_sim_01.png
│   │   │   ├── P3_sim_02.json
│   │   │   ├── P3_sim_02.png
│   │   │   ├── P3_sim_03.json
│   │   │   ├── P3_sim_03.png
│   │   │   └── source_manifest.json
│   │   └── manifest.json
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
| 当前入口与测量器 | P3/evaluator/evaluate.py、evaluate_raw_legacy.py | V3 门控后实际调用 evaluate_raw_legacy.py；文件名含 legacy 但处于当前物理执行链。 |
| 历史评分适配 | P3/evaluator/measure_backend.py | 旧评分 replay 入口依赖旧 scoring_v2 包，当前 V3 不调用。 |
| 跟踪与配置 | P3/evaluator/track_worker_p3.py、config.yaml、config_v2_geometric.json | 冻结跟踪种子/几何配置和旧几何评分设置。当前 P3 用已关联坐标缓存重新计算物理量。 |
| 历史批量与重评分 | P3/evaluator/run_batch.py、run_batch_pre_continuous_aggregate_20260826.py、rescore_continuous.py、rescore_v2_geometric.py | 保留的原始批处理与旧分数迁移，不替代当前 scripts/run_eval.sh。 |
| 审计与原有回归 | P3/evaluator/audit_formal_artifacts.py、write_human_sanity_review.py、validate_synthetic.py、test_rescore_continuous.py、synthetic_validation.json | 审核、人工检查、合成回归代码及历史验证记录；没有再复制到 utils/。 |
| 首帧生成 | P3/first_frames/simulation/code/p3_sop_variants.py、simulation/common.py、simulation/__init__.py、tasks_all.json | 仿真生成和参数归于首帧来源目录。 |
| 来源与 prompt | P3/first_frames/manifest.json、各来源 source_manifest.json、simulation/*.json、video_prompts/*.txt | 首帧来源、仿真参数和逐样本 prompt 的历史记录。 |
| 全局清单/工具 | data/manifest.json、tools/audit_document_only_package.py、build_document_only_package.py | 历史 G3 交付包清单及 document-only 审计/构建工具，需要旧交付输入。 |
| 当前批量入口 | P3/scripts/run_eval.sh、scripts/run_eval.sh、scripts/evaluate_v2.py | 文件名 evaluate_v2 保留，但该包装器调用当前 V3 run_all_eval；旧 --verify 已移除。 |

P4/P6/P9/P11 也已合并 evaluator/utils。P9 同样需要坐标缓存；P4 使用已有 CPU 提取选项；P11/prompts/index.txt 是批次说明，不是生成 prompt。G3 的多样本 video_prompts/ 保持现有形式。

## G4：共享库与独立仿真提示词；典型题 P14 — 点光源与影线共点

本组题目：P14, P16, P18, P20, P21。正式 metadata 登记 36 个样本关联。此处只展开 P14，并保留本组全部公共文件。

```text
g4/
├── P14/
│   ├── data/
│   │   ├── metadata.json
│   │   └── samples.csv
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   ├── gpt_02.png
│   │   │   └── prompt.txt
│   │   └── simulation/
│   │       └── sim_01.png
│   ├── prompts/
│   │   ├── video.txt
│   │   └── video_simulation.txt
│   ├── reports/
│   │   └── report.md
│   ├── scripts/
│   │   └── run_eval.sh
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前入口与后端 | P14/evaluator/evaluate.py、measure_backend.py | V3 入口调公共 runtime；测量后端调用 shared/physeval 的 P14 共点规则。 |
| 正式与兼容索引 | P14/data/metadata.json、samples.csv | 登记样本关联与旧 route/seed 索引。 |
| 分路线提示词 | P14/prompts/video.txt、video_simulation.txt | 两份 prompt 内容不同，因此分别保留；根 prompt_sim.txt 已归档。 |
| 首帧与报告 | P14/first_frames/gpt/、simulation/、reports/report.md | 首帧保持来源分组；报告从题目根目录移入 reports。 |
| 依赖与运行 | P14/evaluator/requirements.txt、P14/scripts/run_eval.sh、scripts/run_eval.sh | 物理依赖、单题和整组运行。 |

P16/P18/P20/P21 复用同一套 physeval；P21 也有独立 simulation prompt。首帧路线及数量以各题文件和 metadata 为准。

## G5：独立 OpenCV 测量、ROI 与来源缺口记录；典型题 P36 — 磁体与对照物运动

本组题目：P21b, P21c, P23, P36, P44。正式 metadata 登记 7 个样本关联。此处只展开 P36，并保留本组全部公共文件。

```text
g5/
├── P36/
│   ├── data/
│   │   └── metadata.json
│   ├── evaluator/
│   │   ├── utils/
│   │   │   └── README.md
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   └── prompt.txt
│   │   ├── real/
│   │   │   └── .gitkeep
│   │   └── simulation/
│   │       ├── code/
│   │       │   └── README.md
│   │       ├── README.md
│   │       ├── sim_01.png
│   │       └── sim_first_frame.png
│   ├── prompts/
│   │   └── video.txt
│   ├── scripts/
│   │   ├── annotate_roi.py
│   │   └── run_eval.sh
│   └── README.md
├── reports/
│   └── report.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前评测实现 | P36/evaluator/evaluate.py、measure_backend.py | V3 门控入口与磁体/对照物运动测量后端。 |
| 样本与提示词 | P36/data/metadata.json、prompts/video.txt | 正式样本关联和题目续写文本。 |
| ROI 与运行 | P36/scripts/annotate_roi.py、run_eval.sh、scripts/run_eval.sh | 标注 evaluator/roi.json；单题/整组运行使用公共批量器。 |
| 保留的仿真首帧 | P36/first_frames/simulation/sim_01.png、sim_first_frame.png | 独立原始图片移入 simulation，当前包没有源 renderer。 |
| 来源说明与占位 | P36/first_frames/simulation/README.md、code/README.md、real/.gitkeep、evaluator/utils/README.md | 明确源 renderer 缺失；real 和 utils 为保留位置，不是隐藏的生成器或提取器。 |
| 分组报告 | reports/report.md | 分组历史汇总移入分组 reports，未归到某一道题。 |

P44 也有保留的 simulation 图和来源缺口说明；P21b/P21c/P23 的 simulation 目录仅有说明文件，没有可执行 renderer。

## G6：共享测量库与题目报告；典型题 P48 — 液滴合并体积守恒

本组题目：P45, P48, P8b。正式 metadata 登记 20 个样本关联。此处只展开 P48，并保留本组全部公共文件。

```text
g6/
├── P48/
│   ├── data/
│   │   ├── metadata.json
│   │   └── samples.csv
│   ├── evaluator/
│   │   ├── evaluate.py
│   │   ├── measure_backend.py
│   │   └── requirements.txt
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   └── prompt.txt
│   │   └── simulation/
│   │       └── sim_01.png
│   ├── prompts/
│   │   └── video.txt
│   ├── reports/
│   │   └── report.md
│   ├── scripts/
│   │   └── run_eval.sh
│   └── README.md
└── scripts/
    └── run_eval.sh
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前入口与共享后端 | P48/evaluator/evaluate.py、measure_backend.py | V3 入口；共享 physeval 的 P48 液滴体积规则。 |
| 样本与提示词 | P48/data/metadata.json、samples.csv、prompts/video.txt | 正式样本关联、旧 route/seed 索引和视频 prompt。 |
| 首帧素材 | P48/first_frames/gpt/、simulation/ | GPT 与仿真来源下的首帧逐文件保留。 |
| 历史报告 | P48/reports/report.md | 测量报告从题目根目录归档；不由当前运行脚本生成。 |
| 运行与依赖 | P48/evaluator/requirements.txt、P48/scripts/run_eval.sh、scripts/run_eval.sh | 题目物理依赖及单题/分组运行。 |

P45/P8b 同样使用公共 physeval。P45 当前只有 GPT 首帧，因此无需人为增加 simulation 资产。

## G7：题目模块拆分与分组分发运行时；典型题 P7 — 实心球与圆环滚动

本组题目：P10, P12, P27, P7, P8c。正式 metadata 登记 125 个样本关联。此处只展开 P7，并保留本组全部公共文件。

```text
g7/
├── data/
│   ├── first_frame_manifest.json
│   ├── manifest.json
│   └── tasks.json
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
│   ├── data/
│   │   └── metadata.json
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
│   ├── first_frames/
│   │   ├── gpt/
│   │   │   ├── gpt_01.png
│   │   │   ├── gpt_02.png
│   │   │   ├── gpt_03.png
│   │   │   ├── prompt.txt
│   │   │   └── video_prompt.txt
│   │   ├── provided/
│   │   │   └── first_frame_01.png
│   │   └── simulation/
│   │       ├── code/
│   │       │   ├── render.py
│   │       │   └── requirements.txt
│   │       ├── sim_01.png
│   │       ├── sim_02.png
│   │       └── sim_03.png
│   ├── prompts/
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
| 当前题目入口 | P7/evaluator/evaluate.py | 调用统一 V3 runtime；一致性通过后，runtime 调分组 evaluator/evaluate.py。 |
| 本题物理模块 | P7/evaluator/p7_rotational.py | 从原 evaluator/tasks 拆出，现由分组解码器调用；共用 shared/g7_cv_common.py。 |
| 保留的旧接口 | P7/evaluator/evaluate_raw_legacy.py、measure_backend.py、batch.py | 旧 task adapter / evaluator_common 入口；不作为当前运行方法，部分依赖未随源码交付。 |
| 另一条提取方案 | P7/evaluator/evaluate_p7.py | 保留的 SAM3/CoTracker 方案，需额外模型环境，当前默认分发不调用。 |
| 首帧/生成代码 | P7/first_frames/gpt/、provided/、simulation/code/render.py、requirements.txt | 保留首帧来源、生成 prompt 和可用仿真渲染入口。 |
| 提示词与正式关联 | P7/prompts/video.txt、video_prompts/*.txt、data/metadata.json | 题目与逐样本文本分开；旧 metadata 已合并到 legacy_metadata。 |
| 运行脚本 | P7/scripts/run_eval.sh、scripts/run_eval.sh、scripts/evaluate_v2.py | 当前单题/分组批量入口；带 legacy 的脚本只作历史保留。 |
| 全局生成器 | tools/blender_scene_builder.py | 离线场景/输入生成工具，不放进题目 evaluator。 |

其余四题分别归档为 P8c/evaluator/p8c_pendulum.py、P10/evaluator/p10_mechanics.py、P12/evaluator/p12_optics.py、P27/evaluator/p27_thermal.py。P10/P12/P27 的旧 prompt 有内容差异，保存在 prompts/video_legacy.txt。

### G7 全局 evaluator 与 data 的保留职责

| 保留文件 | 用途 |
| --- | --- |
| evaluator/evaluate.py | 统一解码并向各题 evaluator 模块分发，是当前 V3 的 G7 物理后端入口。 |
| evaluator/physics.py | 物理条件判定与失败原因。 |
| evaluator/contract.py | G7 原公开结果合同与代理评分构建，用于历史单样本/批量适配。 |
| evaluator/evaluate_one.py、run_batch.py | 读取 g7/data 的历史批次适配，调用共享物理分发；默认 V3 shell 则读取每题 metadata。 |
| evaluator/task_entrypoint.py | 旧固定题号 CLI 适配器，供保留的 evaluate_raw_legacy.py 使用。 |
| evaluator/result_schema.py、renormalize_results.py | 历史结果 schema、结果归一化与格式迁移。 |
| evaluator/validate_results.py、validate_schema.py、test_proxy.py | 历史输出校验与原有代理评分回归。 |
| evaluator/requirements.txt | G7 共享物理运行时的基础依赖。 |
| data/tasks.json | 跨题规格、参数及 prompt；P8c 从该清单读取理论角度/周期比。 |
| data/manifest.json、first_frame_manifest.json | 历史批次样本清单与首帧资产清单；内部资产路径仍相对于 g7 根目录。 |

## G8：标注、分割跟踪与共享 refined 后端；典型题 P37 — 完整环与开口环升高

本组题目：P34, P37, P38, P39, P41。正式 metadata 登记 5 个样本关联。此处只展开 P37，并保留本组全部公共文件。

```text
g8/
├── P37/
│   ├── annotations/
│   │   └── first_frame_annotations.json
│   ├── data/
│   │   └── metadata.json
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
│   ├── first_frames/
│   │   └── provided/
│   │       ├── first_frame_01.png
│   │       └── prompt.txt
│   ├── prompts/
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前题目入口 | P37/evaluator/evaluate.py、measure_backend.py | V3 单视频门控入口；物理适配导入 v3/refined_evaluators/runtime.py。 |
| manifest 批量适配 | P37/evaluator/batch.py | 调用公共 manifest 批处理并运行本题公开入口；标准 README 使用 scripts/run_eval.sh。 |
| 标注与首帧 | P37/annotations/first_frame_annotations.json、first_frames/provided/first_frame_01.png、prompt.txt | 标注与图像分开；provided/prompt.txt 是首帧生成文本，不是视频续写文本。 |
| 视频关联与 prompt | P37/data/metadata.json、prompts/video.txt | 旧 metadata 合并；根重复文本已清理。 |
| 说明与依赖 | P37/docs/evaluator.md、evaluator/README.md、requirements.txt、requirements-tested.txt | 保留历史方法说明及依赖记录；当前评分以 SCORING_V3.md 为准。 |
| 运行与占位 | P37/scripts/run_eval.sh、scripts/run_eval.sh、P37/evaluator/utils/README.md、__init__.py | 标准批量入口；本题 utils 只是保留的命名空间，实际公共提取代码在 refined_evaluators。 |

P38/P39/P41 采用相同公共后端布局。P34 保留题目自己的 evaluator/utils/{common,media,models,measurement,debug}.py 和原有测试；该题没有首帧 annotation 文件。

## G9：终端速度/几何测量与独立气泡题例外；典型题 P49 — 大小球终端速度与半径平方律

本组题目：P43, P47, P49。正式 metadata 登记 3 个样本关联。此处只展开 P49，并保留本组全部公共文件。

```text
g9/
├── P49/
│   ├── annotations/
│   │   └── first_frame_annotations.json
│   ├── data/
│   │   └── metadata.json
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
│   ├── first_frames/
│   │   └── provided/
│   │       ├── first_frame_01.png
│   │       └── prompt.txt
│   ├── prompts/
│   │   └── video.txt
│   ├── scripts/
│   │   └── run_eval.sh
│   └── README.md
├── scripts/
│   └── run_eval.sh
└── README.md
```

| 代码/文件分类 | 文件位置 | 职责与当前调用关系 |
| --- | --- | --- |
| 当前入口与后端 | P49/evaluator/evaluate.py、measure_backend.py | V3 门控入口与 refined_evaluators 物理适配。 |
| 样本/首帧/标注 | P49/data/metadata.json、first_frames/provided/first_frame_01.png、annotations/first_frame_annotations.json | 明确关联视频、首帧与审核过的几何标注。 |
| 两种提示词 | P49/prompts/video.txt、first_frames/provided/prompt.txt | 分别是视频续写文本与首帧生成文本。 |
| 说明与依赖 | P49/docs/evaluator.md、evaluator/README.md、requirements.txt、requirements-tested.txt | 历史测量说明与依赖清单；当前入口与计分按新 README。 |
| 批量与命名空间 | P49/evaluator/batch.py、scripts/run_eval.sh、scripts/run_eval.sh、evaluator/utils/README.md、__init__.py | 备用 manifest 适配、当前单题/分组运行与保留命名空间。 |

P43 布局与 P49 接近。P47 没有 first_frame_annotations.json，物理测量使用 refined_evaluators/bubbles.py 的独立轮廓几何流程；同样保留 V3 一致性门控。

## 运行路径与 README 的适配结果

下面是运行资产路径示例，不代表这些 MP4 / 输出文件已经存在于源码目录。默认新结果写入 v3/results/，题目内 eval_results/ 可存历史缓存或显式选定的输出；题目/分组 shell 日志另保存在各自目录。

```text
v3/
├── gX/Pxx/
│   ├── output_videos/<model>/sample_00.mp4
│   └── eval_results/
│       ├── run_logs/
│       └── debug/<source_video_stem>/tracks_raw.npz  # G3/P3、P9 恢复的旧缓存
└── results/gX/Pxx/eval_results/<model>/
    ├── result_sample_00.json
    ├── batch_summary.json
    └── debug/ 或 debug_sample_00/
```

| README 原有问题 | 已完成的修正 |
| --- | --- |
| G3/G7 仍写 --verify 与已有分数复算 | 改为实际 V3 --video / --output、标准批量入口，并说明真正调用的物理提取器。 |
| 题目目录与分组目录的命令工作目录混用 | 每题给出从仓库根目录进入本题的 cd 命令；分组 README 的命令在 v3 根目录执行。 |
| scripts/launch.sh、PACKAGE_MANIFEST、legacy_v1、旧视频目录等不存在路径 | 移除失效运行指引；链接只指向当前存在的文件；运行资产另标前置。 |
| 写死不随仓库交付的 Python 路径 | 解释 EVALUATOR_PYTHON 和默认 v3/.venv/bin/python；共用环境说明给出本地/HTTP 门控条件。 |
| 混用首帧 prompt、视频 prompt、已移动 metadata/标注 | 改为实际新路径，强调 metadata 的路径基准仍是题目根目录。 |
| 旧 --annotate、视频目录位置参数和输出位置说明 | ROI 直接用 scripts/annotate_roi.py；标准 shell 只选登记样本，结果默认写 v3/results/。 |
| V2 含识别奖励的 metric 被误当当前分 | 改为纯物理 metric、原始 raw_metric 与 V3 一致性 15% / 物理 85% 总分语义。 |
| metadata 移入 data 后，批量器/入口生成器把 data 当题目目录 | 修复 run_all_eval.py 与 install_unified_entries.py 的题目/分组定位，并新增两项全局回归测试。 |

检查结果：40 道题的单视频 --help 与标准批量 --help 共 80 项通过；9 个分组批量入口及 G3/G7 两个 evaluate_v2 包装器的 --help 通过。96 个 README 中的 162 个 bash 代码块语法通过，当前本地 Markdown 链接无失效项；409 个首帧/标注关联均可找到实际文件。新增批量路径/入口生成回归 2 项和现有一致性流程回归 15 项通过。

验证范围：本次验证目录、链接、shell 语法、参数入口以及隔离的批量路径逻辑，没有运行完整视频/模型端到端评测。当前可用环境缺少 cv2 / scipy，源码仓库也不含视频、模型权重和 G3 坐标缓存。各题 README 已写明恢复输入、模型和缓存的条件，因此不会把 --help 成功表述为完整物理评测成功。

维护入口：v3/README.md 为公共环境和运行规范；各题 README.md 为当前单题说明；v3/RESTRUCTURE_NOTES.md 为移动/去重记录；本文件为逐组文件级概览。目录树反映本次重构后的文件快照，后续代码移动时需同时更新 README、runtime 路径与该概览。
