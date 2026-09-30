# P6 — 纯滚动

本题属于 G3，使用重构后的 V3 文件布局。单视频入口为 `evaluator/evaluate.py`，执行一致性门控和物理评测。`scripts/run_eval.sh` 保留原批量接口，目前仍依赖已删除的样本清单。请保留完整 `v3/` 目录，公共实现通过相对路径加载。

目录归档规则见 [重构说明](../../RESTRUCTURE_NOTES.md)，当前评分与结果语义见 [SCORING_V3.md](../../SCORING_V3.md)。

## 输入与目录

本题标准首帧位于根目录的 `first_frame.png`。首帧生成提示词与视频续写提示词分别归档，评测时应使用与输入视频对应的实际续写文本。题目 `data/` 和 `reports/` 已按要求删除，不再附带 metadata 样本清单。

| 文件或目录 | 用途 |
| --- | --- |
| [first_frame.png](first_frame.png) | 本题选定的标准首帧。 |
| 首帧生成提示词（未提供） | 仓库仅有 first_frames/gpt/source_manifest.json 来源记录，原文未交付，本次没有迁移生成提示词。 |
| [video_prompts/](video_prompts/) | 逐样本视频续写文本；按视频来源选择对应文件。 |
| [first_frames/gpt/](first_frames/gpt/) | 其余 GPT 首帧和已有生成来源记录。 |
| [first_frames/simulation/](first_frames/simulation/) | 仿真首帧、来源记录及可用生成代码。 |
| [evaluator/evaluate.py](evaluator/evaluate.py) | 当前 V3 单视频入口，调用 v3/unified_evaluators/runtime.py。 |
| [evaluator/measure_backend.py](evaluator/measure_backend.py) | 历史 V2 分数复算入口，依赖旧 scoring_v2 包；当前 V3 不调用它。 |
| [evaluator/evaluate_raw_legacy.py](evaluator/evaluate_raw_legacy.py) | 当前 V3 调用的原始物理提取器；同目录保留配置、审计与回归工具。 |
| [evaluator/requirements.txt](evaluator/requirements.txt) | 本题物理后端依赖；V3 一致性模型依赖另见公共环境说明。 |
| [scripts/run_eval.sh](scripts/run_eval.sh) | 原 metadata 批量入口；样本清单已删除，批量运行方式尚需适配。 |

`output_videos/<model>/` 是运行输入目录，`eval_results/` 是题目内历史结果或显式选择的输出目录。源码仓库未附带 MP4、模型权重和完整评测结果；这些运行资产不作为现存文件链接。

## 测量方法

- M1：测量球心累计位移、标记转角和球半径，计算积分形式的 v/(omega*R)-1。
- M2：由球心速度与 omega*R 的差估计瞬时接触点速度残差。

## 环境与工作目录

先按照 [V3 公共环境说明](../../README.md#运行环境) 配置一致性模型及所选物理后端。以下代码块从仓库根目录进入本题，后续命令均在题目根目录执行：

```bash
cd v3/g3/P6
python -m pip install -r evaluator/requirements.txt
export EVALUATOR_PYTHON="$(command -v python)"
python evaluator/evaluate.py --help
```

`EVALUATOR_PYTHON` 指定批量脚本使用的解释器；未设置时脚本使用 `v3/.venv/bin/python`。该虚拟环境需自行准备。本题 requirements 只定义物理测量依赖，公共环境还需满足 V3 一致性模型的运行条件。

## 单视频评测

准备输入视频后，在题目根目录运行。以下命令显式指定标准首帧和视频续写提示词；首帧生成提示词不用作视频续写 prompt。其他来源的视频应传入对应的首帧、实际续写文本、seed 和模型信息。

```bash
python evaluator/evaluate.py \
  --video output_videos/minimax_h3/sample_00.mp4 \
  --image first_frame.png \
  --video_prompt_file video_prompts/P6__results_v1__P6_gpt_01_modern_seed42.txt \
  --output ../../results/g3/P6/eval_results/minimax_h3/result_sample_00.json
```

本例首帧是 `first_frame.png`。自定义视频可以使用 `--image`、`--video_prompt_file` 或 `--video_prompt` 显式传入关联；一致性参数通过 `--consistency-*` 设置，详见 `--help`。

## 批量评测

`scripts/run_eval.sh` 仍调用依赖 `data/metadata.json` 的公共批量执行器。该清单已删除，当前不能直接使用原登记样本批量命令；请先按上面的单视频入口逐个运行，批量目录扫描接口待后续适配。

默认新结果位于 `v3/results/g3/P6/eval_results/<model>/result_<sample_id>.json`，同目录有 `batch_summary.json`；本题批量日志位于 `eval_results/run_logs/`。`--output-root` 下始终追加 `g3/P6/eval_results/<model>/`。

## 结果语义

一致性分 C 达到默认阈值 0.8 后才运行物理后端；总分为 `0.15 × C + 0.85 × P`。P 是已定义物理指标的纯物理分等权平均；定义但提取失败的指标贡献 0，未定义指标不进入分母。一致性未通过时只保留 `0.15 × C`；一致性评测出错时总分为 null。

`metrics.M1/M2.metric` 保存纯物理分，原始测量在 `verbose.M*.raw_metric`，总分读取 `verbose.M1._scoring_summary.score`。调试目录含一致性记录、后端日志和可用测量证据；是否执行物理流程见 `physics_attempted`。单视频退出码 0 表示完成且有量可测，1 表示有效拒绝或不可测，2 表示运行错误；批量脚本会在汇总中保留单视频状态。

旧 V2 复算、旧 `run_batch`/`batch.py` 及带 `legacy` 的脚本保留作历史资料，不作为本文运行入口；它们可能依赖未随源码交付的旧包或目录。
