# P3 — 30° 与 60° 斜抛

本题属于 G3，使用重构后的 V3 文件布局。单视频入口为 `evaluator/evaluate.py`，批量入口为 `scripts/run_eval.sh`；两者都执行一致性门控和物理评测。请保留完整 `v3/` 目录，公共实现通过相对路径加载。

目录归档规则见 [重构说明](../../RESTRUCTURE_NOTES.md)，当前评分与结果语义见 [SCORING_V3.md](../../SCORING_V3.md)。

## 输入与目录

`data/metadata.json` 的 `samples` 登记了 24 个输入关联。`video_path`、`source_video_path`、`image_path` 和 `annotation` 均相对于本题根目录解析；`video_prompt` 保存实际生成提示词，不因 metadata 移入 `data/` 而改变路径基准。

| 文件或目录 | 用途 |
| --- | --- |
| [data/metadata.json](data/metadata.json) | 唯一正式样本清单，提供视频、首帧、prompt、模型、seed 与哈希关联。 |
| [video_prompts/](video_prompts/) | 逐样本历史 prompt 文本；精确关联以 metadata 为准。 |
| [first_frames/gpt/](first_frames/gpt/) | GPT 首帧及生成 prompt。 |
| [first_frames/simulation/](first_frames/simulation/) | 仿真首帧、来源记录及可用生成代码。 |
| [evaluator/evaluate.py](evaluator/evaluate.py) | 当前 V3 单视频入口，调用 v3/unified_evaluators/runtime.py。 |
| [evaluator/measure_backend.py](evaluator/measure_backend.py) | 历史 V2 分数复算入口，依赖旧 scoring_v2 包；当前 V3 不调用它。 |
| [evaluator/evaluate_raw_legacy.py](evaluator/evaluate_raw_legacy.py) | 当前 V3 调用的原始物理提取器；同目录保留配置、审计与回归工具。 |
| [evaluator/requirements.txt](evaluator/requirements.txt) | 本题物理后端依赖；V3 一致性模型依赖另见公共环境说明。 |
| [scripts/run_eval.sh](scripts/run_eval.sh) | 读取 data/metadata.json，只选择本题与指定生成模型的登记样本。 |

`output_videos/<model>/` 是运行输入目录，`eval_results/` 是题目内历史结果或显式选择的输出目录。源码仓库未附带 MP4、模型权重和完整评测结果；这些运行资产不作为现存文件链接。

## 测量方法

- M1：分别测量两条完整抛物轨迹的射程，计算 R30/R60 - 1。
- M2：分别拟合 30°、60° 轨迹，按球直径归一化残差，并比较初速度。

## 环境与工作目录

先按照 [V3 公共环境说明](../../README.md#运行环境) 配置一致性模型及所选物理后端。以下代码块从仓库根目录进入本题，后续命令均在题目根目录执行：

```bash
cd v3/g3/P3
python -m pip install -r evaluator/requirements.txt
export EVALUATOR_PYTHON="$(command -v python)"
python evaluator/evaluate.py --help
```

`EVALUATOR_PYTHON` 指定批量脚本使用的解释器；未设置时脚本使用 `v3/.venv/bin/python`。该虚拟环境需自行准备。本题 requirements 只定义物理测量依赖，公共环境还需满足 V3 一致性模型的运行条件。

## 单视频评测

以下例子使用 metadata 中登记的首个样本；先按清单恢复对应视频及其原始文件名别名。入口自动读取首帧、实际 prompt、seed 和模型关联，并核对文件哈希。

```bash
python evaluator/evaluate.py \
  --video output_videos/minimax_h3/sample_00.mp4 \
  --output ../../results/g3/P3/eval_results/minimax_h3/result_sample_00.json
```

本例首帧是 `first_frames/gpt/P3_gpt_01_modern.png`。自定义视频可以使用 `--image`、`--video_prompt_file` 或 `--video_prompt` 显式传入关联；一致性参数通过 `--consistency-*` 设置，详见 `--help`。

本题当前 V3 使用原视频关联的坐标缓存，并重新计算物理量。需要恢复 `eval_results/debug/<source_video_stem>/` 中的 `tracks_raw.npz` 及缓存校验文件，同时保留 `evaluator/config.yaml`。仅提供一个未知视频不足以运行当前缓存路径；真实重新跟踪需使用原提取器及其外部模型环境。

## 批量评测

```bash
bash scripts/run_eval.sh minimax_h3
```

脚本按 `data/metadata.json` 的样本关联运行，不接受旧版视频目录、结果目录等位置参数。可加 `--samples sample_00` 只运行单个登记样本，或用 `--output-root`、`--log-dir` 设置输出基准目录；任意输入目录评测不属于该脚本接口。

默认新结果位于 `v3/results/g3/P3/eval_results/<model>/result_<sample_id>.json`，同目录有 `batch_summary.json`；本题批量日志位于 `eval_results/run_logs/`。`--output-root` 下始终追加 `g3/P3/eval_results/<model>/`。

## 结果语义

一致性分 C 达到默认阈值 0.8 后才运行物理后端；总分为 `0.15 × C + 0.85 × P`。P 是已定义物理指标的纯物理分等权平均；定义但提取失败的指标贡献 0，未定义指标不进入分母。一致性未通过时只保留 `0.15 × C`；一致性评测出错时总分为 null。

`metrics.M1/M2.metric` 保存纯物理分，原始测量在 `verbose.M*.raw_metric`，总分读取 `verbose.M1._scoring_summary.score`。调试目录含一致性记录、后端日志和可用测量证据；是否执行物理流程见 `physics_attempted`。单视频退出码 0 表示完成且有量可测，1 表示有效拒绝或不可测，2 表示运行错误；批量脚本会在汇总中保留单视频状态。

旧 V2 复算、旧 `run_batch`/`batch.py` 及带 `legacy` 的脚本保留作历史资料，不作为本文运行入口；它们可能依赖未随源码交付的旧包或目录。
