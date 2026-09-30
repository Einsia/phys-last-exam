# P36 — 磁体与对照物运动

本题属于 G5，使用重构后的 V3 文件布局。单视频入口为 `evaluator/evaluate.py`，执行一致性门控和物理评测。`scripts/run_eval.sh` 保留原批量接口，目前仍依赖已删除的样本清单。请保留完整 `v3/` 目录，公共实现通过相对路径加载。

目录归档规则见 [重构说明](../../RESTRUCTURE_NOTES.md)，当前评分与结果语义见 [SCORING_V3.md](../../SCORING_V3.md)。

## 输入与目录

本题标准首帧位于根目录的 `first_frame.png`。首帧生成提示词与视频续写提示词分别归档，评测时应使用与输入视频对应的实际续写文本。题目 `data/` 和 `reports/` 已按要求删除，不再附带 metadata 样本清单。

| 文件或目录 | 用途 |
| --- | --- |
| [first_frame.png](first_frame.png) | 本题选定的标准首帧。 |
| [prompts/first_frame.txt](prompts/first_frame.txt) | 生成首帧图片所用的原始提示词。 |
| [prompts/video.txt](prompts/video.txt) | 视频续写 prompt。 |
| [first_frames/simulation/](first_frames/simulation/) | 仿真首帧、来源记录及可用生成代码。 |
| [evaluator/evaluate.py](evaluator/evaluate.py) | 当前 V3 单视频入口，调用 v3/unified_evaluators/runtime.py。 |
| [evaluator/measure_backend.py](evaluator/measure_backend.py) | 本题 OpenCV 测量后端，由 V3 入口在一致性通过后调用。 |
| [evaluator/requirements.txt](evaluator/requirements.txt) | 本题物理后端依赖；V3 一致性模型依赖另见公共环境说明。 |
| [scripts/run_eval.sh](scripts/run_eval.sh) | 原 metadata 批量入口；样本清单已删除，批量运行方式尚需适配。 |

`output_videos/<model>/` 是运行输入目录，`eval_results/` 是题目内历史结果或显式选择的输出目录。源码仓库未附带 MP4、模型权重和完整评测结果；这些运行资产不作为现存文件链接。

## 测量方法

- M1：从两物体运动轨迹检测到达事件，比较运动时间比。
- M2：由轨迹估计运动速度，计算对应速度比。

## 环境与工作目录

先按照 [V3 公共环境说明](../../README.md#运行环境) 配置一致性模型及所选物理后端。以下代码块从仓库根目录进入本题，后续命令均在题目根目录执行：

```bash
cd v3/g5/P36
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
  --video_prompt_file prompts/video.txt \
  --output ../../results/g5/P36/eval_results/minimax_h3/result_sample_00.json
```

本例首帧是 `first_frame.png`。自定义视频可以使用 `--image`、`--video_prompt_file` 或 `--video_prompt` 显式传入关联；一致性参数通过 `--consistency-*` 设置，详见 `--help`。

需要人工 ROI 时，直接调用首帧标注脚本：

```bash
python scripts/annotate_roi.py --image first_frame.png --output evaluator/roi.json
```

交互标注需要支持 GUI 的 OpenCV；无 GUI 时使用脚本的 `--coords` 和 `--no-gui`。`run_eval.sh` 不提供 `--annotate` 模式。

## 批量评测

`scripts/run_eval.sh` 仍调用依赖 `data/metadata.json` 的公共批量执行器。该清单已删除，当前不能直接使用原登记样本批量命令；请先按上面的单视频入口逐个运行，批量目录扫描接口待后续适配。

默认新结果位于 `v3/results/g5/P36/eval_results/<model>/result_<sample_id>.json`，同目录有 `batch_summary.json`；本题批量日志位于 `eval_results/run_logs/`。`--output-root` 下始终追加 `g5/P36/eval_results/<model>/`。

## 结果语义

一致性分 C 达到默认阈值 0.8 后才运行物理后端；总分为 `0.15 × C + 0.85 × P`。P 是已定义物理指标的纯物理分等权平均；定义但提取失败的指标贡献 0，未定义指标不进入分母。一致性未通过时只保留 `0.15 × C`；一致性评测出错时总分为 null。

`metrics.M1/M2.metric` 保存纯物理分，原始测量在 `verbose.M*.raw_metric`，总分读取 `verbose.M1._scoring_summary.score`。调试目录含一致性记录、后端日志和可用测量证据；是否执行物理流程见 `physics_attempted`。单视频退出码 0 表示完成且有量可测，1 表示有效拒绝或不可测，2 表示运行错误；批量脚本会在汇总中保留单视频状态。

旧 V2 复算、旧 `run_batch`/`batch.py` 及带 `legacy` 的脚本保留作历史资料，不作为本文运行入口；它们可能依赖未随源码交付的旧包或目录。
