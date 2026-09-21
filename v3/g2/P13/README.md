<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P13 Plane Mirror

M1 and M2 are `[0,1]` scores where higher is better. M1 is
`clip(1-|theta_i-theta_r|/90,0,1)` and M2 is
`clip(1-incidence_point_error_norm,0,1)`. The raw angle difference in degrees
and normalized incidence-point geometry error remain in `verbose.measurements`. The evaluator
uses the OpenCV Hough-line extractor and rejects clips where the mirror, both rays,
or a stable locked-off view cannot be measured.

`evaluator/evaluate.py` is a standalone implementation. It imports no code from
other tasks or from the parent repository. Install its complete dependency set with
`python -m pip install -r evaluator/requirements.txt`.

Single video:

```text
python P13/evaluator/evaluate.py --video VIDEO.mp4 --task_id P13 \
  --output P13/eval_results/json/sample_00.json
```

The JSON contains `task_id`, `video_path`, `image_path`, `seed`, `model`, nested
`metrics.M1/M2` objects (`extract_success` and `metric`), and
the intermediate measurements/debug paths under `verbose`.

Batch evaluation of all MP4 files in the default H3 directory:

```text
bash scripts/run_eval.sh minimax_h3
```

Pass a video directory, output directory, and model name as the first three
arguments. Each run writes JSON files under `eval_results/json`, audit artifacts
under `eval_results/debug/<sample_id>`, and `results.csv`.

Standard delivery paths are `first_frames/{gpt,simulation,real}/`,
`output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Use
`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh <model_name>`.
