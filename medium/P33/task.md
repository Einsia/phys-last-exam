# P33：闭合环与开口环的电磁起跳

分类：7. 静电、磁学与电磁感应（Electrostatics, Magnetism, and Electromagnetic Induction）；原编号：`P37`。

难度：中等；本轮任务排名：28/40；平均最终综合分：0.246987。

## 场景与目标

固定镜头观察并排的两套跳环装置，其中一个铝环闭合，另一个有缺口。两线圈同时通电，环沿各自铁芯自由运动；比较闭合环和开口环的升高量，观察闭合回路中的感应效应。线圈、铁芯、底座及导线保持固定，两环始终可辨。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical jumping-ring apparatuses sit side by side. Each copper coil, iron core, stand and wiring stay bolted in place at the same compact size and height and do not stretch, lift, tilt or translate. Each aluminium ring is a separate loose part around its own core. Both coils are switched on at the same instant; thereafter each ring is free to slide along its core. Both rings remain clearly visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 物理指标

M1 比较两环相对初始位置的最大升高量。每个高度先除以该环的初始外径，得到无量纲 `h_closed`、`h_open`；原始像素高度和峰值时刻另行保留。

`r = h_open / h_closed`

`M1 = clip((1 - r) / 0.2, 0, 1)`

默认 `margin=0.2`，即开口环相对升高不超过闭合环的 80% 时得 1。峰值必须有平台或下降证据；未完整观察到峰值不作为可靠高度。经连续观测确认的零参考高度可记 0，未定义的高度比保持空值。

实现依据：[tasks.py](evaluator/_shared/refined_evaluators/tasks.py) 的 `p33` 与 [observed_zero.py](evaluator/observed_zero.py)。

## 评测入口

从本题目录运行：

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use the configured evaluation environment. The evaluator automatically loads `first_frame_annotations.json` bundled with this task. It verifies the input-image hash, scales the coordinates to the video resolution, and checks correspondence with the decoded first frame before tracking. No per-video annotation is needed for the fixed task image when this check passes. Use `--annotation /absolute/path/to/video_annotations.json` only to override initialization for a different image or layout; custom video annotations retain their video/image hash checks.
