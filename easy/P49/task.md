# P49：大小球在黏性液体中的终端沉降

难度：简单；本轮任务排名：6/40；平均最终综合分：0.642551。

## 场景与目标

固定正视镜头观察两个材质相同、半径明显不同的球在同一深甘油槽内竖直下落。两球在触底前都需保持可见足够长时间，形成可辨识的稳定终端速度区间，用于比较终端速度与球半径平方的关系。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, front view. Two spherical balls made of exactly the same material but with clearly different radii are released into the same deep transparent tank filled with glycerin. Both balls fall vertically through the liquid and remain visible long enough to reach clear steady terminal-speed regimes before reaching the bottom. The camera does not move, pan, or zoom. Plain background, no other objects.
```

## M1 物理指标

M1 分别测量大球、小球的半径 `r_large`、`r_small`（px）及终端速度 `v_large`、`v_small`（px/s）。

`E = |(v_large/v_small) / (r_large/r_small)² - 1|`

`M1 = 1 / (1 + E / a)`，默认 `a=1.0`。

比值、误差及分数均无量纲。每个球独立选择触底前最长的合格非零终端速度窗口，同长时取较早窗口；默认半径比至少 1.1。相同材质与流体是任务前提；未进行物理尺寸和流体参数标定，因此不将像素速度当作 SI 速度，也不声称直接测得低雷诺数。

实现依据：[tasks.py](evaluator/_shared/refined_evaluators/tasks.py) 的 `p49`。

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
