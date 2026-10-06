# P3 · 互余角抛射

难度：中等；本轮任务排名：29/40；平均最终综合分：0.222125。

任务 ID：`P3`。

两个固定发射器分别以 30° 和 60° 仰角，同时向右发射初速度大小相同的两个球。每球保持独立飞行，在自己的水平通道上落到与发射时等高的位置，完整飞行和首次落点可见。理想重力抛射中，两球射程应相同。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：R₃₀/R₆₀−1，比较互余角射程；无量纲。
- M2：组合两条轨迹各自按球直径归一化的抛物线拟合残差，以及初速度大小的相对偏差；各分量均无量纲，归一化后的物理分数取几何均值。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off, slightly elevated near-orthographic side camera, matching the input first frame.

Preserve the exact two-lane parallel layout and all existing apparatus. Exactly two identical, rigidly mounted launchers release exactly two identical balls simultaneously toward the right with exactly the same initial speed. One launcher is oriented at 30 degrees above the horizontal, and the other is oriented at 60 degrees above the horizontal. Preserve each launcher's existing angle and lane assignment from the input first frame. Do not swap, rotate, move, or reorder either launcher.

After release, both balls move freely through the air under gravity. Each ball remains a distinct rigid sphere and lands on its own corresponding horizontal landing lane. At first landing contact, the center of each ball returns to the same vertical level as its center at launch. Both complete flights and both first landing points remain visible inside the frame.

The launchers, lanes, supports, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No new objects or people. No trajectory lines, annotations, motion trails, ghost images, duplicated balls, disappearing balls, or collisions between the two balls. End shortly after both balls make their first landing contact.
```

## 评测入口

当前入口为 `evaluator/evaluate.py`。视频由外部路径传入，首帧参数为 `--image first_frame.png`，提示词文件参数为 `--prompt`。现行测量后端仍使用题内场景标定；输入视频须与所选首帧标定对应。

指标定义依据：`v4/unified_evaluators/physics.py:22`；`v4/g3/P3/evaluator/evaluate_raw_legacy.py:735`；`v4/g3/P3/evaluator/evaluate_raw_legacy.py:1`。

## 当前场景标定

当前测量后端根据文件名选择已有首帧标定。使用本题唯一首帧生成的视频应命名为 `P3_gpt_01_modern_seedN.mp4`（N 为任意整数），并保持与配置相同的 1344×768 测量画布；视频可位于题包外。`--sample-id` 不替代后端的文件名匹配。

首帧到测量画布的变换仍为 PIL LANCZOS 缩放到 1344×768。根目录 `first_frame.png` 已于 2026-10-06 更新为新场景，不能沿用此前与旧标定输入图像素一致的结论；现有后端标定尚未按新首帧更新，评测新场景前需重新核对并更新标定。
