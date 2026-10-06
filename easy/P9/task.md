# P9 · 单摆周期与摆长关系

难度：简单；本轮任务排名：14/40；平均最终综合分：0.479792。

任务 ID：`P9`。

两个相同摆球悬挂在同高固定悬点上，摆长比为 1:2，从同侧近似相同小角度同时释放。每球至少完成一次完整往返，悬线长度不变、完整摆动范围可见。周期平方之比应等于摆长之比。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：|(T短/T长)²/(L短/L长)−1|，无量纲。
- M2：短摆和长摆各自的周期变异系数；均无量纲，归一化后的物理分数取几何均值。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off, near-orthographic frontal camera, matching the input first frame.

Preserve the exact support frame, common overhead beam, two fixed pivot points, exactly two strings, exactly two identical spherical bobs, camera view, framing, materials, and background from the input first frame. Preserve each pendulum’s existing position and lane assignment; do not swap or reorder the short and long pendulums.

Preserve the existing free-string length ratio of 1:2, the equal pivot height, the identical bob sizes and appearances, and the initial string directions shown in the input frame. Both pendulums are initially motionless on the same side of their equilibrium positions, at approximately the same initial angular displacement.

At the beginning of the shot, both pendulums are released simultaneously from rest without any visible hand, push, added impulse, or newly appearing release mechanism. Each bob swings freely back and forth under gravity about its own fixed pivot and remains within its own separate vertical swing plane.

Each string remains taut, straight, attached to its original pivot and bob, and unchanged in length throughout the shot. Each bob remains attached to the end of its original string. The strings must not stretch, bend, detach, cross, merge, switch bobs, or change length. The two bobs must not collide or pass into each other’s swing lanes.

Continue the shot long enough for each pendulum to complete at least one full oscillation, from its initial side to the opposite turning point and back to its initial side. Keep every turning point and the complete swing arc of both pendulums visible. Do not skip, merge, or conceal any turning point.

The support frame, beam, pivots, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional pendulums, strings, bobs, or support parts. No auxiliary vertical lines, rods, catches, dots, guide marks, angle arcs, clocks, trajectory lines, arrows, measurements, annotations, ghost images, or motion trails. End shortly after both pendulums have each completed at least one full oscillation.
```

## 评测入口

当前入口为 `evaluator/evaluate.py`。视频由外部路径传入，首帧参数为 `--image first_frame.png`，提示词文件参数为 `--prompt`。现行测量后端仍使用题内场景标定；输入视频须与所选首帧标定对应。

指标定义依据：`v4/unified_evaluators/physics.py:22`；`v4/g3/P9/evaluator/evaluate_raw_legacy.py:1`。

## 当前场景标定

当前测量后端根据文件名选择已有首帧标定。使用本题唯一首帧生成的视频应命名为 `P9_gpt_01_modern_seedN.mp4`（N 为任意整数），并保持与配置相同的 1344×768 测量画布；视频可位于题包外。`--sample-id` 不替代后端的文件名匹配。

首帧到测量画布的变换仍为 PIL LANCZOS 缩放到 1344×768。根目录 `first_frame.png` 已于 2026-10-06 更新为新场景，不能沿用此前与旧标定输入图像素一致的结论；现有后端标定尚未按新首帧更新，评测新场景前需重新核对并更新标定。
