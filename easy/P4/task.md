# P4 · 弹跳高度衰减

难度：简单；本轮任务排名：9/40；平均最终综合分：0.574569。

任务 ID：`P4`。

一个球由固定电磁支架释放，竖直落到坚硬水平台面并连续反弹。生成要求至少四次清楚回跳，反弹峰高逐次降低；释放点、接触和各峰值均在画面内，台面及支架保持固定。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：相邻回跳的恢复系数一致性 mean(|√(h后/h前)−Δt后/Δt前|)，并在分数中结合恢复系数超过 1 的幅度；无量纲。
- M2：相邻峰高增加的平均相对幅度 mean(max(0,h后/h前−1))，并在可测时结合相邻恢复系数的变异系数；无量纲。
- 当前实现对可跟踪但仅单段运动、没有多次回跳的情况，另记录该结果，两项物理分数均为 0.1。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off, near-orthographic side camera, matching the input first frame.

Preserve the exact apparatus, ball, electromagnetic holder, hard horizontal impact plate, camera view, framing, and background from the input first frame. Exactly one ball is initially motionless at its existing release position, directly above the impact plate.

At the beginning of the shot, the existing electromagnetic holder releases the ball cleanly without moving, falling, or following the ball. The ball then falls vertically under gravity, strikes the hard plate, and makes at least four clear consecutive bounces, with each rebound reaching a lower height than the previous one. All impacts occur at approximately the same horizontal position, with no noticeable sideways drift.

The complete motion remains visible inside the frame: the initial release point, every impact, every rebound apex, and the full ball must never be cropped or occluded. The ball remains exactly one intact sphere throughout the shot, without duplication, disappearance, morphing, or permanent deformation.

The holder, support frame, impact plate, table, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, ghost images, motion trails, trajectory lines, arrows, measurements, annotations, or new objects. End shortly after the ball completes its fourth clearly visible rebound arc and returns to the plate.
```

## 评测入口

当前入口为 `evaluator/evaluate.py`。视频由外部路径传入，首帧参数为 `--image first_frame.png`，提示词文件参数为 `--prompt`。现行测量后端仍使用题内场景标定；输入视频须与所选首帧标定对应。

指标定义依据：`v4/unified_evaluators/physics.py:22`；`v4/g3/P4/evaluator/evaluate_raw_legacy.py:102`；`v4/g3/P4/evaluator/evaluate_raw_legacy.py:651`。
