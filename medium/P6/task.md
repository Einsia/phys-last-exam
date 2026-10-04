# P6 · 实心球纯滚动

难度：中等；本轮任务排名：18/40；平均最终综合分：0.359700。

任务 ID：`P6`。

斜坡上带非对称条带和偏心标记的球由挡板无推力释放，沿坡滚动并进入水平延伸段。球与表面持续接触，标记刚性附着并随球转动。应满足纯滚动关系 v=ωR，平动与转动均清楚可见。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：球心累计位移/(球半径×累计展开转角)−1，检验积分形式的纯滚动关系；无量纲。
- M2：接触点切向速度残差 |v−ωR| 的归一化平均绝对值；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off, near-orthographic side camera, matching the input first frame.

Preserve the exact inclined ramp, level runout, patterned ball, retractable gate, supports, camera view, framing, materials, and background from the input first frame. Exactly one ball is initially motionless at its existing position on the incline, in contact with the ramp and held by the existing gate. Preserve the ball’s exact asymmetric surface band and offset marker.

At the beginning of the shot, the existing gate withdraws cleanly out of the ball’s path without pushing, striking, or imparting an additional impulse to the ball. The gate then remains stationary outside the travel path. The ball moves downhill under gravity, visibly rotating as it travels along the incline, through the existing transition, and onto the level runout.

The asymmetric band and offset marker remain rigidly attached to the ball’s surface and rotate continuously with the same ball. The pattern must not slide across the surface, remain fixed relative to the camera, swim, morph, mirror, disappear, or change design. The ball remains in contact with the ramp and follows the existing surface without floating, bouncing, sinking into the ramp, or passing through it.

Keep the complete ball and its entire travel path visible inside the frame, including the initial position, the ball–ramp contact region, the full incline, the transition, and the level runout. The ball’s circular boundary, surface pattern, and contact region remain sharp enough to observe throughout the motion. Do not use motion blur that conceals the pattern or contact region.

The ramp, rails, supports, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, duplicated balls, ghost images, motion trails, rotation arrows, trajectory lines, measurements, annotations, or new objects. End while the ball is still completely visible on the level runout.
```

## 评测入口

当前入口为 `evaluator/evaluate.py`。视频由外部路径传入，首帧参数为 `--image first_frame.png`，提示词文件参数为 `--prompt`。现行测量后端仍使用题内场景标定；输入视频须与所选首帧标定对应。

指标定义依据：`v4/unified_evaluators/physics.py:22`；`v4/g3/P6/evaluator/evaluate_raw_legacy.py:1080`；`v4/g3/P6/evaluator/evaluate_raw_legacy.py:1084`。

## 当前场景标定

当前测量后端根据文件名选择已有首帧标定。使用本题唯一首帧生成的视频应命名为 `P6_gpt_01_modern_seedN.mp4`（N 为任意整数）；视频可位于题包外。标定以 1344×768 为基准，后端只按其原有规则处理尺度与宽高比。`--sample-id` 不替代后端的文件名匹配。
