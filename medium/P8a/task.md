# P8a · 小角度单摆等时性

难度：中等；本轮任务排名：17/40；平均最终综合分：0.417340。

任务 ID：`P8a`。

同一固定横梁上悬挂两个等长单摆，红球初始角度较小、蓝球较大。两摆同时从静止释放，分别完成多次往返；悬点和摆长保持不变。小角度条件下，两摆完整周期应近似相等。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：T左/T右−1，比较两摆周期；无量纲。
- M2：两摆 θ(t) 的正弦拟合均方根残差，分别以弧度计量；左右残差归一化后的物理分数取几何均值。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame, which shows two pendulums hanging from the same horizontal bar: a red ball on the left and a blue ball on the right, on strings of exactly the same length, each already pulled aside and held at rest.

Both pendulums are let go at the very same instant. From the first moment of the clip the red ball and the blue ball are both already moving: each one immediately leaves the position it holds in the first frame, swings down through the lowest point, up to the far side, and back, and each keeps swinging back and forth for several complete cycles until the clip ends. Neither ball is ever stationary, neither ball stays parked at its starting position, and neither ball starts later than the other. The red ball swings through a smaller arc than the blue ball because it was released from a smaller angle, but both take the same time to complete one full swing, since their strings are the same length. Both strings stay straight and taut and keep their length; the bar and the two pivot points do not move.

Hard negative constraints: no frozen, still or motionless ball; no ball that stays hanging at its first-frame position while the other swings; no delayed or staggered release; no change of string length; no stretching, bending or slack string; no ball leaving its string; no collision between the two balls; no camera pan, zoom, shake or reframing; no hand, person, arrow, label, number, ruler, text or watermark; nothing else enters the frame.
```

## 评测入口

在本题目录中执行，输入和输出视频路径均可放在题包外：

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```

指标定义依据：`v4/g1/P8a/evaluator/utils/physeval/tasks/p8a.py:28`；`v4/unified_evaluators/contract.py:100`；`v4/g1/P8a/evaluator/utils/physeval/tasks/p8a.py:34`。
