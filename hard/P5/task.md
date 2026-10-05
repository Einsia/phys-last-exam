# P5 · 等质量钢球正碰

难度：困难；本轮任务排名：34/40；平均最终综合分：0.157530。

任务 ID：`P5`。

水平表面上两个等大小、等质量球正碰：左侧红球匀速向右撞击初始静止的蓝球。应呈现等质量弹性碰撞后的速度传递，运动始终保持水平。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：(v₁碰后+v₂碰后)/v₁碰前−1，衡量等质量体系的动量关系；无量纲。
- 本题只定义 M1。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, side view. Two identical balls of equal size sit on a level horizontal surface. The left (red) ball slides to the right at a steady speed and strikes the right (blue) ball, which is initially at rest. After the head-on collision the balls behave as equal-mass elastic spheres. The motion is purely horizontal and stays inside the frame. The camera does not move, pan, or zoom. Plain flat background, only the two balls on the surface, nothing else enters the frame.
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

指标定义依据：`v4/g1/P5/evaluator/utils/physeval/tasks/p5.py:49`；`v4/unified_evaluators/contract.py:100`；`v4/g1/P5/evaluator/utils/physeval/tasks/p5.py:58`。
