# P2 · 自由落体

分类：1. 平动与碰撞（Translational Motion and Collisions）；原编号：`P1`。

难度：中等；本轮任务排名：27/40；平均最终综合分：0.266128。

任务 ID：`P2`。

固定侧视镜头中，一个小球从静止释放并沿竖直方向下落。完整可测下落段保持在画面内，片段结束前不触地。应呈现重力作用下速度逐渐增大的自由落体。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：等时间间隔竖直速度增量的变异系数 CV(Δv_y)，无量纲。
- M2：连续三等分时段位移比相对 1:3:5 的均方根相对偏差，无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, side view. A small ball is released from rest and falls straight downward under gravity. The entire measured fall stays inside the frame, and the ball does not reach or touch the ground during the clip. The camera does not move, pan, or zoom. Plain flat background, no other objects, nothing enters the frame.
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

指标定义依据：`v4/g1/P1/evaluator/utils/physeval/tasks/p1.py:129`；`v4/unified_evaluators/contract.py:100`；`v4/g1/P1/evaluator/utils/physeval/tasks/p1.py:134`。
