# P2 · 斜抛运动

难度：困难；本轮任务排名：33/40；平均最终综合分：0.183322。

任务 ID：`P2`。

固定侧视镜头中，一个红球从地面以 45° 方向起抛，连续上升、经过最高点并落回同一地面高度。应呈现完整抛物轨迹、近似恒定的水平速度和恒定的竖直加速度。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：H/R−tan(θ)/4，其中 H 为最高点高度、R 为射程、θ=45°，无量纲。
- M2：组合水平速度 CV(v_x)、竖直速度增量 CV(Δv_y) 和抛物线拟合 RMS/R；三项均无量纲，分别归一化后的物理分数取几何均值。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, side view. A red ball on the ground launches immediately at 45 degrees above the horizontal, rises smoothly to the top of its arc, and falls back to the same ground level in one continuous trajectory. The whole arc stays inside the frame. The camera does not move, pan, or zoom. Plain flat background, no other objects, nothing enters the frame.
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

指标定义依据：`v4/g1/P2/evaluator/utils/physeval/tasks/p2.py:37`；`v4/unified_evaluators/contract.py:100`；`v4/g1/P2/evaluator/utils/physeval/tasks/p2.py:45`。
