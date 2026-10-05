# P18 · 静止液面与重力方向垂直

难度：中等；本轮任务排名：23/40；平均最终综合分：0.323625。

任务 ID：`P18`。

透明容器内静止液面的旁边，一个无绳裸钢球从固定电磁支架释放并竖直下落。完整可测下落段与平液面同时可见，球触地前结束。重力加速度方向应垂直于液面，球轨迹应符合恒加速度运动。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：球加速度方向与液面切向夹角−90°，单位为度。
- M2：组合液面直线拟合 RMS 与球轨迹恒加速度拟合 RMS；两项原始单位均为像素，分别归一化后的物理分数取几何均值。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing transparent container, clear still liquid, flat horizontal free surface, steel ball, stationary release apparatus, camera view, framing, and background.

Exactly one small dense steel ball is initially motionless at its existing position beside the container, held by the existing stationary electromagnetic holder. The ball is one bare, smooth, rigid sphere with no string, hook, ring, cable, rod, clip, cap, or other object attached to it.

At the beginning of the shot, the existing holder releases the ball cleanly and remains completely fixed. The ball separates fully from the holder and falls vertically downward under gravity as one bare sphere. No part of the holder detaches, falls, or follows the ball. The ball does not remain tethered, swing like a pendulum, or drift noticeably sideways.

Keep the complete measured fall inside the frame and end before the ball reaches or touches the ground. The full ball and the flat liquid surface remain clearly visible simultaneously throughout the measured interval. The liquid surface remains still and horizontal.

The container, liquid, release apparatus, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, duplicated objects, motion trails, annotations, or newly appearing objects.
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

指标定义依据：`v4/g4/P18/evaluator/utils/physeval/tasks/p18.py:33`；`v4/unified_evaluators/contract.py:100`；`v4/g4/P18/evaluator/utils/physeval/tasks/p18.py:40`。
