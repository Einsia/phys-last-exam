# P16 · 刚体共线点交比

分类：4. 光学与投影几何（Optics and Projective Geometry）；原编号：`P16`。

难度：简单；本轮任务排名：4/40；平均最终综合分：0.710831。

任务 ID：`P16`。

一根带四个不同颜色共线标记的刚直杆，上端接触竖墙、下端接触水平地面。杆从静止释放，上端沿墙下滑、下端沿地面外移，发生平移和转动；标记固定、杆保持直线。透视投影中的四点交比应保持不变。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：有效帧中四标记交比的变异系数 std(χ)/|mean(χ)|，无量纲。
- M2：每帧四标记共线拟合残差/标记跨度，再跨有效帧取平均；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing rigid straight rod, the four clearly separated coloured markers fixed along the same straight line, the vertical wall, the horizontal floor, the camera view, framing, and background.

The rod is initially motionless in its existing inclined position, with its upper end in contact with the vertical wall and its lower end in contact with the horizontal floor. After a brief still moment, the rod is released from rest and slides under gravity within the same vertical plane. Its upper end moves downward along the wall while its lower end moves horizontally away from the wall along the floor. Both ends remain in contact with their respective surfaces throughout the visible motion.

The rod translates and rotates smoothly as one rigid body. It remains perfectly straight and unchanged in length. All four markers remain permanently fixed at their original positions on the rod, remain collinear, and stay clearly visible throughout the motion. The markers must not slide, detach, swap positions, duplicate, disappear, or change shape or colour.

Keep the complete rod, both endpoints, all four markers, and the wall-floor contact regions visible inside the frame. End after the rod has undergone a clearly visible combination of translation and rotation, before either endpoint leaves its corresponding surface.

The wall, floor, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No new objects, trajectories, guide lines, arrows, measurements, labels, or annotations.
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

指标定义依据：`v4/g4/P16/evaluator/utils/physeval/tasks/p16.py:48`；`v4/unified_evaluators/contract.py:100`；`v4/g4/P16/evaluator/utils/physeval/tasks/p16.py:55`。
