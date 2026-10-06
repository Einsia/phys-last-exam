# P19 · 点光源投影共点

分类：4. 光学与投影几何（Optics and Projective Geometry）；原编号：`P14`。

难度：中等；本轮任务排名：19/40；平均最终综合分：0.357083。

任务 ID：`P19`。

固定镜头中，地面附近的一个小点光源照亮分散放置的四根竖杆。四杆、杆底和各自清晰的长阴影同时可见，光源及杆均不移动。各阴影轴线反向延长后应交于同一光源投影位置。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：各阴影反向延长线到共同交点的均方根距离/画面对角线长度；无量纲。
- M2：独立线对估计的光源位置在 x、y 方向标准差的平均值/画面对角线长度；无量纲。当前实现为标准差离散程度，而非方差。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Task: P19_point_source_four_shadows.
Start exactly from the supplied first frame: a pale matte floor panel, one small bare white bulb standing upright at the centre close to the floor as the only light, and four slender upright rods - red, blue, yellow, green - spaced well apart around the bulb, each casting one long crisp dark shadow radially outward away from the bulb.

Video action: Locked-off static camera with a clear view of the horizontal ground plane. A single compact point light source illuminates four vertical rods positioned at different locations on the ground, producing four distinct and clearly visible shadows. Every rod, rod base, and shadow tip is visible simultaneously. The camera does not move, pan, or zoom. Plain background, no other objects.
Hold this exact geometry. The bulb stays lit at constant brightness and does not move. All four rods stay standing upright at their starting positions and do not fall, slide, or wobble. All four shadows stay long, sharp, clearly darker than the pale floor beside them, separate from one another, and pointing radially away from the bulb, exactly as in the first frame.

Hard negative constraints: no second light source, no moving or flickering light, no additional or disappearing rods or shadows, no shadows fading out or losing contrast, no person or hand, no labels, numbers, arrows, watermark, logo, cartoon styling, no camera pan, zoom, shake or reframing, no cropped rods or shadow tips.
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

指标定义依据：`v4/g4/P14/evaluator/utils/physeval/tasks/p14.py:48`；`v4/unified_evaluators/contract.py:100`；`v4/g4/P14/evaluator/utils/physeval/tasks/p14.py:56`。
