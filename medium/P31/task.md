# P31 · 带电小球对称平衡

分类：7. 静电、磁学与电磁感应（Electrostatics, Magnetism, and Electromagnetic Induction）；原编号：`P28`。

难度：中等；本轮任务排名：20/40；平均最终综合分：0.351582。

任务 ID：`P31`。

两个相同带电球由等长绝缘线对称悬挂，彼此排斥并自由达到静止平衡。两个球和两根完整悬线保持清楚可见；最终应形成向外分开的对称平衡构型。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：稳定构型中两根悬线相对竖直方向的角度绝对差，单位为度。
- M2：两球相对装置中轴水平位移比的绝对对数 |ln(d左/d右)|，无量纲。
- 当前实现还检查向外分离、静止及悬线等长固定的可见构型；已观察但不满足构型时，两项物理分数为 0。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view. Two identical charged balls hang from two insulating threads of exactly equal length, arranged symmetrically from the same support. The balls repel each other and freely settle into a stable static configuration. Both balls and both complete threads remain clearly visible. The camera does not move, pan, or zoom. Plain background, no other objects.
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

指标定义依据：`v4/g2/P28/evaluator/measure_backend.py:378`；`v4/unified_evaluators/contract.py:155`。
