# P19 · 连通器液面平衡

难度：简单；本轮任务排名：1/40；平均最终综合分：0.964831。

任务 ID：`P19`。

透明 U 形连通管的两臂粗细不同，其中盛有同一种连续液体。液体从受扰动状态自由恢复到静止平衡，两侧液面始终可见，液体总量不变。平衡时两臂液面应等高。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：末段左右液面高度中位数之差，除以参考高度；无量纲。
- M2：末段两侧液面线性变化速度绝对值的较大值，除以画面高度；单位为每帧，即 frame⁻¹。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view. A transparent U-shaped tube has two vertical arms with clearly different diameters and contains the same continuous liquid. The liquid is initially disturbed and then freely settles back to static equilibrium. You need to display how the liquid in the U-shaped tube reaches a state of equilibrium. The total amount of liquid remains constant throughout the entire process, with no liquid added, removed, appearing, or disappearing. Both liquid surfaces remain clearly visible throughout the process. The camera does not move, pan, or zoom. Plain background, no other objects.
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

指标定义依据：`v4/g2/P19/evaluator/measure_backend.py:350`；`v4/unified_evaluators/contract.py:155`。
