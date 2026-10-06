# P7：链条悬垂静态形状

分类：2. 滚动、摩擦与刚体静力学（Rolling, Friction, and Rigid-Body Statics）；原编号：`P44`。

难度：简单；本轮任务排名：5/40；平均最终综合分：0.708567。

## 场景与目标

连续链条两端固定于等高支点，从首帧的非平衡形状释放。链条在重力下振动并逐渐静止，保持链节连续、总长度和支点不变；检验最终轮廓是否符合悬链线及支点等高关系。

## 输入与评测

首帧：[first_frame.png](first_frame.png)。生成提示词：[prompt.txt](prompt.txt)。视频为外部评测输入，应与所用首帧和提示词对应。

在本题目录、已安装项目运行依赖的 Python 环境中执行：

```sh
python evaluator/evaluate.py \
  --video /absolute/path/video.mp4 \
  --image first_frame.png --prompt prompt.txt \
  --output /absolute/path/result.json
```

## 当前指标

下表 P 为各指标的纯物理分，范围为0–1。原始量与物理分分别保留；观测不足时原始量为 null，适用的未观察到现象规则可计物理分0。完整输出另含一致性判断。

| 指标 | 定义与公式 | 单位 |
| --- | --- | --- |
| M1：悬链线拟合误差 | e=轮廓拟合RMSE/实测下垂尺度；P=1/(1+abs(e)/0.10)。 | e 无量纲；几何长度为像素。 |
| M2：支点等高误差 | e=支点高度误差/水平跨度；P=1/(1+abs(e)/0.02)。 | e 无量纲；几何长度为像素。 |

当前公式依据：`v4/g5/P44/evaluator/measure_backend.py:151`；`v4/g5/P44/evaluator/measure_backend.py:179`；`v4/g5/P44/evaluator/measure_backend.py:152`。

## 生成提示词原文

```text
A single continuous real-time shot from a locked-off, exact front near-orthographic camera, matching the input first frame.

The input first frame captures the instant immediately after a temporary constraint has been removed. No hand, tool, magnet, or temporary support remains visible. The first and last links stay fixed to the two equal-height anchors, while the rest of the chain immediately begins moving freely under gravity from its existing non-equilibrium shape.

Preserve the existing support frame, the two equal-height fixed anchors, the complete chain, its material and link structure, the camera view, framing, and background. The first and last links remain fixed to their existing anchors throughout the shot.

The chain swings and oscillates naturally as the motion gradually damps, then settles into one smooth, stable, deep hanging shape. The chain remains one continuous flexible chain of rigid interlocked links, with unchanged total length. No link detaches, stretches, fuses, duplicates, disappears, or changes material.

Keep the complete chain, both anchors, and the full motion visible inside the frame. The support frame, anchors, background, and camera remain stationary. No camera movement, cuts, slow motion, time jumps, hands, people, added supports, external forces, text, formulas, plotted curves, arrows, or annotations.

End after the chain has settled and remained essentially motionless for a short moment.
```
