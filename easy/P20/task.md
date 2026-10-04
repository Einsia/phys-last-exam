# P20 · 冰柱漂浮吃水比

难度：简单；本轮任务排名：3/40；平均最终综合分：0.761774。

任务 ID：`P20`。

透明直壁容器内，一根均匀竖直淡水冰柱从首帧浸没深度释放，在重力和浮力下自由调整并稳定漂浮。冰柱完整、不融化、不触壁或触底，水线和完整冰柱始终可见。平衡时浸没高度比应接近冰水密度比 0.917。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：|稳定段浸没高度比中位数−0.917|，无量纲。
- M2：冰柱左右水线高度差绝对值/容器宽度的中位数；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view, matching the input first frame.

A uniform vertical rectangular block of pure freshwater ice is initially motionless at its existing immersion depth in fresh water inside a transparent straight-walled container.

After a brief still moment, the ice is released from its existing position and moves freely under gravity and buoyancy until its motion naturally settles. No support, attachment, or external force acts on the ice after release.

The ice remains one intact upright rectangular block and does not melt, deform, or touch the container walls or bottom. Keep the complete ice block, waterline, and container visible throughout the shot.

The container, background, and camera remain stationary. No camera movement, cuts, people, new objects, labels, arrows, or annotations.

End after the motion has settled for a short moment.
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

指标定义依据：`v4/g4/P20/evaluator/utils/physeval/tasks/p20.py:35`；`v4/unified_evaluators/contract.py:100`；`v4/g4/P20/evaluator/utils/physeval/tasks/p20.py:41`。
