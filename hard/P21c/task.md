# P21c：淡水冰在盐水中融化

难度：困难；本轮任务排名：36/40；平均最终综合分：0.148438。

## 场景与目标

淡水冰漂浮于透明直壁容器中的盐水表面，逐渐融化并与盐水混合。全过程不增减或洒出液体；比较初末水位，并检查末帧是否仍有固体冰，目标行为是水位上升且冰完全融化。

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
| M1：融化后水位上升 | e=(y初−y末)/H初；当 y初−y末 大于测量不确定度时 P=1，否则 P=0。 | e 无量纲；y、H 为像素。 |
| M2：残余固体证据 | r=末段固体证据/初段固体证据；P=clip(1−r,0,1)。末帧检测到残冰时 P=0。 | r 与 P 无量纲。 |

当前公式依据：`v4/g5/P21c/evaluator/measure_backend.py:292`；`v4/g5/P21c/evaluator/measure_backend.py:235`；`v4/g5/P21c/evaluator/measure_backend.py:293`；`v4/g5/P21c/evaluator/measure_backend.py:299`。

## 生成提示词原文

```text
Locked-off static camera, front view, matching the input first frame.

A piece of freshwater ice initially floats freely in denser salt water inside a transparent straight-walled container.

The freshwater ice remains floating at the salt-water surface while it gradually melts completely, leaving no solid ice by the end of the shot. It does not sink as an intact solid block. The meltwater mixes naturally with the surrounding salt water.

No liquid is added, removed, spilled, or visibly evaporated. The liquid surface remains clearly visible throughout the shot.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```
