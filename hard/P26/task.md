# P26：含石浮冰融化

分类：6. 相变与融化（Phase Transitions and Melting）；原编号：`P21b`。

难度：困难；本轮任务排名：40/40；平均最终综合分：0.034453。

## 场景与目标

透明直壁容器内，包裹高密度石块的淡水冰漂浮在淡水中。冰逐渐融化，释放石块并使其沉下；在不增减水或其他物质的条件下，比较融化前后的水位，目标行为是水位降低。

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
| M1：融化后水位变化 | e=(y初−y末)/H初；P=1 当 e<0，否则 P=0。须先观察到融化，且水位变化大于测量分辨率。 | e 无量纲；y、H 为像素。 |

当前公式依据：`v4/g5/P21b/evaluator/measure_backend.py:460`；`v4/g5/P21b/evaluator/measure_backend.py:558`；`v4/g5/P21b/evaluator/measure_backend.py:568`。

## 生成提示词原文

```text
Locked-off static camera, front view, matching the input first frame.

A dense stone is completely frozen inside a piece of freshwater ice that initially floats freely in fresh water inside a transparent straight-walled container.

The ice remains at the water surface while it gradually melts completely and releases the stone. No solid ice remains by the end of the shot. The ice does not sink as an intact solid block; only the released stone sinks to the bottom of the container.

No water or other material is added or removed. The waterline and the stone remain clearly visible throughout the shot.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```
