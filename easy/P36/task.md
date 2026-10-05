# P36：磁性物块涡流制动

难度：简单；本轮任务排名：10/40；平均最终综合分：0.568750。

## 场景与目标

铜斜板上并排放置外形、质量与接触面相同的两个物体，其中一个含强磁体，另一个含非磁性配重。二者同时从静止释放；目标行为是涡流制动使磁性物体相对对照运动更迟、更慢。

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
| M1：轨迹检测事件时间比 | r_t=(f磁−f起+1)/(f对照−f起+1)；P=1 当 r_t>1，否则 P=0。f 为后端检测的事件帧。 | r_t 无量纲；f 为帧号。 |
| M2：运动速度比 | r_v=v磁/v对照；P=1 当 r_v<1，否则 P=0。 | r_v 无量纲；v 为像素/秒。 |

当前公式依据：`v4/g5/P36/evaluator/measure_backend.py:271`。

## 生成提示词原文

```text
Locked-off static camera, side view. Two externally identical blocks with the same total mass and identical flat contact surfaces are placed side by side at the same height on the same inclined copper plate. One block contains a strong neodymium magnet and the other contains an equal-mass non-magnetic insert. They are released simultaneously from rest and slide freely down the copper plate. Both complete motions remain visible. The camera does not move, pan, or zoom. Plain background.
```
