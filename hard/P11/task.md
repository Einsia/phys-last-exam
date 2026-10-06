# P11：粗糙斜面往返滑动

分类：2. 滚动、摩擦与刚体静力学（Rolling, Friction, and Rigid-Body Statics）；原编号：`P10`。

难度：困难；本轮任务排名：37/40；平均最终综合分：0.146558。

## 场景与目标

滑块沿粗糙直斜面向上运动，减速停止后自行沿原轨道下滑。动摩擦系数设为0.20，斜面目标角度约30°；根据视频中可读斜角与两段轨迹，检验上下行加速度大小之比。

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
| M1：上下行加速度比误差 | R理=(sinθ+0.20cosθ)/(sinθ−0.20cosθ)；e=abs((a上/a下)/R理−1)；P=1/(1+e/0.10)。当前另有已确认仅上行的部分事件规则：P=0.1。 | 比值与 e 无量纲；加速度为像素/秒²；θ 为斜角。 |

当前公式依据：`v4/g7/evaluator/tasks/p10_mechanics.py:913`；`v4/unified_evaluators/physics.py:151`。

## 生成提示词原文

```text
Locked-off static camera, side view. A rigid block is launched upward along a straight rough ramp whose rendered inclination is close to 30 degrees above horizontal; the kinetic friction coefficient is 0.20. The block slides upward, slows continuously, comes to a complete stop, and then slides back down along exactly the same path on its own. Keep the ramp edge and the full upward/downward motion visible so the rendered incline angle can be measured from pixels. The camera does not move, pan, or zoom. Plain background, only the ramp and block.
```
