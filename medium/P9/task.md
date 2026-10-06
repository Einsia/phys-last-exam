# P9：实心球与圆环滚动对比

分类：2. 滚动、摩擦与刚体静力学（Rolling, Friction, and Rigid-Body Statics）；原编号：`P7`。

难度：中等；本轮任务排名：25/40；平均最终综合分：0.284114。

## 场景与目标

同外半径的均匀实心球和薄圆环在同一斜面、同一高度同时从静止释放并无滑动滚下。检验两物体通过相同路程所需时间之比，理论圆环/实心球耗时比为sqrt(10/7)。

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
| M1：等路程耗时比误差 | 在多个虚拟等路程位置取圆环/实心球耗时比 r 的中位数；e=abs(r/sqrt(10/7)−1)；P=1/(1+e/0.10)。 | r、e 与 P 无量纲；时间为秒。 |

当前公式依据：`v4/g7/evaluator/tasks/p7_rotational.py:691`；`v4/unified_evaluators/physics.py:150`。

## 生成提示词原文

```text
Locked-off static camera, side view. A uniform solid sphere and a thin circular ring have exactly the same outer radius. They are placed side by side at the same height on the same straight incline and released simultaneously from rest. Both objects roll down the incline without slipping. Their complete motions and the common finish position remain visible. The camera does not move, pan, or zoom. Plain background, no other objects.
```
