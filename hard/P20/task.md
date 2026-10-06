# P20：折射与全反射临界角

分类：4. 光学与投影几何（Optics and Projective Geometry）；原编号：`P12`。

难度：困难；本轮任务排名：31/40；平均最终综合分：0.192774。

## 场景与目标

固定水面上有四束可区分的有色光束及各自的界面交点。先补全穿过界面的光路，再保持交点固定地改变入射方向；目标行为是遵循折射定律及适用条件下的全反射。当前评分比较独立光束推得的折射率是否一致。

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
| M1：折射率估计一致性 | 各可测光束按 Snell 几何估计折射率 n；e=std(n)/mean(n)；P=1/(1+e/0.05)。普通全反射只提供下界，不冒充临界角等式。 | n、e 与 P 无量纲。 |

当前公式依据：`v4/g7/evaluator/tasks/p12_optics.py:1193`；`v4/unified_evaluators/physics.py:150`。

## 生成提示词原文

```text
Locked-off exact front orthographic view. Continue the incident-only frame in two stages while keeping the horizontal water surface and all four surface hit points fixed. In stage one, complete each coloured beam across the interface with its physically correct Snell-law outcome. In stage two, smoothly swing each beam pair about its own fixed hit point, continuously preserving the refractive relationship between the air and water arms. Keep the tank, water level, dashed normals, colours, camera, scale and background fixed; do not slide a corner along the surface or add text or extra apparatus.
```
