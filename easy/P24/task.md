# P24：水冻结体积膨胀

分类：6. 相变与融化（Phase Transitions and Melting）；原编号：`P23`。

难度：简单（人工调整）；冻结评测排名：16/40；平均最终综合分：0.438451。

## 场景与目标

透明直壁容器中的液态水完全冻结成冰，不添加、移走或洒出物质。观察初始水位和最终冰面，在容器截面不变时检验水结冰后的体积膨胀。

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
| M1：冻结高度比误差 | e=H冰/H水−1000/917；P=1/(1+abs(e))。 | e 无量纲；H 为像素。 |
| M2：截面宽度一致性 | e=abs(W末−W初)/max(W初,1像素)；P=1/(1+abs(e))。 | e 无量纲；W 为像素。 |

当前公式依据：`v4/g5/P23/evaluator/measure_backend.py:168`；`v4/g5/P23/evaluator/measure_backend.py:199`。

## 生成提示词原文

```text
Locked-off static camera, front view. Pure liquid water fills part of a transparent straight-walled container and then freezes completely into solid ice. No water is added, removed, spilled, or visibly evaporated during the process. The initial liquid-water level and final top surface of the ice are both clearly visible. The camera does not move, pan, or zoom. Plain background.
```
