# P27：碎冰与整冰融化对比

难度：困难；本轮任务排名：32/40；平均最终综合分：0.192187。

## 场景与目标

相同透明直壁烧杯左右并排，左侧为整块冰，右侧为同质量碎冰。二者同时融化并形成可见液层；当前指标检验确认两侧可见融化后，碎冰杯按杯高归一的液位是否出现持续领先，不把二维冰面积当作质量。

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
| M1：碎冰杯液位持续领先 | 先确认两侧持续融化；同帧以各自杯底为基准、按杯高归一液位。碎冰杯领先超过两侧定位误差之和且持续≥0.25秒：P=1；存在≥0.50秒共同可读连续片段而未观察到领先：P=0。 | 归一液位差无量纲；持续时间为秒；定位误差由像素归一。 |

当前公式依据：`v4/unified_evaluators/physics.py:132`；`v4/g7/P27/README.md:30`；`v4/g7/P27/README.md:36`。

## 生成提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical transparent straight-walled beakers sit side by side; the left holds one compact ice block and the right holds crushed ice of the same total mass. Melting begins at the same time and continues until all visible ice has turned into water, and a liquid surface appears and rises in each beaker. Both beakers remain fully visible. The camera does not move, pan, or zoom. Plain dark background.
```
