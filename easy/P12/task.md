# P12：单摆周期与质量无关

分类：3. 单摆运动与振荡（Pendulum Motion and Oscillations）；原编号：`P8b`。

难度：简单；本轮任务排名：11/40；平均最终综合分：0.530868。

## 场景与目标

同一支架悬挂两个等摆长单摆，左侧摆锤小而轻，右侧摆锤大而重。两摆从相同角度同时释放并振荡若干周期；检验等摆长下周期不随摆锤质量改变。

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
| M1：质量不同的两摆周期比 | e=T重/T轻−1；P=1/(1+abs(e)/0.10)。 | e 无量纲；T 使用一致时间单位。 |
| M2：首帧摆长相对差 | e=abs(L左−L右)/mean(L左,L右)；P=1/(1+e/0.10)。悬点由视频摆动轨迹拟合。 | e 无量纲；L 为像素。 |

当前公式依据：`v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:51`；`v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:148`；`v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:56`；`v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:118`。

## 生成提示词原文

```text
Locked-off static camera, side view. Two pendulums of equal rod length hang from the same horizontal support; the left bob is light and small, the right bob is heavy and large. They are released together from the same angle and swing back and forth for several cycles. The camera does not move, pan, or zoom. Plain flat background, only the two pendulums, nothing else enters the frame.
```
