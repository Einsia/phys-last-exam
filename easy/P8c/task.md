# P8c：大角度单摆周期

难度：简单；本轮任务排名：8/40；平均最终综合分：0.574597。

## 场景与目标

两个等摆长单摆从首帧所示的不同振幅同时静止释放，持续完成多个可读周期，摆锤与悬点保持可辨。检验有限振幅下大振幅摆周期较长，并与当前代码的有限振幅理论周期比比较。

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
| M1：有限振幅周期比误差 | r=T大振幅/T小振幅；e=abs(r−r理)，r理由当前有限振幅理论计算；P=1/(1+e/0.05)。 | r、e 与 P 无量纲；周期为秒。 |
| M2：周期方向程度 | P=clip((r−1)/(r理−1),0,1)。 | 比值与 P 无量纲。 |

当前公式依据：`v4/g7/evaluator/tasks/p8c_pendulum.py:755`；`v4/unified_evaluators/physics.py:150`；`v4/unified_evaluators/physics.py:111`；`v4/unified_evaluators/physics.py:116`。

## 生成提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. The red pendulum on the left and the blue pendulum on the right are both released from rest at the same instant. After release, both bobs immediately leave their first-frame poses and each swings back and forth on its own string for several complete periods while remaining fully visible. The red bob is not frozen and does not hang still. String lengths stay the same, colours stay the same, and the two pendulums do not collide. The support frame stays fixed. The camera does not move, pan, or zoom. Plain background, only the two pendulums.
```
