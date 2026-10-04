# P48：液滴合并体积守恒

难度：中等；本轮任务排名：21/40；平均最终综合分：0.342852。

## 场景与目标

两个大小不同、轮廓完整的自由水滴在空中接近，接触并合为一个液滴。保持液体体积及完整边界可见；检验合并前后的体积守恒与圆度变化。

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
| M1：液滴体积残差 | e=abs(r末³/(r1³+r2³)−1)；P=1/(1+e/0.10)。 | e 无量纲；半径 r 为像素。 |
| M2：圆度变化 | 圆度=1−轮廓径向RMS误差/拟合半径；e=abs(末圆度−初两滴平均圆度)；P=1/(1+e/0.10)。 | 圆度、e 与 P 无量纲。 |

当前公式依据：`v4/g6/P48/evaluator/utils/physeval/tasks/p48.py:32`；`v4/g6/P48/evaluator/utils/physeval/tasks/p48.py:112`；`v4/g6/P48/evaluator/utils/physeval/tasks/p48.py:37`；`v4/g6/P48/evaluator/utils/physeval/tasks/p48.py:121`。

## 生成提示词原文

```text
Task: P48_free_droplet_coalescence.
A high-speed macro laboratory view freezes two clean spherical water droplets fully detached in air, separated by a very small gap and approaching one another along the same horizontal axis. Their radii are visibly different but both outlines are complete, sharply focused, and undeformed before contact. The release nozzles are outside the central measurement region, the background is dark and plain, and ample empty space surrounds the expected merged droplet.

Video action: use a locked-off short-exposure teaching-laboratory camera. Begin with the two complete, unequal, same-liquid droplets exactly as shown, with a small visible air gap. Then let them translate gently toward one another, touch, form one continuous liquid volume, and settle into one approximately spherical merged droplet while conserving volume. Keep every droplet fully inside frame before, during, and after coalescence. Preserve soft liquid refraction and natural diffuse highlights.

Hard negative constraints: no contact or liquid bridge in the first frame, no nozzle, support, surface, pool impact, splash crown, satellite droplets, extra droplets, material loss, hard glass-ball shell, hollow bubble, plastic bead, marble, frosted or milky material, motion blur, camera movement, labels, radius guides, arrows, trajectories, watermark, logo, or toy/illustration styling.
```
