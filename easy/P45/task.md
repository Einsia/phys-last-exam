# P45：毛细上升与管径

难度：简单；本轮任务排名：12/40；平均最终综合分：0.511797。

## 场景与目标

两根不同内径的干燥玻璃毛细管最初悬在同一储液体上方，随后同时浅浸入液体。观察公共外液面与两个管内弯月面；目标行为是细管毛细上升高度大于粗管。当前评分检验高度方向，不检验严格反比幅值。

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
| M1：毛细上升高度方向 | h=y公共液面−y管内液面；d=末段成对高度差 h细−h粗 的中位数。d 大于测量分辨率时 P=1，否则 P=0；两管及公共液面须可成对读取。 | h、d、分辨率均为像素；P 无量纲。 |

当前公式依据：`v4/g6/P45/evaluator/utils/physeval/tasks/p45.py:141`；`v4/g6/P45/evaluator/utils/physeval/tasks/p45.py:230`。

## 生成提示词原文

```text
Task: P45_capillary_rise_two_radii.
Start exactly from the supplied dry first frame: two clean, vertical, parallel, open-ended glass capillary tubes are held side by side above one continuous transparent water reservoir. Their lower rims are completely above the sharp horizontal water surface, with a clear air gap below both rims. Both tube interiors are completely dry and empty at frame 0. One tube has a narrow inner radius and the other has an inner radius about twice as large.

Video action: use a locked-off, fixed, straight-on laboratory teaching-video camera with no pan, zoom, shake, or reframing. Hold the dry suspended state briefly, then lower both tubes together at the same speed into the same reservoir until their lower ends are immersed to the same shallow depth. After insertion, show the capillary phenomenon: water wets the identical glass walls and rises inside both open tubes from the common reservoir level, forming two attached concave menisci. The rise is modest and physically plausible, not a tall liquid column. The narrower tube rises higher than the wider tube because capillary rise height is inversely proportional to inner radius (approximately h_narrow*r_narrow = h_wide*r_wide). Keep both liquid columns clearly below the available tube length, with the narrow-tube column visibly higher but only moderately so. Keep the shared external water surface, tube bores, lower openings, menisci, clamp, and tank visible throughout, with realistic transparent-glass refraction and liquid reflections.

Hard negative constraints: no liquid or meniscus inside either tube at frame 0; no tube touching or crossing the water surface at frame 0; no exaggerated or near-top liquid columns; no equal final rise heights; no wider tube rising higher than the narrow tube; no closed or rounded tube ends; no separate reservoirs; no different immersion depths; no person or hand; no labels, numbers, ruler, equations, arrows, trajectories, watermark, logo, cartoon styling, camera motion, or cropped key objects.

Throughout the clip the water keeps the same distinctly light cyan-blue tint it has in the first frame, in the tank and in the risen columns alike, so each liquid column reads as an obviously coloured body against the dry glass above it. Nothing else in the scene takes on that blue.
```
