# P38：实心板与开槽板的涡流阻尼

难度：困难；本轮任务排名：39/40；平均最终综合分：0.142813。

## 场景与目标

固定镜头观察质量及转动惯量匹配的实心导体板和开槽导体板。两板从相同角度、零初速同时释放，经过等效磁场区域并持续摆动；比较同一观察窗口中的完整摆动次数，观察开槽对涡流阻尼的影响。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. A solid conducting plate and a slotted conducting plate, matched in total mass and rotational inertia, are released simultaneously from the same initial angle with zero initial speed and swing through equivalent magnetic-field regions. Both oscillations remain fully visible for multiple cycles. The camera does not move, pan, or zoom. Plain background.
```

## M1 物理指标

M1 在共同观察窗口、共同角振幅门槛下，分别统计实心板和开槽板的有效完整周期数 `N_solid`、`N_slotted`（整数）。

`M1 = 1，若 N_solid < N_slotted；否则为 0`

当 `N_slotted=0` 时记 0，次数比保持空值。默认观察窗口至少 2 秒，振幅门槛为 `max(1°, 0.1 × 两板初始绝对角度的平均值)`。完整周期由同一极性的边界配对；截断的部分周期不计数。当前指标不估计衰减率。

实现依据：[measured_center.py](evaluator/measured_center.py) 的 `p38` 与 [observed_zero.py](evaluator/observed_zero.py)。

## 评测入口

从本题目录运行：

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use the configured evaluation environment. The evaluator automatically loads `first_frame_annotations.json` bundled with this task. It verifies the input-image hash, scales the coordinates to the video resolution, and checks correspondence with the decoded first frame before tracking. No per-video annotation is needed for the fixed task image when this check passes. Use `--annotation /absolute/path/to/video_annotations.json` only to override initialization for a different image or layout; custom video annotations retain their video/image hash checks.
