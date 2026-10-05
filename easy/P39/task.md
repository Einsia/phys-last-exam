# P39：磁铁进入线圈与指示灯发光

难度：简单；本轮任务排名：13/40；平均最终综合分：0.503125。

## 场景与目标

固定镜头观察条形磁铁、固定线圈及连接的双向指示灯。磁铁由静止开始运动，进入并穿过线圈，从另一端离开后再静止；观察磁体运动与灯光响应，保持磁铁、线圈和灯清晰可见。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. A bar magnet first remains completely stationary near a fixed coil connected to a small bidirectional indicator lamp. The magnet then moves into the coil, passes completely through it, moves out the other side, and finally becomes stationary again. The magnet, coil and lamp remain clearly visible throughout the entire sequence. The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```

## M1 物理指标

当前 M1 由两个直接可观测事件组成：磁铁进入线圈，以及进入期间灯可确认发光。

`M1 = 0.5 × I(已观测到进入) + 0.5 × I(进入期间已确认灯亮)`

`I` 为事件成立时取 1、否则取 0 的指示量，M1 无量纲。灯在磁铁进入前已亮也可计入；默认事件匹配容差为 0.5 秒。已确认进入但灯状态无法判断时，M1=0.5 是证据下界，另报告 `[0.5, 1]` 的不确定范围，不将未知灯态认作熄灭。

完整穿过和再次静止属于视频任务描述；当前 M1 不测磁通、电流、电压或灯光时间重心。

实现依据：[entry_light.py](evaluator/entry_light.py) 的 `evaluate`，版本 `p39_entry_light_v5`。

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
