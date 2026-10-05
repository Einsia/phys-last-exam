# P43：持续推动下块体绕支撑边转动

难度：中等；本轮任务排名：26/40；平均最终综合分：0.270230。

## 场景与目标

侧视固定镜头观察执行器推动块体。左侧推杆持续向右伸出，推垫始终接触块体左面的上部，使块体绕右下支撑边转动直至躺下；块体不沿地面滑动，不离开推垫自行倾倒。块体、接触区域和支撑面保持可见。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, side view. Continue from the supplied first frame. The linear actuator on the left slowly extends its ram to the right. The pad stays pressed against the upper part of the block's left face and keeps advancing with the ram, pushing the block. The block does not slide along the surface, does not leave the pad, and does not fall over by itself. Keep the ram extending until the block has rotated about its right-hand bottom edge and is lying on the surface. The actuator body stays fixed; only the ram lengthens. The complete block, contact region and supporting surface remain visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 物理指标

当前 M1 检查可见转动期间的支撑点漂移、离地间隙和刚体形状变化。

`E = max(e_pivot, e_contact, e_shape)`

`M1 = I(可见转角 ≥ 5°) / (1 + E / a)`，默认 `a=0.02`。

`e_pivot`、`e_contact` 分别将支撑点漂移和接触间隙的第 95 百分位减去两倍像素噪声下限、截为非负值，再除以初始块体对角线；`e_shape` 为扣除像素噪声后的相对边长变化。因此 `E` 和 M1 均无量纲，原始距离单位为 px，转角为度。

使用所有至少 4 帧且持续 0.3 秒的连续可见区间，取这些区间中最大的可见转角。没有达到 5° 的可见转动时记 0；隐藏姿态不插值。当前指标不使用“质心越界与自由倾倒起点的帧差”。

实现依据：[observed_support.py](evaluator/observed_support.py) 的 `measure`、`normalized_measurement`，版本 `p43_observed_support_v4`。

## 评测入口

从本题目录运行：

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --annotation /absolute/path/to/video_annotations.json \
  --output /absolute/path/to/result.json
```

需使用已配置评测依赖及一致性模型的 Python 环境。 `video_annotations.json` 必须针对本次外部视频审查并绑定其视频及首帧哈希；不得直接复用其他视频的标注。
