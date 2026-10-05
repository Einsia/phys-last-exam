# P41：水与沙的漏斗排放

难度：困难；本轮任务排名：35/40；平均最终综合分：0.156825。

## 场景与目标

固定镜头观察两个相同的透明圆锥漏斗，出口尺寸相同、初始填充高度相同。左侧为浅蓝色水，右侧为干沙；两出口同时打开并保持打开，材料排至接近空。观察两侧料位及流束，比较水和沙的流量随剩余高度变化的关系。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical transparent funnels sit side by side with identical outlet sizes and the same initial fill height; the left contains pale blue water and the right contains dry sand. Both outlets open simultaneously and remain fully open while each funnel discharges its own material until nearly empty. Both fill levels and both outlet streams remain clearly visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 物理指标

M1 分别拟合水和沙的 `Q ∝ h^β`，再比较水的指数与 0.5、沙的指数与 0 的偏差。

`E = |β_water - 0.5| + |β_sand|`

`M1 = 1 / (1 + E / a)`，默认 `a=1.0`。

指数和误差均无量纲。高度以像素测量；由已审查的圆锥内腔及自由表面恢复相对体积，通过积分体积/水头关系自由拟合指数。相对体积单位为 px³，流量为 px³/s，不是经过物理尺寸标定的 SI 体积或流量。有限指数误差平滑扣分，拟合不确定度单独报告。

实现依据：[measured_discharge.py](evaluator/measured_discharge.py) 的 `p41` 与 [discharge_integral.py](evaluator/discharge_integral.py)，版本 `p41_fixed_cone_integrated_exponent_v5`。

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
