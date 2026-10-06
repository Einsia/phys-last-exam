# P39：不同大小肥皂泡的共同隔膜曲率

分类：9. 表面张力与黏性流（Surface Tension and Viscous Flow）；原编号：`P47`。

难度：中等；本轮任务排名：24/40；平均最终综合分：0.320569。

## 场景与目标

固定侧视近景中，两个大小明显不同的近球形肥皂泡相向运动，接触后通过一片可见的内部隔膜相连，保持连接且不破裂。两泡外轮廓及完整隔膜始终清晰可见，用于比较曲率与压差关系。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off static camera, close-up side view. Continue from the supplied first frame. Two roughly spherical soap bubbles of clearly different sizes move toward each other and stay connected by one clearly visible internal partition. The outer boundaries of both bubbles and the complete partition remain sharply visible and inside the frame. The bubbles do not pop or detach. The camera does not move, pan, or zoom. Plain dark background.
```

## M1 物理指标

M1 独立拟合小泡外弧半径 `r_small`、大泡外弧半径 `r_large` 和隔膜有符号半径 `r_partition`，三者单位均为 px。

`E_frame = |r_partition × (1/r_small - 1/r_large) - 1|`

`E = 按真实帧时间间隔加权的 E_frame 中位数`

`M1 = 1 / (1 + E / a)`，默认 `a=1.0`。

误差及分数均无量纲。使用全部至少 3 帧且持续 0.08 秒、隔膜可唯一辨识的连续区间；隔膜方向错误或非理想半径通过误差扣分。若已证明不等泡间隔膜近直线，且曲率不确定度排除了理论预测，则单独记录无界误差极限并记 0；普通曲率不可辨不等同该极限。

实现依据：[measured_bubbles.py](evaluator/measured_bubbles.py) 的 `inspect_frame`、`summarize`。

## 评测入口

从本题目录运行：

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

需使用已配置评测依赖及一致性模型的 Python 环境。
