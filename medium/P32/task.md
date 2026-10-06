# P32：两根指南针的最终方向

分类：7. 静电、磁学与电磁感应（Electrostatics, Magnetism, and Electromagnetic Induction）；原编号：`P34`。

难度：中等；本轮任务排名：30/40；平均最终综合分：0.210206。

## 场景与目标

俯视固定导线与两只指南针。导线通电后，两根针各自绕中心转动并停稳；比较导线两侧针的最终磁极朝向。导线、底板和指南针外壳保持固定。

## 输入

- 首帧：[first_frame.png](first_frame.png)。
- 视频生成提示词：[prompt.txt](prompt.txt)。
- 待评视频：外部输入，通过 `--video` 指定。

## 提示词原文

```text
Locked-off top-down camera. Continue from the supplied first frame. A current begins to flow in the vertical wire. The only motion is the two compass needles turning on their own pivots on the compass faces; they then settle into stable orientations. The wire, the board, the compass housings and the rest of the apparatus stay exactly as in the first frame. The camera does not move, pan, or zoom. Plain background.
```

## M1 物理指标

M1 衡量两针最终朝向的相反程度。取视频末尾 5 帧中两针均可读的帧，至少需要 2 帧；计算每帧两针方向的最小夹角，再取中位数 `d`（度，范围 0–180）。

`M1 = (1 - cos(d × π / 180)) / 2`

反向 180° 得 1，同向 0° 得 0。当前 M1 只使用最终方向，不以相对首帧的偏转量、转动轨迹或稳定时长计分。

实现依据：[measurement.py](evaluator/utils/measurement.py) 的 `summarize`。

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
