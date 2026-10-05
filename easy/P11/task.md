# P11 · 光的折射

难度：简单；本轮任务排名：15/40；平均最终综合分：0.445497。

任务 ID：`P11`。

固定激光源照向透明水槽中静止的空气—水界面。激光由关闭转为稳定照明，入射光、界面及水中折射光的方向和交点清楚可见。应呈现符合 Snell 定律的折射；各段光线在各自均匀介质内保持直线。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：sin(入射角)/sin(折射角)−n水；角度相对界面法线测量，结果无量纲。
- M2：入射光和折射光各自与液面相交位置之间的距离，除以首帧标定容器宽度；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

### 当前生成提示词

采用与 30° 首帧对应的 v4 提示词；以下是本题唯一活动生成提示词。

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing transparent tank, clear still fresh water, laser source, support apparatus, framing, and background. The laser source and its mount remain completely stationary.

The laser is initially switched off. After a brief still moment, it switches on and emits a thin, clearly visible red beam along its existing optical axis. The beam travels through the air and enters the water obliquely at one fixed point on the flat air-water interface.

Exactly one incident ray in the air and one refracted ray in the water are clearly visible. They meet cleanly and continuously at the same point of incidence and remain straight within their respective media. The flat air-water interface, point of incidence, and a thin stationary interface normal perpendicular to the water surface at that point are all clearly visible.

The light remains thin and sharp without excessive bloom or overexposure. The water surface remains still and horizontal. Preserve the clean photographic appearance and existing uncluttered background of the input frame.

No camera movement, panning, zooming, cuts, slow motion, or time jumps. No people, hands, angle arcs, arrows, measurements, equations, numbers, labels, or annotations other than the interface normal.

End after the visible optical behavior has remained stable for a short moment.
```

## 评测入口

在本题目录中显式传入外部视频、根目录首帧和保留的完整提示词文件：

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```

现行测量后端使用题内场景标定；输入视频须与所选 30° 首帧标定对应。

指标定义依据：`v4/unified_evaluators/physics.py:22`；`v4/g3/P11/evaluator/evaluate_raw_legacy.py:900`。

## 当前场景标定

为使用与本题唯一 30° 首帧对应的已有标定，外部视频应命名为 `P11_gpt_01_30deg_seedN.mp4`（N 为任意整数）。其他文件名会触发后端原有的通用标定，不能视为同一场景标定。
