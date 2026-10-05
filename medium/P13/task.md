# P13 · 光的反射

难度：中等；本轮任务排名：22/40；平均最终综合分：0.337269。

任务 ID：`P13`。

固定镜头下，一束细激光斜射平面镜，入射光、反射光、镜面、入射点及法线清楚可见。应呈现同一入射点处的镜面反射，入射角与反射角相等。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：入射角与反射角的绝对差，单位为度。
- M2：反射光在镜面处的接触定位误差中位数，除以画面对角线长度；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera. A thin clearly visible laser beam strikes a flat plane mirror at an oblique angle. The incident ray, reflected ray, mirror surface, point of incidence, and mirror normal are all clearly visible. The camera does not move, pan, or zoom. Plain dark background, no other objects.
```

## 评测入口

在本题目录中执行，输入和输出视频路径均可放在题包外：

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```

指标定义依据：`v4/g2/P13/evaluator/measure_backend.py:652`；`v4/unified_evaluators/contract.py:155`；`v4/g2/P13/evaluator/measure_backend.py:551`。
