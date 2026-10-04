# P21 · 浮冰融化液面变化

难度：困难；本轮任务排名：38/40；平均最终综合分：0.142956。

任务 ID：`P21`。

透明直壁杯内的一块淡水浮冰逐渐完全融化，始终浮在水面，末尾不再有固体冰。没有溢流、蒸发或液体增减，初末水位清楚可见，淡青色水和不透明白冰可区分。理想情况下浮冰融化前后水位不变。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：(融化后水位高度−融化前水位高度)/容器高度；无量纲。
- M2：|末段容器宽度−初段容器宽度|/初段容器宽度，检查容器边界一致性；无量纲。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view, matching the input first frame.

A piece of pure freshwater ice initially floats freely in fresh water inside a transparent straight-walled glass. The ice remains floating at the water surface while it gradually melts completely, leaving no solid ice by the end of the shot. It does not sink as an intact solid block.

There is no overflow, visible evaporation, or addition or removal of liquid or material. The initial and final water levels remain clearly visible.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.

Throughout the clip the water keeps the same distinctly light cyan-blue tint it has in the first frame and the ice stays opaque white, so the waterline and the ice are never confusable. The wall behind the glass stays one flat tone, with no dark horizontal band appearing behind or across the beaker at any time. Nothing else in the scene takes on that blue.
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

指标定义依据：`v4/g4/P21/evaluator/utils/physeval/tasks/p21.py:33`；`v4/unified_evaluators/contract.py:100`；`v4/g4/P21/evaluator/utils/physeval/tasks/p21.py:40`。
