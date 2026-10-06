# P35 · 沙堆休止角尺度一致性

分类：8. 颗粒介质与排出流动（Granular Media and Discharge Flow）；原编号：`P40`。

难度：简单；本轮任务排名：2/40；平均最终综合分：0.855037。

任务 ID：`P35`。

同一水平台面上放置两堆大小不同、材料相同的干燥无黏性沙，两股相同的细沙流分别落在堆顶。沙粒沿坡面滑落并稳定，两沙堆不合并且完整轮廓可见。大小沙堆的休止角应近似相同。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：|小沙堆休止角/大沙堆休止角−1|，无量纲。
- M2：分别计算每个沙堆左右坡面的角度差，并取两堆中的较大值；单位为度。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two piles of the same dry non-cohesive sand stand at rest on the same flat horizontal bench, the left pile clearly larger than the right one. A thin steady stream of the same dry sand begins to fall straight down onto the apex of each pile from above the top edge of the frame, at the same steady rate for both. Each pile grows as the sand lands, and the loose grains keep running down its slope faces and settling, so both piles stay conical and keep their slope faces clean and continuous down to the bench line. The two piles stay separate and never merge. The sand source stays outside the frame; no hopper, funnel, tube, container or hand ever enters the frame. Both complete pile profiles and the straight horizontal bench line remain clearly visible for the whole clip. The bench, the background and the camera stay exactly as in the first frame. The camera does not move, pan, or zoom. Plain background, no other objects.
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

指标定义依据：`v4/g2/P40/evaluator/measure_backend.py:315`；`v4/unified_evaluators/contract.py:155`。
