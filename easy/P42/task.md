# P42 · 摩擦起滑与质量无关

难度：简单；本轮任务排名：7/40；平均最终综合分：0.608458。

任务 ID：`P42`。

两个不同质量物块并排静置在近水平板上，板的铰接端固定，另一端被螺旋升降装置连续抬高。两块在倾角足够大时自行起滑，持续拍摄到两者明显滑动。相同接触条件下，临界起滑倾角应与质量无关。

## 输入

唯一参考首帧为题包根目录的 [first_frame.png](first_frame.png)。待评测视频由外部提供，不属于题包内容。

## 当前物理指标

- M1：两块起滑帧之差的绝对值/视频总帧数，无量纲。
- M2：两块各自起滑时板面倾角的绝对差，单位为度。
- 当前实现对“两块主体可测、均未开始滑动”另记录静止结果，两项物理分数均为 0.7。

以上为原始测量量；归一化后的物理分数取值为 0–1。

## 生成提示词原文

```text
Locked-off static camera, side view. Continue from the supplied first frame. The board starts lying almost flat with the two blocks resting side by side on it. The screw jack under the free end then extends steadily and the board is tilted up: its free end rises continuously and the tilt angle grows smoothly from nearly horizontal at the start of the clip to steeply inclined by the end. This lifting never stops, never pauses, never reverses and never jumps — the board is visibly at a larger angle in every later frame than in every earlier one. The hinged end stays fixed on the bench the whole time. While the board is still shallow both blocks stay exactly where they are on it, and each block starts to slide down the board on its own once the board has become steep enough; keep filming until both blocks have clearly broken away and are sliding. Nothing is added to or removed from either block and no hand ever enters the frame. Both blocks and the full board stay inside the frame. The camera does not move, pan, or zoom. Plain background, no other objects.
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

指标定义依据：`v4/g2/P42/evaluator/measure_backend.py:503`；`v4/unified_evaluators/contract.py:155`。
