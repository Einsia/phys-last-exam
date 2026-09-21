<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# P14 — 点光源照四杆 影子共点

Group 4。本题包含 gpt 首帧与仿真首帧两路（若保留），每路 seed 42/43/44/45
各一个续写视频，共 8 个。

## 指标

指标定义严格取自 `题包Benchmark.md`，不增不减。题包为某题列出多个辅助量时，
每个量占一个槽位（M2、M3、…），不合并成一个数。

| 槽位 | 测量对象 | 提取成功率 |
|---|---|---|
| M1 | common-intersection residual of the back-extended object-shadow lines | 8/8 |
| M2 | variance of the independent light-source position estimates | 8/8 |

## 目录

- `prompt.txt` 生成这批视频用的续写 prompt
- `first_frames/gpt/gpt_01.png` + `prompt.txt` gpt 首帧及其生成 prompt
- `first_frames/simulation/sim_01.png` 仿真首帧：`first_frames/simulation/sim_01.png`
- `evaluator/evaluate.py` 单视频评测入口，`evaluator/utils/physeval/` 是随包
  自带的测量库，不依赖包外代码
- `scripts/launch.sh` 批量生成，`scripts/run_eval.sh` 批量评测
- `output_videos/minimax_h3/{gpt,sim}_seed<NN>.mp4`
- `eval_results/minimax_h3/result_<sample>.json` 逐样本结果
- `eval_results/minimax_h3/debug/<sample>.png` 测量过程图：每张图上都能直接
  读出该视频被测的物理量

## 复现

```bash
pip install -r evaluator/requirements.txt
bash scripts/run_eval.sh minimax_h3
```

单个视频：

```bash
python evaluator/evaluate.py \
    --video output_videos/minimax_h3/gpt_seed42.mp4 \
    --image first_frames/gpt/gpt_01.png \
    --out /tmp/one.json --debug /tmp/one.png
```

## 输出结构

`metrics.<Mk>.extract_success` 为 false 时 `metric` 一律为 null；题包未给该题定义
某个槽位时 `extract_success` 为 null（与「有定义但没提取出来」区分）。
`verbose.<Mk>` 里写明该指标的物理原理、逐步测量过程和用到的中间量，
`verbose.debug_image` 指向对应的过程图。
