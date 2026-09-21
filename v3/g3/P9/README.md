<!-- unified-v2-20260911 -->
> **当前入口与输出（2026-09-11）**：`evaluator/evaluate.py` 和 `scripts/run_eval.sh` 已接入 [G1–G9 统一规范](../../OUTPUT_FORMAT_V2.md)。`metric` 保存 0–1 的最终指标分（含识别奖励），原始量保存在 `verbose.M*.raw_metric`；总分仅对已定义指标等权计算，提取失败的已定义指标贡献为零。视频、首帧和提示词对应关系见 `metadata_v2.json`。结果位于 `eval_results/<model>/result_sample_xx.json`，`verbose` 中列出测量原理与证据。下方原有方法说明和参数供参考，以统一规范为当前输出合同。

# G3 / P9 — 重新评分 V2

当前正式分数在 `eval_results/minimax_h3/`，均使用 V2。原始视频、首帧、prompt 和测量证据已保留。
V2：残差 q=1/(1+abs(e)/a)，每项识别成功得 0.15+0.85q，不可测项记 0；综合分为 0.5M1+0.5M2。

读取 JSON 的 video_path/image_path 以 `G3/` 为基准，完整配对和文件路径见根目录 PACKAGE_MANIFEST.json。
单视频对应的精确 video_prompt 位于结果 JSON；P11 不同批次不可混用 prompt。

```bash
python evaluator/evaluate.py --verify
bash scripts/run_eval.sh minimax_h3 --output-root /new/empty/output
```

默认入口是**已有测量的评分复算**，不重新分割或跟踪，不支持把未知视频当成已有测量。
`evaluate_raw_legacy.py` 及同目录工具是保留的历史提取实现；它们的旧式原始输出不能当成 V2 最终分。
历史说明和分数在包根目录 `legacy_v1/G3/P9/`；新版以根目录评分规则和汇总为准。
