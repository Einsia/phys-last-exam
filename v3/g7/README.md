# G7 完整重评分交付

共 120 条视频，题目：P7, P8c, P10, P12, P27。
每题 `eval_results/minimax_h3/` 是新版结果。历史分数已移到包根目录 `legacy_v1/`，不作为当前分数。
包根目录包含新版规则、总表、可复算代码与全文件校验清单。

```bash
python evaluate_v2.py --verify
python evaluate_v2.py --output-root /new/empty/output
```

这是基于已保存测量的重新评分，不代表缺失测量已修复。重跑原始提取器可能需要它原有依赖、外部权重和环境路径配置。
