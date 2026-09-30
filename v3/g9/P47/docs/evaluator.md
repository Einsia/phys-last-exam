请为物理视频任务 P47 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。沿用 P34 的文档结构，以项目根目录 `bench.md` 的 M1 为准。

## 1. 任务定义

两个大小明显不同的近球形皂泡相连，形成清晰、完整的内部弯曲隔膜。评估隔膜曲率与两侧泡半径之间的压差关系。

只计算 M1：

```text
E_M1 = abs(r_s * (1/r_small - 1/r_large) - 1)
```

`r_small`、`r_large` 为两泡外边界所对应球面的半径，`r_s` 为共用隔膜的曲率半径，**不是小泡半径**。相同表面张力的近球形双泡，理想关系为 `1/r_s = 1/r_small - 1/r_large`，隔膜向大泡内部鼓起。

为避免反向鼓起也得满分，实现将隔膜半径扩展为有符号量：向大泡内部鼓起为正，反向为负。正常方向下与 bench 原公式完全相同；不得对负号取绝对值再代入。

外轮廓与隔膜的圆拟合残差是 bench 的 M2，不形成第二个分数；它们只能用于判断半径是否可可靠测量。物理依据见 [MIT 双泡与 Laplace 压力说明](https://web.mit.edu/nnf/education/wettability/bubbles.html)。

## 2. 文件结构与接口

以下路径相对于项目根目录：

* `g9/P47/output_videos/minimax_h3/sample_00.mp4`：视频。
* `g9/P47/first_frame.png`：对应首帧。
* `g9/P47/prompts/video.txt`：读取内容作为 `video_prompt`。
* `g9/P47/first_frames/provided/prompt.txt`：首帧提示词，不作为 `video_prompt`。

在 `g9/P47/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python g9/P47/evaluator/evaluate.py \
  --video_path g9/P47/output_videos/minimax_h3/sample_00.mp4 \
  --image_path g9/P47/first_frame.png \
  --video_prompt "$(cat g9/P47/prompts/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output g9/P47/eval_results/minimax_h3/result_sample_00.json

bash g9/P47/scripts/run_eval.sh minimax_h3
```

支持可选 `--model`、`--sample_id`、`--seed`、`--device`、本地权重和几何配置。示例显式指定当前输入为 `minimax_h3/sample_00`，不代表已有 metadata；未知种子和其他缺失元数据为 null。

批量脚本根据自身位置定位任务目录，只扫描直属输入 `.mp4`，不扫描 debug。按 `data/metadata.json` 显式配对模型、样本、视频、首帧和提示词；无 sample ID 回退视频 stem，未知模型目录为 `unknown_model`，JSON model 为 null。不能按遍历顺序编号或将本样本首帧配给所有视频。单样本失败不中断整批。

```text
g9/P47/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

实际名称统一为 `result_<sample_id>.json` 和 `debug_<sample_id>/`。

## 3. 工具与测量流程

* PyAV：解码并保存真实 PTS。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：定位两泡并提供外轮廓搜索区域。
* OpenCV：亚像素边缘、轮廓拓扑、隔膜端点与反光排除。
* SciPy、NumPy：独立圆弧/有符号曲率拟合、参数不确定度、逐帧 M1 聚合。
* Matplotlib、Python json：拟合图、曲线和结果。

接口参考：[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。泡体区域的 SAM 掩码不能直接给出透明隔膜，必须额外检测实际可见薄膜边缘。

测量流程：

1. 从初态确认两个泡的身份及大小关系，排除底部喷嘴和镜面高光。对应首帧只用于场景和身份核验；初始尚为很小的接触颈时不要求立即得到隔膜半径。
2. 跟踪外边界与连接处，找到隔膜与外表面交接的两个端点。分开标注小泡外弧、大泡外弧、内部隔膜弧，避免把镜面反射线或两个重叠外轮廓当隔膜。
3. 独立拟合两段外圆弧，避开连接形变区、喷嘴和遮挡。半径取圆弧拟合值，不取包围框宽度的一半。两泡身份不因单帧拟合噪声而交换。
4. 独立拟合隔膜的带符号曲率 `k_s`，`r_s=1/k_s`。符号由隔膜相对两端点连线向大泡一侧的鼓出方向确定，不能由理论公式推定。近直线优先拟合曲率及其置信区间，避免巨大圆半径的数值病态。
5. 同一平面的近正视成像下可直接使用一致像素尺度，因为该 M1 无量纲。必须说明侧视三圆弧可近似代表相应球面截面的条件；明显非轴对称、出平面偏转或未知投影不能靠单一椭圆拉伸伪造球面半径。
6. 在隔膜形成后选择持续、可解析的连接阶段，按固定的弧长、可见性和参数精度标准确定有效帧；不得选择误差最小的一帧。并不要求视频开头静止 0.5 秒。有效阶段内逐帧独立计算误差，再按真实时间权重聚合。
7. 稳定但错误弯向、错误半径比或可证实的平直隔膜不能因为不符合理论被当作跟踪失败；相反，隔膜不可见、太短或被高光遮住导致半径不可估时必须返回提取失败。

几何拟合不得加入 `r_s*(1/r_small-1/r_large)=1` 约束，否则评分只是重复拟合先验。不能从提示词直接给半径、曲率方向或理论正确性。

## 4. M1 连续评分规则

逐帧保留 bench 的无量纲误差，在预先确定的有效连接窗口内取按 PTS 加权的中位数：

```text
E_t = abs(r_s(t) * (1/r_small(t) - 1/r_large(t)) - 1)
E_M1 = time_weighted_median(E_t)
metric = clip(1 - E_M1 / error_at_zero, 0, 1)
error_at_zero = 1.0
```

不能先把三个半径各自平均再代入，因为它们必须来自同一帧几何。保存逐帧误差、有效时长和分位数，说明聚合范围。

例如 `r_small=100, r_large=200` 时，理论隔膜半径为 200：`r_s=200` 得 1.0，`r_s=100` 得 0.5，`r_s=400` 得 0.0；`r_s=-200` 反向弯曲得 0.0。数值单位只需一致。

* 所需圆弧可靠可测：`extract_success=true, status=scored`，按连续公式评分，不用二值容差替代。
* 两泡大小差可辨而隔膜可证实近似直线，且测量精度足以排除理论曲率：属于公式误差发散的极限，计 0.0。不要在 JSON 中写 Infinity；将该帧误差标为 null 并附 `error_kind=unbounded_straight_partition`。聚合时这些帧保留其时间权重，按发散误差排序，若加权中位数落于此类则最终得 0.0。
* 隔膜曲率太小但证据不足以区分平直和理论曲率、两泡几乎等大、有效可见弧不足或视角不可标定：false/null，不将“不确定的直线”当作确定错误。

建议可配置默认参数：

```text
min_radius_ratio = 1.10
min_partition_chord_small_diameter_ratio = 0.20
min_valid_duration_sec = 0.30
min_valid_frame_fraction = 0.80
max_relative_radius_uncertainty = 0.20
error_at_zero = 1.0
```

这些是测量和评分的工程参数，全部写入输出。有效帧比例只针对隔膜已形成的候选阶段，不因正常接触形成过程而拒绝整个视频。不添加 M2 的圆拟合残差分项。

## 5. 输出格式

```json
{
  "task_id": "P47",
  "video_path": "g9/P47/output_videos/minimax_h3/sample_00.mp4",
  "image_path": "g9/P47/first_frame.png",
  "video_prompt": null,
  "model": "minimax_h3",
  "sample_id": "sample_00",
  "seed": null,
  "metrics": {
    "M1": {"extract_success": false, "metric": null},
    "M2": {"extract_success": null, "metric": null}
  },
  "verbose": {
    "M1": {
      "principle": "独立测量两泡外弧与隔膜半径，计算 abs(r_s*(1/r_small-1/r_large)-1)。",
      "status": "extraction_failed",
      "measurements": {
        "r_small_px_median": null,
        "r_large_px_median": null,
        "r_partition_signed_px_median": null,
        "raw_m1_error": null,
        "valid_window_sec": null,
        "valid_duration_sec": null
      },
      "score_details": {"formula": "clip(1-E_M1/error_at_zero,0,1)", "range": [0, 1]},
      "thresholds": {},
      "evidence": [],
      "reason": "填写实际弧线、半径、聚合依据或失败原因"
    },
    "M2": null
  }
}
```

半径中位数仅作汇总展示，不能替代逐帧误差计算。示例不是视频实际评分，运行时填入实际提示词及数值。M2 仅兼容占位；缺依赖/权重为 `environment_error` 且 false/null。未知值为 null，禁止 NaN、Infinity 和 JSON 注释。

## 6. 实现与交付要求

* 不硬编码两泡位置、半径、隔膜弯向或最终正确关系。提供依赖、CUDA 设备和本地模型配置说明。
* 默认在 `g9/P47/eval_results/minimax_h3/debug_sample_00/` 保存 `small_bubble.mp4`、`large_bubble.mp4`、`partition.mp4`、两泡 mask 视频、`segmentation_overlay.mp4`、`circle_fit_overlay.mp4`；隔膜视频必须包含实际检测弧与端点。
* 保存无损 `masks.npz`、`timestamps.json`、`contours.json`、`circle_fits.json`、`radii_error.csv`、`radii_error.png` 和弯向判定关键帧。保留未拟合的原始边缘，不能只保存理想圆叠加图。
* 在 `debug_<sample_id>/calculation.md` 写出 bench 公式、三个符号的含义、同帧示例数值代入、逐帧聚合方法、具体得分和不确定性。
* 测试理想双泡、隔膜反向、平直隔膜、两泡近等大、短弧病态、镜面反光、遮挡和身份交换。
* 实现后实际运行本目录素材；未能可靠观察到隔膜时如实报告，不以生成提示词代替测量。
