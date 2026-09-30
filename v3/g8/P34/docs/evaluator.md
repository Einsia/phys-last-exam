请为物理视频任务 P34 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。

任务与主指标以项目根目录 `bench.md` 的 P34 行为准，只计算 M1。M2 仅保留统一 JSON 的 null 占位，不实现通电前方向一致性或通电同步性的独立评分。

1. 任务定义

两个完全相同的装置并排放置，每个装置由一根竖直导线和一个小指南针组成。两根导线中的电流大小相同、方向相反，同时通电，随后两个指南针自由转动至稳定。

上述为 `bench.md` 的任务定义。当前目录的 `first_frames/provided/prompt.txt`、`first_frames/provided/first_frame_01.png` 和 `prompts/video.txt` 实际采用“一根中央竖直导线、左右等距各一个指南针”的素材变体。应在 `verbose.M1` 中记录这一场景差异，不声称检测到了两根导线，也不修改素材或将提示词当作测量证据；本次仍保留 bench 指定的反向、近等幅偏转 M1。

只评估 M1：根据两个指南针相对于各自初始方向的有符号偏转量 δ₁、δ₂，对**偏转方向相反、幅度接近**的程度输出连续的 0–1 得分，0 最差、1 最好。理想关系为：

```text
δ₁ × δ₂ < 0
|δ₁ + δ₂| 足够小
```

2. 文件结构与接口

以下路径均相对于项目根目录，示例命令从项目根目录执行。当前任务素材为：

* `g8/P34/output_videos/minimax_h3/sample_00.mp4`：待评测视频。
* `g8/P34/first_frames/provided/first_frame_01.png`：对应首帧。
* `g8/P34/prompts/video.txt`：视频生成提示词，读取文件内容作为 `video_prompt`。
* `g8/P34/first_frames/provided/prompt.txt`：首帧生成提示词，不作为 `video_prompt`。

在任务目录内实现：

* `g8/P34/evaluator/evaluate.py`：单视频评测入口。
* `g8/P34/evaluator/requirements.txt`：依赖。
* `g8/P34/evaluator/`：检测、跟踪和测量工具。
* `g8/P34/scripts/run_eval.sh`：批量评测入口。

单视频调用：

```bash
python g8/P34/evaluator/evaluate.py \
  --video_path g8/P34/output_videos/minimax_h3/sample_00.mp4 \
  --image_path g8/P34/first_frames/provided/first_frame_01.png \
  --video_prompt "$(cat g8/P34/prompts/video.txt)" \
  --model minimax_h3 \
  --sample_id sample_00 \
  --output g8/P34/eval_results/minimax_h3/result_sample_00.json
```

保留可选的 `--model`、`--sample_id`、`--seed` 参数。当前 `data/metadata.json` 明确记录生成模型 `minimax_h3`、样本 ID `sample_00`；随机种子未知，写 `null`。

批量调用：

```bash
bash g8/P34/scripts/run_eval.sh minimax_h3
```

脚本根据自身位置定位 `g8/P34/`，默认读取该目录直属的待评测 `.mp4` 文件，不递归扫描结果目录。按 metadata 中的模型名和样本 ID 输出 `eval_results/<model>/result_<sample_id>.json`，必要时创建目录。样本编号必须有显式对应关系，不能按遍历顺序临时编号；无样本 ID 时才回退到视频 stem，未知模型使用目录占位 `unknown_model`，JSON 中模型仍写 null。当前素材输出至：

```text
g8/P34/eval_results/minimax_h3/result_sample_00.json
```

当前 `data/metadata.json` 明确关联 `sample_00`、`output_videos/minimax_h3/sample_00.mp4`、`first_frames/provided/first_frame_01.png` 和 `prompts/video.txt`。增加其他视频时，在 metadata 中登记对应的 `sample_01`、`sample_02` 等 ID；不能凭文件遍历顺序配对，也不能自动将当前首帧和提示词用于所有视频。缺失的元数据写 `null`。单个样本失败不能中断整批评测。中间产物放在对应模型下的 `debug_<sample_id>/`，批量汇总放在该模型的 `batch_summary.json`。

3. 工具与测量流程

采用以下工具：

| 工具                                                      | 用途                          |
| ------------------------------------------------------- | --------------------------- |
| PyAV                                                    | 解码视频，保留真实时间戳                |
| Grounding DINO Tiny：`IDEA-Research/grounding-dino-tiny` | 通过 Transformers 自动定位两个指南针   |
| SAM 2.1 Hiera Small                                     | 根据检测框分割并跟踪两个指南针区域           |
| OpenCV                                                  | 提取针体、旋转中心和同一磁极针尖，通过光流辅助帧间关联 |
| NumPy                                                   | 计算有符号角度、角度展开、稳定窗口统计和 M1     |
| Matplotlib、Python json                                  | 保存调试曲线和评测结果                 |

模型接口参考 [Grounding DINO 官方文档](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) 和 [SAM 2 官方仓库](https://github.com/facebookresearch/sam2)。

测量流程：

* 使用 Grounding DINO 定位左右指南针，再用 SAM 2 跟踪各自区域，保持对象身份一致。
* 在各区域内使用 OpenCV 的颜色分割、轮廓分析和直线拟合提取针体，持续追踪同一磁极端点，例如红色针尖。指南针整体掩码不能直接当作针体，检测框中心不能直接当作旋转中心。
* 统一角度正负约定，避免针尖切换和 ±180° 边界造成虚假旋转。不能利用“两针应该反向”的预期修正观测。
* 存在明显透视时，仅在有可靠几何依据时校正；无法可靠测量则返回提取失败。
* 初始方向优先从对应的首帧图片测量：先验证图片与视频第 0 帧的场景对应关系，再独立验证两个轴心及同一磁极针尖的方向一致性。图片不可用或核验不通过时，使用视频第 0 帧的可靠针尖观测，并记录初始参考的实际来源。取消“视频开头必须静止 0.5 秒”的要求；允许磁针从开头立即运动。
* 最终方向仍从末段稳定窗口估计，默认时长至少 `stable_window_sec`、角度极差不超过 `stable_angle_range_deg`。必须检查整个过程的连续轨迹，不能只比较首尾两帧。第 0 帧无法可靠测量、轨迹无法连续关联或末段不稳定时，返回提取失败。

沿可靠的连续轨迹展开角度后计算：

```text
δ₁ = θ₁_post − θ₁_pre
δ₂ = θ₂_post − θ₂_pre
```

所有测量必须来自实际视频或对应首帧，video_prompt 不能作为指标通过的证据。

4. M1 连续评分规则

测量成功时 `metric` 为 `[0.0, 1.0]` 的浮点数，越大越好，按有符号偏转残差归一化：

```text
S = |δ₁| + |δ₂|
symmetry_score = 1 - |δ₁ + δ₂| / S
motion_score = min(1, min(|δ₁|, |δ₂|) / min_deflection_deg)
metric = clip(symmetry_score * motion_score, 0, 1)
```

若 `S = 0`，直接令 `metric = 0.0`，避免除零和“两针不动却满分”。`symmetry_score` 衡量反向偏转的相互抵消程度，`motion_score` 连续降低偏转不足的得分。反向等幅且运动充分得 1.0；反向但幅度不匹配得到中间值；同向偏转、仅一根针转动或均不动得 0.0。例如偏转 `(30, -20)` 得 0.8，`(30, -10)` 得 0.5，`(2, -2)` 得 0.4，`(30, 20)` 得 0.0。

这是工程定义的连续归一化分数，不是概率，也不是把二值通过/失败改写为浮点数。原来的方向、最小偏转和绝对/相对容差布尔检查保留为 `verbose.M1.conditions` 诊断信息，不用于把分数截成 0 或 1。`verbose.M1.score_details` 必须记录公式、归一化误差、两个分项和分数范围。

默认阈值：

```text
min_deflection_deg = 5
symmetry_abs_tol_deg = 5
symmetry_rel_tol = 0.20
stable_window_sec = 0.5
stable_angle_range_deg = 3
initial_reference_tolerance_deg = 5
```

这些是工程容差，必须支持配置并记录到结果中。`stable_window_sec` 和 `stable_angle_range_deg` 仅用于末段稳定窗口；`initial_reference_tolerance_deg` 用于首帧图片与视频第 0 帧的针尖方向核验，不要求初始阶段静止。

区分以下情况：

* **测量成功**：返回 `extract_success = true, metric = 实际归一化得分`，`status = scored`。分数可为区间内任意值；同向或零偏转得 0.0，反向但幅度不匹配按公式连续扣分。
* **提取失败**：缺少初始状态、过程截断、无法确认最终稳定状态，或无法可靠跟踪针尖。返回 `extract_success = false, metric = null`。

两根针都不转动不能因为偏转和为零而通过。无需检测通电时刻，不能因缺少开关或指示灯画面而判定提取失败。

5. 输出格式

严格输出合法 JSON。只计算 M1，M2 按统一格式保留 null 占位，不实现计算逻辑。

```json
{
  "task_id": "P34",
  "video_path": "g8/P34/output_videos/minimax_h3/sample_00.mp4",
  "image_path": "g8/P34/first_frames/provided/first_frame_01.png",
  "video_prompt": null,
  "model": "minimax_h3",
  "sample_id": "sample_00",
  "seed": null,
  "metrics": {
    "M1": {
      "extract_success": false,
      "metric": null
    },
    "M2": {
      "extract_success": null,
      "metric": null
    }
  },
  "verbose": {
    "M1": {
      "principle": "计算两根指南针相对于各自初始方向的有符号偏转，判断方向是否相反、幅度是否接近。",
      "status": "extraction_failed",
      "measurements": {
        "theta_1_pre_deg": null,
        "theta_2_pre_deg": null,
        "theta_1_post_deg": null,
        "theta_2_post_deg": null,
        "delta_1_deg": null,
        "delta_2_deg": null,
        "opposite_direction": null,
        "absolute_deflection_sum_deg": null,
        "applied_symmetry_tolerance_deg": null
      },
      "thresholds": {},
      "evidence": [],
      "reason": "填写实际结果或提取失败原因"
    },
    "M2": null
  }
}
```

上述 JSON 是结构示例；使用本目录素材评测时，`video_prompt` 应填入 `g8/P34/prompts/video.txt` 的实际内容。`verbose.M1` 必须展示测量依据，包括初始参考来源及核验结果、实际使用的末段稳定窗口、角度、偏转量、阈值、各条件判断和失败原因。无法测得的数值写 null，不能用零代替。禁止输出 NaN、Infinity 或 JSON 注释。

6. 实现与交付要求

* 不要硬编码特定视频的指南针坐标、方向或最终角度。
* 提供依赖安装、模型权重配置和运行说明，支持指定设备及本地权重路径。
* 缺少依赖或权重时明确报告环境错误，不能当作物理指标不通过。
* 保存跟踪叠加图、左右指南针分割视频、逐帧角度数据和角度曲线，当前样本保存至 `g8/P34/eval_results/minimax_h3/debug_sample_00/`。
* 同时保留无损分割掩码、真实 PTS、逐帧原视频对应关系和校准信息；有损 mask 视频只用于查看。在 `debug_<sample_id>/calculation.md` 写出实际初态与末段窗口、两针偏转、各分项及最终分数的数值代入，不只输出公式。
* 检查角度跨界、针尖切换、同向偏转、零偏转、幅度不匹配及跟踪失败等情况。
* 使用现有的 `g8/P34/output_videos/minimax_h3/sample_00.mp4` 和对应的 `g8/P34/first_frames/provided/first_frame_01.png` 实际运行；依赖或权重缺失时如实说明完成的检查，不得编造评测结果。
