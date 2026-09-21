请为物理视频任务 P38 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。沿用 P34 的组织方式，以项目根目录 `bench.md` 的 M1 为准。

## 1. 任务定义

实心导体板和开槽导体板以相同初始条件在等效磁场区域内摆动，两者质量和转动惯量匹配。比较涡流阻尼导致的可辨认振荡次数差异。

只评估 M1：

```text
oscillation_count_ratio = N_solid / N_slotted < 1
```

`N_solid`、`N_slotted` 是同一观察时间段、同一有效振幅判据下的完整振荡次数。本实现明确采用“共同观察窗口内的有效完整周期数”，不把它称为无限时间内的总振荡次数。指数包络衰减率 `lambda_solid/lambda_slotted` 属于 bench 的 M2，本任务不拟合、不计算、不纳入评分。

对外 `metric` 为 0–1，使用 `p38_count_balance_v2` 平滑次数比评分：相同正次数给 0.5 中性分，实心板次数更少得分更高、更多得分更低。0.5 表示当前窗口未区分出次数差异，不代表已经满足阻尼目标；分数不是物理成立概率。原始 M1 次数比仍单独保存。

原理参考：[科罗拉多州立大学涡流摆演示](https://www.physics.colostate.edu/physics-demos/eddy-current-pendulum/)。质量、惯量及磁场相同是任务假设，不能声称仅凭图像验证了这些量。

## 2. 文件结构与接口

路径相对于项目根目录。当前输入：

* `videos/g8/P38/continuation.mp4`：视频。
* `videos/g8/P38/first_frame.png`：对应首帧。
* `videos/g8/P38/video.txt`：内容作为 `video_prompt`。
* `videos/g8/P38/first_frame.txt`：首帧提示词，不作为 `video_prompt`。

在 `videos/g8/P38/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/utils/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python videos/g8/P38/evaluator/evaluate.py \
  --video_path videos/g8/P38/continuation.mp4 \
  --image_path videos/g8/P38/first_frame.png \
  --video_prompt "$(cat videos/g8/P38/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output videos/g8/P38/eval_results/minimax_h3/result_sample_00.json

bash videos/g8/P38/scripts/run_eval.sh minimax_h3
```

支持可选 `--model`、`--sample_id`、`--seed`、`--device`、本地权重路径。示例命令显式关联当前视频与 `minimax_h3/sample_00`，不代表当前已存在 metadata。未知模型、种子等元数据写 null，不猜测。

批处理按 `metadata.json` 显式配对模型、样本、视频、图片和提示词，不能按遍历顺序编号。缺少 sample ID 回退视频 stem；未知模型目录使用 `unknown_model`，JSON model 仍为 null。仅扫描任务目录直属 `.mp4`，排除所有结果目录；脚本按自身位置定位素材，单个失败不中断整批。

```text
videos/g8/P38/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

文件名统一为 `result_<sample_id>.json` 和 `debug_<sample_id>/`，不能把多个输入自动配给同一首帧或提示词。

## 3. 工具与测量流程

* PyAV：逐帧解码，保留 PTS 和原始帧序号。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：定位、分割两块板、悬挂结构及磁极区域。
* CoTracker3：跟踪板上多个实体点和固定悬点；不能追踪槽内背景。
* OpenCV、SciPy `find_peaks`、NumPy：摆角估计、噪声过滤、极值和完整周期配对。
* Matplotlib、Python json：诊断图及结果。

接口参考：[CoTracker](https://github.com/facebookresearch/co-tracker)、[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。

测量流程：

1. 根据板面上可见的通槽区分实心板与开槽板，保持身份，不根据振动快慢交换标签。估计真实悬点、板的固定参考点和竖直方向；不能把边界框中心当作固定悬点。
2. 相机运动用支架参考点校正。由悬点到板参考点的方向计算有符号摆角 `theta_i(t)`，零位由重力竖直线和装置几何确定，不强行把视频末帧设为零。
3. 用核验匹配的首帧图片或视频第 0 帧建立初始角度；不要求开头静止 0.5 秒。若开头已运动但释放初态可由对应首帧可靠确定，允许立即计数；不能把任意中途帧当释放峰值。
4. 观察窗口 `W` 对两板相同，默认从可确认的释放参考时刻到视频结束；可配置固定时长，但不得对每块板各挑一段或事后选得分最高的窗口。末段不要求静止。记录仍在振荡的对象为 `right_censored=true`，它只表示未观察到自然停止，不妨碍报告窗口内计数。
5. 在 `theta_i(t)` 上用正负极值交替识别周期。一个完整周期由“同侧极值—另一侧极值—同侧极值”组成。固定首个有效极值的极性作为周期边界，按每两个半周期计一次；例如 `正—负—正—负—正` 计 2 次，不将全部重叠三元组计成 3 次。相邻周期可共用边界极值。首帧仅在能确认释放零速时可作为边界极值。
6. 设共同角振幅阈值 `A_cut=max(amplitude_floor_deg, amplitude_fraction*A_ref)`，其中 `A_ref` 为两板初始绝对角度的均值。完整周期的三个极值均须达到 `A_cut`，且峰突出度高于噪声。所有参数两板共享，禁止分别按各自最终振幅归一化。
7. 保存每个候选极值和周期的接受/拒绝原因。片尾未完成周期不计入 N，单独记录部分周期；有长时间遮挡时不能把缺失周期当成零。

实心板经过磁场后直接停住、没有足够极值，是可能的强阻尼表现；只要连续可见且未发生跟踪丢失，可以得到 `N_solid=0`，不能因为拟合不出指数包络而失败。开槽板没有足够可观察过程、视频太短等情况需要与真实零周期区别。

## 4. M1 平滑次数比评分规则（v2）

```text
当 N_slotted > 0：
    oscillation_count_ratio = N_solid / N_slotted
    metric = N_slotted / (N_solid + N_slotted)
           = 1 / (1 + oscillation_count_ratio)
当 N_slotted == 0 且已确认不是缺失观测：
    oscillation_count_ratio = null
    metric = 0.0
```

该规则对所有样本统一使用，不根据当前视频选阈值。次数同比放大不改变分数，也不单独奖励绝对次数增加。

| N_solid | N_slotted | 次数比 | metric |
| ---: | ---: | ---: | ---: |
| 0 | 5 | 0 | 1.0 |
| 1 | 3 | 0.3333 | 0.75 |
| 1 | 2 | 0.5 | 0.6667 |
| 1 | 1 | 1 | 0.5 |
| 2 | 1 | 2 | 0.3333 |
| 3 | 1 | 3 | 0.25 |
| 0 | 0 | null | 0.0 |

2026-09-07 按用户要求替换旧的 `clip((1-ratio)/0.1,0,1)`。旧规则把同次数记为 0，并在比值 0.9–1 内覆盖整个分数范围；新规则将相同正次数锚定为 0.5，在其两侧平滑变化。P38 不再接受旧 `margin` 参数或配置项，避免将无效配置静默忽略。不同评分版本的分数不可直接合并比较；结果中记录 `score_details.version`。

这是次数比的工程评分映射，不是二值 `<1` 判断。原始次数是离散测量，因此单个有限视频的可得分值有粒度；不得为制造小数而虚构周期或替换成 M2 衰减率。窗口末尾仍在运动时继续记录 `right_censored`，不推断无限时间总周期数。

* 在足够长、完整且可靠的观察下，`N_slotted=0`：输出 `extract_success=true, metric=0.0`，原始比值为 null，说明 `zero_reference_count`，避免 `0/0` 满分。
* 短片尚不足完成可辨认周期、释放参考缺失、身份不明或跟踪断裂：`extract_success=false, metric=null`。不能凭没有峰就推断板已停止。
* 观察窗口内相同正次数给 0.5 中性分；振幅衰减不额外加分。

建议可配置默认值：

```text
amplitude_floor_deg = 1.0
amplitude_fraction = 0.10
peak_prominence_deg = 0.5
smooth_window_sec = 0.08
min_observation_sec = 2.0
min_peak_separation_sec = 0.10
max_track_gap_sec = 0.10
```

`min_observation_sec` 只是最低采样要求，不保证任意摆长都能完成周期；必要时结合实际可辨周期尺度判断截断，禁止固定假设视频有几个周期。参数、窗口及计数规则必须写入 `verbose.M1`，包括评分版本、公式和中性分含义。v2 只改变评分映射，不改变计数或提取失败判据。

## 5. 输出格式

只计算 M1，M2 为兼容占位。

```json
{
  "task_id": "P38",
  "video_path": "videos/g8/P38/continuation.mp4",
  "image_path": "videos/g8/P38/first_frame.png",
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
      "principle": "在共同窗口和共同振幅阈值下比较实心板与开槽板的有效完整振荡次数。",
      "status": "extraction_failed",
      "measurements": {
        "N_solid": null,
        "N_slotted": null,
        "oscillation_count_ratio": null,
        "amplitude_cutoff_deg": null,
        "observation_window_sec": null,
        "right_censored": null
      },
      "score_details": {
        "version": "p38_count_balance_v2",
        "formula": "N_slotted / (N_solid + N_slotted) if N_slotted > 0 else 0",
        "range": [0, 1],
        "equal_positive_counts_score": 0.5,
        "zero_reference_score": 0.0,
        "interpretation": "0.5 表示共同窗口内次数相同；高于 0.5 表示实心板次数较少，低于 0.5 表示实心板次数较多。分数不是物理成立概率。"
      },
      "thresholds": {"amplitude_floor_deg": 1.0, "amplitude_fraction": 0.1, "peak_prominence_deg": 0.5, "smooth_window_sec": 0.08, "min_observation_sec": 2.0, "min_peak_separation_sec": 0.1, "max_track_gap_sec": 0.1},
      "evidence": [],
      "reason": "填写实际计数依据或失败原因"
    },
    "M2": null
  }
}
```

这是格式示例，不是视频实际评测。成功 `status=scored`，缺依赖或权重 `status=environment_error` 且 false/null。无法测量的数据为 null，不能用 0 替代；禁止 NaN、Infinity 和 JSON 注释。

## 6. 实现与交付要求

* 不硬编码板坐标、峰值数量、周期或预期阻尼关系。支持 CUDA、指定设备和本地 checkpoint，给出完整安装运行说明。
* 保存 `solid_plate.mp4`、`slotted_plate.mp4`、`solid_mask.mp4`、`slotted_mask.mp4`、`segmentation_overlay.mp4`、`tracking_overlay.mp4`，默认进入 `videos/g8/P38/eval_results/minimax_h3/debug_sample_00/`。
* 保存无损 `masks.npz`、`tracks.json`、`timestamps.json`、`angles.csv`、`angles_peaks.png`、`peaks.json`、`cycles.json` 和初态/悬点校准图。中间视频与原视频逐帧对应，裁剪坐标和无效帧必须可追溯。
* 在 `debug_<sample_id>/calculation.md` 列出观察窗口、阈值、每个有效周期的首尾时刻、两个 N、比值、评分版本、公式代入和具体得分；明确 0.5 的中性含义及没有计算 M2。
* 测试强阻尼无回摆、相同周期数、反向阻尼关系、微小噪声、半周期尾段、长遮挡和片段截断；覆盖 0/1/2/3 个周期的评分锚点、相同比值缩放不变、比值 1 两侧连续性及单调性、零参考次数及非法周期数。
* 实现后实际运行本目录素材；未运行时如实说明，不编造计数或评测结果。
