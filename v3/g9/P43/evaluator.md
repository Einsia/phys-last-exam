请为物理视频任务 P43 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。沿用 P34 的文档结构，以项目根目录 `bench.md` 的 M1 为准。

## 1. 任务定义

均匀矩形刚体在粗糙水平面上受到极缓慢水平推动，最终绕支撑底边倾倒，全程不滑动。

只评估 M1：

```text
Delta_N = N_COM_cross - N_tip
```

`N_COM_cross` 为质心的竖直投影越过倾倒支撑边的事件帧；`N_tip` 为倾倒 onset 的事件帧。两个事件越接近，M1 得分越高。保留有符号原始帧差，但评分使用差值的绝对大小。不得替换为“转了多少度”“转轴是否固定”或整段轨迹拟合分数。

bench 的 M2“倾倒前底面接触点保持静止”不单独评分；接触几何仅用于确定 M1 的支撑边和模型适用性。

**定义边界：** 质心越过支撑边是重力相对该边的力矩换向条件，不保证任何持续外力驱动下的首次转动都发生于此。当前 `video.txt` 描述执行器持续贴住、推动刚体的过程，与自由失稳并不完全等价。实现不得用 COM 越界来定义 `N_tip`，也不能为得到零帧差把两事件强行绑定；需记录 `prompt_benchmark_mismatch`。没有可独立辨认的倾倒 onset 时，应报告无法提取该 M1，而不是偷偷改指标。

## 2. 文件结构与接口

路径相对于项目根目录：

* `videos/g9/P43/continuation.mp4`：视频。
* `videos/g9/P43/first_frame.png`：对应首帧。
* `videos/g9/P43/video.txt`：内容作为 `video_prompt`。
* `videos/g9/P43/first_frame.txt`：首帧提示词，不作为 `video_prompt`。

在 `videos/g9/P43/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/utils/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python videos/g9/P43/evaluator/evaluate.py \
  --video_path videos/g9/P43/continuation.mp4 \
  --image_path videos/g9/P43/first_frame.png \
  --video_prompt "$(cat videos/g9/P43/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output videos/g9/P43/eval_results/minimax_h3/result_sample_00.json

bash videos/g9/P43/scripts/run_eval.sh minimax_h3
```

支持可选 `--model`、`--sample_id`、`--seed`、`--device` 和本地权重。示例显式关联当前输入与 `minimax_h3/sample_00`，不代表已存在 metadata。未知元数据写 null，不猜模型或种子。

批量脚本按自身位置定位任务目录，只读直属输入 `.mp4`，排除所有结果目录。通过 `metadata.json` 显式配对视频、图片、提示词、模型、样本；不得按遍历顺序编号或复用不匹配首帧。无样本 ID 回退视频 stem；未知模型目录用 `unknown_model`，JSON model 为 null。单样本失败不中断整批。

```text
videos/g9/P43/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

结果文件为 `result_<sample_id>.json`，中间产物为 `debug_<sample_id>/`。

## 3. 工具与测量流程

* PyAV：保留实际 PTS、原始帧序号及 time base。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：定位并分割刚体、地面接触区、执行器和推垫。
* OpenCV：矩形边线/角点拟合、地面线估计、刚体姿态与质心投影。
* CoTracker3 或 OpenCV 光流：辅助同一实体角点的连续关联，不能将遮挡预测视为观测。
* SciPy、NumPy：角度展开、时间局部回归、独立 onset 检测和事件差值。
* Matplotlib、Python json：可复核的事件曲线和结果。

接口参考：[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)、[CoTracker](https://github.com/facebookresearch/co-tracker)。

测量流程：

1. 自动分割刚体，分离推垫、地面和阴影，跟踪矩形实体四角及地面线。均匀矩形的 COM 为几何中心，可由四角均值或对角线交点得到；不能用包含推杆的检测框中心或遮挡后可见掩码质心替代。
2. 用背景/执行器固定部分校正相机运动；建立水平向右、竖直向上的校准平面。地面明显不水平或透视失真严重时，仅在可靠标定后计算重力投影，不得把像素竖直默认当真实重力方向。
3. 由实际倾倒方向与底边接触结构识别支撑边投影 `p(t)`，保持同一材料角点身份。局部素材中预期为右下角，但坐标和身份须从图像确认。记录滑移、变形及离地等适用性问题，不将这些诊断另设 M2 分数。
4. 定义沿地面、朝倾倒方向为正的 `d_COM(t)=projection(COM(t))-projection(p(t))`。独立检测其首次从支撑区域内负值越过零到外侧正值的事件。使用与位置噪声相符的滞回和持续时间去抖；事件帧回标到被确认的首次过零区间，而不是确认窗口末端。
5. 独立从 `theta(t)`、`omega(t)` 检测 `N_tip`：本实现将 onset 操作化为“从静止/准静态缓慢姿态变化进入持续倾倒转动的起点”。存在缓慢预转动时，使用角速度变化点；从完全静止直接倾倒时，使用持续角运动 onset。检测器不得读取 `d_COM`、`N_COM_cross` 或理论临界角来选 onset。
6. 变化点前后各拟合局部角速度，要求转动方向一致、后段持续且增量超过测量噪声；不能把单帧抖动当作倾倒。存在多个同样可信候选时报告事件不确定，不按与 COM 最接近者择优。
7. 持续匀速的执行器控制翻转、缺乏速度状态变化时，可能无法区分“慢推预倾斜”和“倾倒 onset”。此时保留角度和 COM 越界观测，`N_tip=null`，给 `no_identifiable_tip_onset`，不把视频第一帧、最终倒地帧或 COM 越界帧补作 onset。
8. 两个事件均映射回原始视频零基帧序号和 PTS，并保存局部插值时刻及不确定区间。不能在插帧、降帧视频的帧号上直接计算 bench 帧差。

不要求开头静止 0.5 秒；需要的是事件前后足够的局部观测证据。若首帧已处于倾倒过程，不能恢复缺失的 onset。对应首帧图片可以辅助身份和形状核验，不能充当一个不存在的时间窗口。

## 4. M1 连续评分规则

保留 bench 原始帧差，并利用真实时间戳实现帧率无关的归一化：

```text
Delta_N = N_COM_cross - N_tip
Delta_t = PTS[N_COM_cross] - PTS[N_tip]
metric = 1 / (1 + abs(Delta_t) / time_error_half_score_sec)
time_error_half_score_sec = 0.5
```

2026-09-07 按用户要求启用 `p43_soft_time_error_v2`，取消旧版“差至少 0.5 秒就截为零”的规则。默认同帧得 1.0，差 0.1 秒约 0.8333，差 0.25 秒约 0.6667，差 0.5 秒得 0.5，差 1 秒约 0.3333，差 2 秒得 0.2。有限时间差持续平滑扣分；分数不是物理成立概率。

对于恒定帧率，`Delta_t=Delta_N/fps`；可变帧率必须使用 PTS，不得用平均 fps 代替。原始事件、帧差及持续外力下的 onset 定义保留，不按评分结果移动事件。

时间尺度是公开、可配置的半分误差尺度，不是刚体理论常数，不随视频长度或测量结果自动改变。旧 `time_error_at_zero_sec` 已移除，配置使用 `time_error_half_score_sec`；两者含义不同。结果标注评分版本，新旧分数不能直接混用。

* 两个事件独立且可靠可测：`extract_success=true, status=scored`，即使帧差很大也正常给低分。
* 没有 COM 越界、没有可辨认 onset、事件截断或 COM/支撑边无法测量：`extract_success=false, metric=null`，因为帧差无定义。不得给缺失事件随意补帧号。
* 支撑边发生无法建模的滑移、刚体明显变形或透视无法校正：记录适用性/测量失败原因，不能套理想矩形公式伪造 COM。

建议可配置默认参数：

```text
onset_window_sec = 0.20
onset_hold_sec = 0.12
min_tip_speed_deg_per_sec = 5
onset_speed_ratio = 3
com_hysteresis_diagonal_ratio = 0.01
max_event_uncertainty_sec = 0.15
time_error_half_score_sec = 0.5
```

相对角速度门槛只在存在可测非零准静态速度时使用；近零基线用绝对门槛和噪声下限，避免除零。所有事件规则、默认值、候选和实际窗口写入结果；不使用附加几何分项改变 M1。

## 5. 输出格式

```json
{
  "task_id": "P43",
  "video_path": "videos/g9/P43/continuation.mp4",
  "image_path": "videos/g9/P43/first_frame.png",
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
      "principle": "独立测得质心投影越过支撑边与倾倒 onset，比较其原始帧号和时间戳。",
      "status": "extraction_failed",
      "measurements": {
        "N_COM_cross": null,
        "N_tip": null,
        "delta_frames": null,
        "t_COM_cross_sec": null,
        "t_tip_sec": null,
        "delta_time_sec": null,
        "event_uncertainty_sec": null
      },
      "score_details": {"version": "p43_soft_time_error_v2", "formula": "1 / (1 + abs(delta_time_sec) / time_error_half_score_sec)", "range": [0, 1], "half_score_error_sec": 0.5, "finite_error_cutoff": false},
      "thresholds": {},
      "evidence": [],
      "reason": "填写独立事件依据、适用性问题或失败原因"
    },
    "M2": null
  }
}
```

只计算 M1，M2 为兼容占位。示例不是当前视频结果。运行时补实际提示词与参数；未知值为 null，禁止 NaN、Infinity、JSON 注释。缺依赖或权重为 `environment_error`，M1 false/null。

## 6. 实现与交付要求

* 不硬编码支撑角、倾倒方向、事件帧、COM 越界时间或理论临界角。提供安装、设备和本地模型配置，支持 CUDA。
* 默认保存到 `videos/g9/P43/eval_results/minimax_h3/debug_sample_00/`：`block.mp4`、`block_mask.mp4`、`contact_region.mp4`、`actuator.mp4`、`segmentation_overlay.mp4`、`com_support_overlay.mp4`。
* 保存无损 `masks.npz`、`timestamps.json`、`corners.json`、`geometry.json`、`com_angle.csv`、`com_angle_events.png`、`event_candidates.json` 和两个事件的关键帧。保留原始帧映射和坐标校准，不只保存最终截图。
* `debug_<sample_id>/calculation.md` 写出两个事件各自如何独立得到、具体帧号/PTS、差值和归一化得分；说明当前推杆提示词与 bench 的适用性区别。
* 测试同时事件、提前倾倒、晚倾倒、仅匀速推转、无 COM 越界、首帧已倾倒、滑移、遮挡、角点错配及可变帧率。
* 实现后实际运行本目录素材；定义不适用或事件不可见时如实报告，不要通过修改 M1 让样本获得分数。
