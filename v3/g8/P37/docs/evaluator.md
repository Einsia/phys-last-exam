请为物理视频任务 P37 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。文档结构沿用 P34，任务与 M1 定义以项目根目录的 `bench.md` 为准。

## 1. 任务定义

两个相同的电磁跳环装置同时启动，一个放完整闭合铝环，另一个放带可见开口的同形铝环。两环从相同高度开始响应，线圈、铁芯和底座保持固定。

只评估 `bench.md` 的 M1：

```text
height_ratio = h_open / h_closed < 1
```

`h_open`、`h_closed` 是开口环、闭合环相对各自初始位置的最大向上升高量，不是环距画面底部的绝对高度，也不是最后一帧高度。bench 要求开口环相对升高更小，不要求闭合环绝对跳得越高越好。对外 `metric` 统一为 0–1，0 最差、1 最好；采用 `margin=0.2` 的达标后饱和评分，比值不超过 0.8 即满分，不继续奖励比值变小。不计算 M2，不把额外的运动幅度奖励混入 M1。

闭合与开口的身份必须由实际可见的环体和缺口确认，不能根据“哪个跳得高”反推身份。原理参考：[滑铁卢大学跳环演示](https://uwaterloo.ca/physics-astronomy/teaching-equipment/demonstration-equipment-catalogue/eddy-current-jumping-rings)。

## 2. 文件结构与接口

路径相对于项目根目录，命令从项目根目录执行。当前输入：

* `g8/P37/output_videos/minimax_h3/sample_00.mp4`：待评测视频。
* `g8/P37/first_frames/provided/first_frame_01.png`：对应首帧。
* `g8/P37/prompts/video.txt`：读取实际内容作为 `video_prompt`。
* `g8/P37/first_frames/provided/prompt.txt`：首帧提示词，不作为 `video_prompt`。

在任务目录内实现 `evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。

```bash
python g8/P37/evaluator/evaluate.py \
  --video_path g8/P37/output_videos/minimax_h3/sample_00.mp4 \
  --image_path g8/P37/first_frames/provided/first_frame_01.png \
  --video_prompt "$(cat g8/P37/prompts/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output g8/P37/eval_results/minimax_h3/result_sample_00.json

bash g8/P37/scripts/run_eval.sh minimax_h3
```

保留 `--model`、`--sample_id`、`--seed`、`--device` 和本地权重路径参数。上述命令显式指定当前输入对应 `minimax_h3/sample_00`；不要声称已有 metadata，也不要从路径猜生成模型或种子。批量模式按 `data/metadata.json` 的显式关联读取视频、首帧、提示词、模型和样本 ID；没有样本 ID 时回退到视频 stem，未知模型目录为 `unknown_model`，JSON 中未知元数据为 null。不得按文件遍历顺序临时生成 sample 编号或把一张首帧配给所有视频。

批量脚本根据自身位置定位任务目录，只扫描该目录直属的输入 `.mp4`，不递归扫描 debug 视频。输出约定：

```text
g8/P37/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

实际文件名为 `result_<sample_id>.json`、`debug_<sample_id>/`；目录必要时创建。单样本失败不能中断整批评测。

## 3. 工具与测量流程

* PyAV：解码并保留真实 PTS、time base 和帧序号。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：自动定位并分割环体、铁芯和固定线圈。
* CoTracker3：跟踪每个环上的多个实体点以及固定装置参考点；遵守可见性输出。
* OpenCV、SciPy、NumPy：稳健融合跟踪点、去除相机漂移、检测高度峰值。
* Matplotlib、Python json：曲线和结果输出。

接口参考：[CoTracker 官方仓库](https://github.com/facebookresearch/co-tracker)、[SAM 2 官方仓库](https://github.com/facebookresearch/sam2)、[Grounding DINO 官方文档](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。

测量流程：

1. 独立识别闭合环和开口环，保存缺口证据图。SAM 掩码不能把整个线圈或线圈顶端固定法兰当成自由环；环内孔、开口和被铁芯遮挡部分不能填成测量点。
2. 利用固定底座和线圈估计相机运动，建立竖直向上的测量轴。沿各自铁芯轴跟踪环的平移；不能把铁芯伸长、线圈升高或检测框变化算作跳高。
3. 初始位置优先取核验匹配的 `first_frames/provided/first_frame_01.png`，否则取视频第 0 帧的可靠观测。不要求视频开头静止 0.5 秒，允许第一帧后立即起跳；首帧已在空中且无初始参考时不能倒推起跳高度。
4. 以环体上同一组实体点的稳健位移估计环中心轨迹。相同成像比例时可用像素高度比；比例不同时，使用同形环的初始外径 `D_i` 归一化，记录单位为“环外径”。不允许用逐帧变化的检测框宽度动态消除真实形变。
5. 从完整响应段找主峰，包括其上升和峰顶/下降证据。对短时尖峰用固定时间尺度的中值平滑，不依据预期高低关系删点。若形成平台，平台需持续至少 `peak_window_sec`；视频结束时还在明显上升，峰值属于右删失，不当作已测到最高点。
6. 全程清晰而不动可测为零升高；看不见、身份丢失和视频截断不能记为零。无需检测开关画面，也不要求最终落回或末段静止。

在向上为正、校正相机运动后计算：

```text
u_i(t) = (z_i(t) - z_i(initial)) / D_i
h_i = max(0, max_t smooth(u_i(t)))
```

保存原始和平滑轨迹、实际峰值帧及时间、初始参考、尺度、误差估计。提示词不能代替观测。

## 4. M1 连续饱和评分规则

```text
margin = 0.2
当 h_closed 可分辨且 > 0：
    height_ratio = h_open / h_closed
    metric = clip((1 - height_ratio) / margin, 0, 1)
```

默认 `margin=0.2` 时：

* `height_ratio <= 0.8`：得 1.0，闭合环继续升高或开口环继续降低不再加分。
* `0.8 < height_ratio < 1`：连续线性过渡，例如比值 0.9 得 0.5，比值 0.95 得 0.25。
* `height_ratio >= 1`：得 0.0。

例如 `(h_closed, h_open) = (1, 0)`、`(1, 0.25)`、`(1, 0.8)` 均得 1.0，`(1, 0.9)` 得 0.5；两环同高或开口环更高得 0.0。两环高度同比放大不改变分数。`margin` 是已约定的工程评分过渡宽度，不是物理常数，也不是从 bench 推导出的唯一取值；该映射不是概率或二值通过/失败。允许配置，但须为有限数且满足 `0 < margin <= 1`，非法值应报配置错误，不得静默修正或用于除法。

边界处理：

* 两环全程可靠可见，闭合环确实未产生可分辨向上位移：`extract_success=true, metric=0.0`，原始比值为 null，记录 `zero_closed_height`。两环都不动不能通过 `0/0` 得到满分。
* 闭合环升高量接近测量噪声，无法区分静止和微小运动：`extract_success=false, metric=null`，不得用任意 epsilon 伪造可靠比值。
* 身份、初始位置或最高点无法可靠确定：提取失败，而不是物理低分。

建议默认参数，均须可配置并写入结果：

```text
margin = 0.2
smooth_window_sec = 0.08
peak_window_sec = 0.12
height_noise_diameter = 0.01
min_height_snr = 3
max_track_gap_sec = 0.10
min_valid_track_fraction = 0.90
```

`margin` 控制评分过渡宽度，其余参数控制测量可靠性，都不是 M2 分项。保存 `height_ratio`、实际使用的 `margin`、满分边界 `1-margin`、公式、分数范围及除零分支；不另外评分起跳同步性或显著运动幅度。

## 5. 输出格式

仅 M1 有实际结果；为兼容 P34，M2 只保留 null 占位。

```json
{
  "task_id": "P37",
  "video_path": "g8/P37/output_videos/minimax_h3/sample_00.mp4",
  "image_path": "g8/P37/first_frames/provided/first_frame_01.png",
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
      "principle": "比较开口环与闭合环相对初态的最大向上升高量之比。",
      "status": "extraction_failed",
      "measurements": {
        "h_closed_diameter": null,
        "h_open_diameter": null,
        "height_ratio": null,
        "closed_peak_time_sec": null,
        "open_peak_time_sec": null
      },
      "score_details": {
        "formula": "clip((1 - h_open / h_closed) / margin, 0, 1)",
        "range": [0, 1],
        "full_score_max_height_ratio": 0.8
      },
      "thresholds": {"margin": 0.2},
      "evidence": [],
      "reason": "填写实际评分依据或失败原因"
    },
    "M2": null
  }
}
```

这是结构示例，不是当前视频结果。运行时填入实际提示词、测量值和参数；成功状态为 `scored`。`full_score_max_height_ratio` 必须按实际 `margin` 计算为 `1-margin`，不能在更改配置后仍写 0.8。未知数值写 null，禁止 NaN、Infinity、JSON 注释。依赖或权重缺失使用 `status=environment_error`，M1 为 false/null，不能输出 0 分冒充评测完成。

## 6. 实现与交付要求

* 不硬编码特定视频坐标、峰值时刻或哪个环运动更大。提供安装、设备选择、本地权重和批处理说明；支持 CUDA，不把 GPU 编号写死。
* 默认保存中间产物到 `g8/P37/eval_results/minimax_h3/debug_sample_00/`：`closed_ring.mp4`、`open_ring.mp4`、`closed_mask.mp4`、`open_mask.mp4`、`segmentation_overlay.mp4`、`tracking_overlay.mp4`。
* 同时保存无损 `masks.npz`、`tracks.json`、`timestamps.json`、`heights.csv`、`heights.png`、`peaks.json`、`calibration.json` 和关键帧。视频需保留与原视频的逐帧映射、缺失标记和时间轴；有损 mask 视频只供查看，不能作为唯一测量数据。
* `debug_<sample_id>/calculation.md` 必须写出 bench 的 M1、实际初态和峰值窗口、原始高度、比值、实际 `margin`、是否进入满分饱和区、代入公式后的具体得分，以及零分/提取失败原因。
* 测试双环不动、仅开口环上升、同高、部分升高、峰值截断、线圈伪运动、身份切换和跟踪丢失；覆盖比值 0、0.8、0.9、0.95、1 和大于 1、满分区不再加分、同比缩放不变及非法 margin 配置。
* 实现后用当前素材实际运行并核验结果；只完成文档或环境准备时，不得编造已运行结果。
