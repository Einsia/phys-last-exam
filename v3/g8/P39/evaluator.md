请为物理视频任务 P39 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。沿用 P34 的文档结构，以项目根目录 `bench.md` 的 M1 为准。

## 1. 任务定义

条形磁铁先在线圈附近静止，随后进入、穿过并离开固定线圈，最后再次静止；线圈连接一个双向均可发光的小灯。磁铁、线圈和灯均需可测量。

只评估 M1。`bench.md` 原指标为“磁通变化事件与灯亮事件的时序匹配 F1”；本任务采用已约定的可观测近似：**磁铁开始进入、开始离开线圈这两个几何事件，与灯亮事件的时序匹配 F1**。

* 进入时刻 `t_enter`：沿穿过方向的磁铁前端开始越过线圈入口平面。
* 离开时刻 `t_exit`：同一前端开始从线圈另一端露出，不等整根磁铁完全离开。
* 每个进出事件与亮灯起始时刻允许前后各 `0.5 秒` 的偏差；一对一匹配，不用区间 tIoU。

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
M1 = F1 = 2*TP / (2*TP + FP + FN)
```

F1 本身即 0–1 分数，0 最差、1 最好，不再转换成二值通过/失败。bench 的 M2“静止亮度接近基线、快速通过时亮度峰值”不作为额外评分；亮度基线只用于 M1 的灯亮事件检测。

这项评分衡量的是进出线圈与亮灯的时序关系，不声称直接测得磁通量或其变化边界。不再使用人为近场范围或磁通模型确定事件，也不要求磁铁在线圈内部运动时始终亮灯。上述几何事件和时间容差是本任务的评测约定，不是法拉第定律给出的精确亮灯时刻。

## 2. 文件结构与接口

以下路径相对于项目根目录：

* `videos/g8/P39/continuation.mp4`：视频。
* `videos/g8/P39/first_frame.png`：对应首帧。
* `videos/g8/P39/video.txt`：实际内容作为 `video_prompt`。
* `videos/g8/P39/first_frame.txt`：首帧提示词，不作为 `video_prompt`。

在 `videos/g8/P39/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/utils/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python videos/g8/P39/evaluator/evaluate.py \
  --video_path videos/g8/P39/continuation.mp4 \
  --image_path videos/g8/P39/first_frame.png \
  --video_prompt "$(cat videos/g8/P39/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output videos/g8/P39/eval_results/minimax_h3/result_sample_00.json

bash videos/g8/P39/scripts/run_eval.sh minimax_h3
```

支持可选 `--model`、`--sample_id`、`--seed`、`--device` 及本地权重。示例命令显式指定当前输入的模型和样本名，不代表已有 metadata；缺失元数据写 null，不能猜种子。

批量模式读取 `metadata.json` 的显式视频—首帧—提示词—模型—样本对应关系。无 sample ID 时回退视频 stem；未知模型目录为 `unknown_model`，JSON model 为 null。不能按遍历顺序编号、错配首帧或将当前提示词复用于所有视频。脚本按自身位置定位任务目录，只扫描直属输入 `.mp4`，不扫描 debug，单样本失败不中断整批。

```text
videos/g8/P39/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

结果和中间产物分别命名为 `result_<sample_id>.json`、`debug_<sample_id>/`。

## 3. 工具与测量流程

* PyAV：解码和真实 PTS；所有事件容差以秒表达。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：定位磁铁、线圈、灯泡，保存对象分割。
* CoTracker3、OpenCV：磁铁实体点与线圈口跟踪，计算轴向相对位置、速度和灯丝/发光区域亮度。
* NumPy、SciPy：时间滤波、进出事件时刻及亮灯区间提取、一对一时间点匹配。
* Matplotlib、Python json：时序图和结果输出。

接口参考：[CoTracker](https://github.com/facebookresearch/co-tracker)、[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。

测量流程：

1. 建立线圈轴、两个实际线圈口及磁铁两端的几何关系，校正相机运动。根据实际穿过方向确定磁铁前端、入口和另一端出口，不硬编码从左向右。交叉边界来自可见线圈口，不能用检测框边缘或磁铁中心代替。
2. 用可见实体点估计相对运动，核验穿入和穿出次序。短时遮挡可以用两端可见部分和刚体长度约束连接，但必须记录推断区间和不确定度；不能把模型在完全遮挡时的预测坐标当作观测真值。
3. 独立于灯亮轨迹提取两个带标签的时间点 `G={enter:t_enter, exit:t_exit}`。入口事件由磁铁前端从入口外侧跨入内侧确定；出口事件由该前端首次从另一端露出确定。用空间滞回和短时持续确认去除边界抖动，确认后将事件回标到首次交叉帧，不使用确认窗口末端。记录原始帧号、真实 PTS、必要的局部插值及时间不确定度；禁止根据灯何时亮反调这两个时间点。
4. 在灯泡掩码内定位实际发光区域，排除玻璃反光、支座和电线。亮度使用固定可追溯 ROI，并用邻近非发光背景校正曝光变化，保留原始及校正后的序列。不能用全画面平均亮度作为灯亮证据。
5. 用可靠不发光样本估计暗态基线及噪声，采用阈值滞回提取灯亮事件集合 `L`，每段保存 `[t_on,t_off]`，匹配只使用暗转亮起点 `t_on`。初始帧不能无条件视为暗态；若灯开始就明显发光，应保存左删失亮段、将未知 `t_on` 写为 null，不得把视频第 0 帧伪造为亮灯起点，也不能扣除基线后让持续亮段消失。无可靠暗态标定、严重饱和或反光无法区分时报告测量不确定，而不是伪造事件。
6. 匹配容差为进出事件前后各 `onset_tolerance_sec=0.5` 秒，即总宽度 1 秒的候选时间窗，不要求亮灯持续 0.5 秒。只比较实际事件时刻，不自由平移灯曲线以最大化 F1。不强制初始静止 0.5 秒；静止样本只需足以估计基线和运动噪声，记录实际时长。
7. 保存进出事件帧、亮段起止帧、PTS、可见性以及检测依据。未匹配的亮段计 FP，未匹配的进出事件计 FN。一个亮段只有一个起点，不能因其持续覆盖两个进出时刻而重复计 TP。无需由提示词保证事件存在。

不能只看首尾状态或最大亮度，也不能把进、出两个几何事件合并成一个“运动中”区间。`merge_gap_sec` 只用于合并因亮度噪声造成的短暂熄灭间隔，合并亮段只保留一个起点；进、出时刻即使相距小于 1 秒、候选时间窗重叠，也保持两个独立参考事件。

## 4. M1 评分规则

对进出事件时刻 `t_j`（`t_enter` 或 `t_exit`）和可测亮灯起点 `t_on_k`，匹配候选仅要求：

```text
onset_tolerance_sec = 0.5
abs(t_on_k - t_j) <= onset_tolerance_sec
```

容差包含边界：提前或延后恰好 0.5 秒均为候选，超过 0.5 秒不匹配。进出是时间点，不再计算或设置区间交并比门槛。例：进入时刻为 1.0 秒，亮灯起点为 1.4 秒可匹配，1.6 秒不能匹配；不能因为某个亮段覆盖 1.0 秒就忽略其真实起点。

在候选图上先求最大匹配数量，再选择总绝对时间差最小的匹配；仍有平局时按事件时间和 ID 确定性排序。不要用逐个就近贪心而漏掉本可匹配的第二个事件。匹配必须一对一，即使容差窗重叠，同一次亮灯也不能算两次。`TP` 为匹配对数，`FP=len(L)-TP`，`FN=len(G)-TP`。

```text
metric = 2*TP / (2*TP + FP + FN)
```

例如 2 个参考事件均匹配且无额外闪光，得 1.0；只匹配其中 1 个且无 FP，得 2/3；2 个均匹配但另有 1 个 FP，得 0.8；全部漏检或不匹配得 0.0。原始事件 F1 的取值受事件数限制，这是 bench 指标本身的粒度，不得添加 M2 分数伪造更细小数。

边界情况：

* `len(G)>0, len(L)=0` 且两种观测可靠：`extract_success=true, metric=0.0`。
* 可靠完整观察证实没有任何规定的磁铁通过事件：使用退化分支 `metric=0.0`，即使灯也没亮也不能给空集满分；记录 `no_reference_events`，不宣称测到了正常进出响应。
* 正常评分需确认同一次完整穿过的两个几何事件；只拍到进入而没拍到离开、交叉时刻被遮挡或灯泡无法测亮度时，`extract_success=false, metric=null`。不能只保留可见的一个事件算出满分，也不能把“看不到”当成“没有”。
* 起始已亮的左删失亮段没有可测起点，不参与候选匹配，但保留为未匹配亮段计 FP；若其未知起点可能落在某个事件的容差窗内，匹配是否成立不可确定，应返回提取失败。若两个容差窗均完整可见、灯始终常亮且没有新亮灯起点，则没有 TP，不能当作两次正确响应。

建议可配置默认值：

```text
crossing_hysteresis_coil_length_ratio = 0.01
crossing_hold_sec = 0.08
brightness_on_sigma = 5
brightness_off_sigma = 3
min_event_duration_sec = 0.08
merge_gap_sec = 0.08
onset_tolerance_sec = 0.5
```

亮度噪声须包含量化噪声下限，不能在基线标准差为零时把任意像素波动认作闪光。`onset_tolerance_sec` 必须为有限非负数；空间滞回和持续确认只用于稳健定位几何交叉，不得把它们变成对磁通的推断。容差、事件定义和退化分支写入 `verbose.M1`；不额外计算 M2。

## 5. 输出格式

```json
{
  "task_id": "P39",
  "video_path": "videos/g8/P39/continuation.mp4",
  "image_path": "videos/g8/P39/first_frame.png",
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
      "principle": "独立测量磁铁前端开始进入和开始从另一端露出的时刻，与亮灯起点按前后各0.5秒容差一对一匹配，计算事件F1。",
      "status": "extraction_failed",
      "measurements": {
        "t_enter_sec": null,
        "t_exit_sec": null,
        "crossing_events": null,
        "light_events": null,
        "matches": null,
        "TP": null,
        "FP": null,
        "FN": null,
        "precision": null,
        "recall": null
      },
      "score_details": {
        "formula": "2*TP/(2*TP+FP+FN)",
        "range": [0, 1],
        "matching_rule": "abs(t_light_onset - t_crossing) <= onset_tolerance_sec"
      },
      "thresholds": {"onset_tolerance_sec": 0.5},
      "evidence": [],
      "reason": "填写实际事件匹配依据或失败原因"
    },
    "M2": null
  }
}
```

示例不是当前视频评分，运行时填实际提示词、容差和测量值；若修改容差，`principle` 的描述也须与实际配置一致。`crossing_events` 记录事件类型、交叉帧、PTS 和时间不确定度，`light_events` 记录亮段起止时间及删失标志，`matches` 记录事件 ID 和实际有符号时间差。未测出的事件列表为 null，已确认无事件才为 `[]`。成功状态为 `scored`；缺依赖或权重为 `environment_error` 且 false/null。禁止 NaN、Infinity、JSON 注释和用零代替未知值。

## 6. 实现与交付要求

* 不硬编码磁铁轨迹、线圈口、灯泡 ROI 或亮灯时刻。提供安装、CUDA 设备和本地权重配置说明。
* 默认在 `videos/g8/P39/eval_results/minimax_h3/debug_sample_00/` 保存 `magnet.mp4`、`coil.mp4`、`lamp.mp4`、对应 mask 视频、`segmentation_overlay.mp4`、`event_overlay.mp4`。
* 保存无损 `masks.npz`、`tracks.json`、`timestamps.json`、`motion_brightness.csv`、`motion_brightness.png`、`events.json`、`event_matches.json`、`brightness_calibration.json`。所有视频保留原始帧对应关系和真实时间轴。
* `debug_<sample_id>/calculation.md` 写明前端穿入口与从出口露出的实际帧/PTS、亮灯起点、前后各 0.5 秒候选窗、每一对匹配的时间差、TP/FP/FN、F1 的数值代入和失败原因；明确本指标没有测量磁通量。事件叠加视频和曲线标出两个几何事件及各自容差窗。
* 测试全暗、全程亮、额外闪光、只亮一次、正确两次、提前/延后恰好 0.5 秒、偏差超过 0.5 秒、候选窗重叠但只有一次亮灯、最大匹配优先于贪心、曝光变化、交叉抖动、遮挡和片段截断。
* 实现后用当前素材实际运行；没有运行或模型不可用时必须如实说明。
