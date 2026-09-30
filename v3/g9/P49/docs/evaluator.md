请为物理视频任务 P49 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。沿用 P34 的组织方式，以项目根目录 `bench.md` 的 M1 为准。

## 1. 任务定义

两个相同材质、半径明显不同的小球在同一甘油槽中同时下落，并分别在触底前进入稳定终端速度阶段。

只计算 M1：

```text
E_M1 = abs((v_1/v_2) / (r_1/r_2)^2 - 1)
```

固定身份约定：球 1 为大球，球 2 为小球；`r_1>r_2`，`v_1`、`v_2` 为各自终端阶段沿重力向下的速度。该约定必须固定，因为交换标签会改变原始相对误差的数值，不能选择更有利的顺序。

Stokes 低 Reynolds 数模型在同材质、同流体、边界影响可忽略时给出 `v_terminal ∝ r^2`，不能只检查“大球更快”。依据见 [弗吉尼亚大学 Stokes 定律讲义](https://galileo.phys.virginia.edu/classes/152.mf1i.spring02/Stokes_Law.htm)。

终端速度波动 CV 是 bench 的 M2，不单独输出或加权成分数；局部速度稳定性仅用于确认 M1 所需的终端测量窗口。甘油黏度、密度和真实 Reynolds 数不可从提示词直接测得，缺少标定时应明确它们是模型假设。

## 2. 文件结构与接口

以下路径相对于项目根目录：

* `g9/P49/output_videos/minimax_h3/sample_00.mp4`：视频。
* `g9/P49/first_frames/provided/first_frame_01.png`：对应首帧。
* `g9/P49/prompts/video.txt`：实际内容作为 `video_prompt`。
* `g9/P49/first_frames/provided/prompt.txt`：首帧提示词，不作为 `video_prompt`。

在 `g9/P49/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python g9/P49/evaluator/evaluate.py \
  --video_path g9/P49/output_videos/minimax_h3/sample_00.mp4 \
  --image_path g9/P49/first_frames/provided/first_frame_01.png \
  --video_prompt "$(cat g9/P49/prompts/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output g9/P49/eval_results/minimax_h3/result_sample_00.json

bash g9/P49/scripts/run_eval.sh minimax_h3
```

支持 `--model`、`--sample_id`、`--seed`、`--device`、本地权重和标定参数。示例命令显式指定当前素材的 `minimax_h3/sample_00` 对应关系，不声称已存在 metadata；未知元数据写 null。

批量脚本按自身位置定位任务目录，仅读取直属输入 `.mp4`，不扫描中间视频。根据 `data/metadata.json` 显式配对模型、样本、视频、首帧和提示词，不能按遍历顺序生成 sample 编号，也不能将首帧或提示词错误复用。无 sample ID 回退视频 stem；未知模型使用目录 `unknown_model`，JSON model 为 null。单个失败不中断整批。

```text
g9/P49/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

结果和中间目录统一命名 `result_<sample_id>.json`、`debug_<sample_id>/`。

## 3. 工具与测量流程

* PyAV：解码，保留原始帧序号与真实 PTS。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：检测分割两球和槽体，排除释放架、反射像。
* CoTracker3：跟踪球上可见实体特征，结合轮廓圆心得到连续中心轨迹。
* OpenCV：球轮廓圆拟合、槽壁/底部/液面定位，相机漂移估计。
* SciPy、NumPy：时间局部回归、终端窗口识别、半径/速度比和不确定度。
* Matplotlib、Python json：位置、速度及半径曲线与结果。

接口参考：[CoTracker](https://github.com/facebookresearch/co-tracker)、[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。

测量流程：

1. 从实际球轮廓确定大球和小球身份并全程保持一致。外半径需由真实轮廓拟合，不能使用高光斑、反射像或框宽一半；不根据哪颗下落快交换身份。
2. 用槽体固定边缘校正相机运动，建立向下为正的竖直轴。沿各自轨迹测圆心 `y_i(t)` 和半径 `r_i(t)`，标记遮挡、出视野、明显形变和尺度漂移。
3. 核验两球是否处于可比较成像尺度。共同近正视成像可用 px 与 px/s；未知深度、明显透视或折射造成尺度差时，需可靠校准。不能逐帧改半径来强迫速度比例满足平方律。
4. 通过支架释放和球运动识别测量阶段；不要求视频开头静止 0.5 秒。若一开始已经下落，只要随后存在完整可靠终端窗口即可，不必恢复初始零速。
5. 在实际 PTS 上局部回归求速度与加速度。独立为每颗球寻找非零向下、持续近匀速的终端窗口，不能强制两球同时达到终端，也不能因速度比接近理论值而选窗口。
6. 默认选择每球满足预先配置稳定性和边界条件的最长连续窗口；相同长度时选择较早者。报告所有候选及选取规则。只比较终端速度，不取全程平均、最大速度或最后两帧差分。
7. 排除释放架接触、刚进入液面、接近底部减速和触底后的静止阶段。底部静止绝不是 `v_terminal=0`。侧壁接近、两球相互靠近可能改变阻力，须记录可见壁距、球间距和适用性警告；没有槽体三维尺寸时不编造壁面修正系数。
8. 在各自终端窗口用线性拟合 `y_i(t)=a_i+v_i*t` 得到速度，以相同窗口的稳健轮廓半径汇总 `r_i`。保存斜率置信区间、残差、速度变化诊断和边界证据，但不把 M2 CV 作为附加分数。

如果实际持续加速且还没进入终端状态，不能把一小段任意直线近似当作终端。理论不适用或视角无法测量时如实说明；不要将提示词中的“达到终端速度”作为测量证据。

## 4. M1 连续评分规则

保留原始误差，使用固定单调归一化：

```text
radius_ratio = r_1 / r_2
velocity_ratio = v_1 / v_2
normalized_velocity_ratio = velocity_ratio / radius_ratio^2
E_M1 = abs(normalized_velocity_ratio - 1)
metric = clip(1 - E_M1 / error_at_zero, 0, 1)
error_at_zero = 1.0
```

例如 `r_1/r_2=2`，理论速度比为 4：实测比值 4 得 1.0，3 得 0.75，2 得 0.5，1 得 0.25，8 得 0.0。这是 bench 相对误差的工程映射，不是概率。仅仅“大球稍快”不等于满分，也不能因大小速度顺序正确而二值通过。

边界情况：

* 两球半径与非零向下终端速度可靠可测：`extract_success=true, status=scored`，按公式计算。
* 可靠可见、已释放且远离边界的小球长时间悬停或向上运动，不能满足本任务的沉降比较：使用退化分支 0.0，保存有符号速度，原始未定义比值/误差为 null，明确 `invalid_settling_motion`。不能让两球同时反向运动却因速度比正确得满分。
* 接近零但无法与测量噪声区分、尚未进入终端阶段、片段太短、半径不可辨或触底前窗口不足：`extract_success=false, metric=null`。不得给分母加任意 epsilon，不能用触底静止替代有效终端速度。

建议可配置默认参数：

```text
terminal_window_sec = 0.50
max_relative_speed_change = 0.10
min_terminal_displacement_diameter = 1.0
min_speed_snr = 3
bottom_clearance_diameter = 2.0
min_radius_ratio = 1.10
error_at_zero = 1.0
```

`max_relative_speed_change` 指窗口内拟合加速度导致的速度变化量除以非零平均速度；这只是终端状态提取条件。`bottom_clearance_diameter` 是球表面到槽底间距相对该球直径的测量筛选值，不表示已证明无边界效应。参数须公开并可配置，不因某个样本得分低临时修改。

## 5. 输出格式

```json
{
  "task_id": "P49",
  "video_path": "g9/P49/output_videos/minimax_h3/sample_00.mp4",
  "image_path": "g9/P49/first_frames/provided/first_frame_01.png",
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
      "principle": "比较两球各自终端速度比与半径比平方，计算 bench 的无量纲相对误差。",
      "status": "extraction_failed",
      "measurements": {
        "r_1_large_px": null,
        "r_2_small_px": null,
        "v_1_large_px_per_sec": null,
        "v_2_small_px_per_sec": null,
        "radius_ratio": null,
        "velocity_ratio": null,
        "normalized_velocity_ratio": null,
        "raw_m1_error": null,
        "large_terminal_window_sec": null,
        "small_terminal_window_sec": null
      },
      "score_details": {"formula": "clip(1-E_M1/error_at_zero,0,1)", "range": [0, 1]},
      "thresholds": {},
      "evidence": [],
      "reason": "填写实际半径、终端窗口和速度依据或失败原因"
    },
    "M2": null
  }
}
```

这是结构示例，不是当前样本的运行结果。实际运行填写提示词、数值和参数；M2 仅兼容占位。缺少依赖/权重时 `status=environment_error`，M1 false/null。未知值写 null，禁止 NaN、Infinity 和 JSON 注释。

## 6. 实现与交付要求

* 不硬编码球坐标、半径、速度或终端时段。提供安装、CUDA 设备、本地权重及标定说明。
* 默认在 `g9/P49/eval_results/minimax_h3/debug_sample_00/` 保存 `large_ball.mp4`、`small_ball.mp4`、`large_mask.mp4`、`small_mask.mp4`、`segmentation_overlay.mp4`、`tracking_overlay.mp4`，保留触底和遮挡标记。
* 保存无损 `masks.npz`、`timestamps.json`、`tracks.json`、`radii.json`、`position_velocity.csv`、`position_velocity.png`、`terminal_windows.json` 和半径/边界拟合关键帧；中间视频保留逐帧原视频映射与裁剪坐标。
* `debug_<sample_id>/calculation.md` 写出两个固定身份、半径、各自终端窗口和速度、两个比值、原始误差以及最终具体得分；不能只写“大球较快”。
* 测试平方律、两球等速、速度比偏差、向上运动、零速度、只有加速段、触底停止、壁面影响、反光假球和尺度变化。
* 实现后用本目录素材实际运行并核验；无法可靠测量时如实报告，不编造终端速度。
