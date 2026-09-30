请为物理视频任务 P41 实现一个可运行的自动 evaluator，直接编写代码，完成单视频评测入口和批量评测脚本。组织方式沿用 P34，指标以项目根目录 `bench.md` 为准。

## 1. 任务定义

两个完全相同的透明漏斗分别装水和干燥非黏性沙子，从相同填充高度同时打开相同大小的出口，持续排出到接近空。评估排出流量对剩余填充高度的依赖关系，不比较谁先流完。

`bench.md` 中 P41 的 M1 以图片给出，这里转写成文本，避免依赖临时图片链接：

```text
Q ∝ h^beta
E_M1 = |beta_water - 0.5| + |beta_sand|
```

`Q` 为单位时间排出的体积或质量，`h` 为自由表面相对出口的填充高度，`beta` 为从数据拟合的指数。水的目标指数约为 0.5，沙子的目标约为 0；原始误差越小越好。只计算这一 M1，不新增 M2。

模型适用于固定开放出口、可近似恒定的流量系数以及非黏性干颗粒主体排出阶段。水的依据见 [OpenStax Torricelli 定律](https://openstax.org/books/college-physics/pages/12-3-the-most-general-applications-of-bernoullis-equation)；颗粒末期可能偏离恒流，参考 [Koivisto 与 Durian 的排料实验](https://www.nature.com/articles/ncomms15551)。不应将末尾颗粒突变强行拟合为全程定律。

## 2. 文件结构与接口

项目根目录下的实际输入：

* `g8/P41/output_videos/minimax_h3/sample_00.mp4`：视频。
* `g8/P41/first_frames/provided/first_frame_01.png`：对应首帧。
* `g8/P41/prompts/video.txt`：读取内容作为 `video_prompt`。
* `g8/P41/first_frames/provided/prompt.txt`：首帧提示词，不作为 `video_prompt`。

在 `g8/P41/` 内实现：`evaluator/evaluate.py`、`evaluator/requirements.txt`、`evaluator/`、`evaluator/README.md` 和 `scripts/run_eval.sh`。这些子路径均相对于本任务目录，不是项目根目录。

```bash
python g8/P41/evaluator/evaluate.py \
  --video_path g8/P41/output_videos/minimax_h3/sample_00.mp4 \
  --image_path g8/P41/first_frames/provided/first_frame_01.png \
  --video_prompt "$(cat g8/P41/prompts/video.txt)" \
  --model minimax_h3 --sample_id sample_00 \
  --output g8/P41/eval_results/minimax_h3/result_sample_00.json

bash g8/P41/scripts/run_eval.sh minimax_h3
```

支持 `--model`、`--sample_id`、`--seed`、`--device`、本地权重和几何标定配置。示例命令显式指定 `minimax_h3/sample_00`；不声称已有 metadata，未知种子为 null。

批量模式根据 `data/metadata.json` 显式配对视频、首帧、提示词、模型、样本；无样本 ID 回退视频 stem，未知模型目录 `unknown_model`，JSON model 为 null。不能按遍历顺序临时编号，不能把当前首帧或提示词自动套给全部视频。脚本根据自身位置定位任务，只读直属输入 `.mp4`，不扫描中间产物；单样本失败不中断整批。

```text
g8/P41/eval_results/<model>/
├── result_sample_00.json
├── result_sample_01.json
├── debug_sample_00/
├── debug_sample_01/
└── batch_summary.json
```

统一使用 `result_<sample_id>.json`、`debug_<sample_id>/`。

## 3. 工具与测量流程

* PyAV：解码并保存真实 PTS。
* Grounding DINO Tiny、SAM 2.1 Hiera Small：定位漏斗，逐帧分割容器、物料和出口流束。
* OpenCV：容器内壁、出口、自由表面轮廓和相机漂移测量。
* SciPy、NumPy：几何体积积分、时间序列平滑求导、稳健对数回归和不确定度估计。
* Matplotlib、Python json：高度、体积、流量及拟合图。

接口参考：[SAM 2](https://github.com/facebookresearch/sam2)、[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)。透明水体的整体掩码只提供 ROI，液面和内壁仍须由可见边界精测。

测量流程：

1. 从实际颜色、纹理、液面和颗粒边界确认水与沙，不能按流速快慢分配身份。定位出口高度和固定漏斗内壁，建立两侧独立的长度尺度。
2. 校正相机漂移；只有具备可靠几何依据时才校正透视和折射。记录漏斗是否可视作轴对称、内壁轮廓及标定误差，不能把外玻璃边界当内腔。
3. 逐帧测水面高度 `h_water(t)`。沙面可能为凹坑或斜面，须拟合可辨认的自由表面；本实现将 `h_sand(t)` 定义为轴线处沙面相对出口的高度，并记录拟合方法。不能把沙面无条件当水平面，无法可靠观测中心凹坑时不能凭轮廓猜体积。
4. 恢复剩余物料量 `V_i(t)`。对于可靠标定的轴对称漏斗，水体体积 `V_water(h)=pi*integral_0^h R(z)^2 dz`。沙体必须结合轴对称自由表面和内壁积分，不能直接套水平水面的 `V(h)`。例如已知自由表面 `z_surface(r,t)` 与底壁 `z_bottom(r)` 时，在内腔半径范围积分 `2*pi*integral r*max(z_surface-z_bottom,0) dr`；所有形状假设必须明示并有图像证据。
5. **禁止直接令 `Q=-dh/dt` 或 `Q=-d(mask_area)/dt`。** 漏斗横截面积随高度变化，二维投影面积也不等于三维体积。若只能得到投影面积而没有可靠体积转换，不可输出伪造的 beta。可提供基于标定流束截面积与真实物料速度的替代流量测量，但不能把光滑水面光流当作流体速度。
6. 在真实 PTS 上对 `V(t)` 用固定时间带宽局部回归求导，得到 `Q=-dV/dt`。像素体积或 `V/V_initial` 均可，只要每侧采用恒定比例，指数不受比例影响。沙体积转为质量只在堆积密度近似恒定的假设下成立，不能从提示词填密度。
7. 按实际观察窗口选择有限、正的高度与排出流量，要求流量超过测量噪声；按固定求导带宽间隔抽样，避免重复使用高度相关窗口。2026-09-07 按用户要求取消 `0.25 <= h/h_initial <= 0.85` 和 `max(h)/min(h)>=2` 两项门槛。不再因为视频仍接近初始填充高度或高度变化不足两倍而拒绝拟合；不根据分数挑区间。每帧接受/拒绝原因写入 `fit_details.json`。
8. 分别拟合 `log(Q)=alpha+beta*log(h)`，两个 beta 均为自由参数，不能固定为理论值。回归输入仅使用可靠且正的 Q、h；真实停流、反向流或不连续事件须另行记录，不能当噪声悄悄删掉。

拟合必须同时保存置信区间、拟合线及未被强制为理论值的散点。窄高度范围造成的宽区间只记为不确定度提示，不阻断评分；OLS 区间以已选数据和体积恢复模型为条件，不包含全部几何系统误差。高度完全恒定、没有正排出流量、有效独立点不足或几何恢复失败仍无法估计 beta。beta 明显偏离目标时按实测值给低分。不要求开头或结尾静止 0.5 秒，也不要求两侧恰好同一时刻流空。

## 4. M1 连续评分规则

保留 bench 原始误差，再做单调归一化：

```text
E_M1 = abs(beta_water - 0.5) + abs(beta_sand)
metric = 1 / (1 + E_M1 / error_half_score)
error_half_score = 1.0
```

2026-09-07 按用户要求启用 `p41_soft_exponent_error_v3`，取消误差达到 1 就截为零的规则。默认误差为 0 得 1，误差为 1 得 0.5，误差为 2 得 1/3，误差为 5 得 1/6；所有有限误差持续平滑扣分。例如 `(beta_water,beta_sand)=(0.5,0)` 得 1，`(0.4,0.1)` 约得 0.8333，`(0,0.5)` 得 0.5。`error_half_score` 表示半分误差尺度，是统一配置的工程映射，不随视频分数改变，也不是物理常数或概率。

* 两个 beta 可拟合：`extract_success=true`，按公式给具体分数，`status=scored`；窄范围拟合的不确定度另报。
* 无可靠体积/流量恢复、高度完全恒定、有效独立点不足、对数输入不可辨或遮挡：`extract_success=false, metric=null`，说明原因。取消固定相对高度窗口与两倍高度范围后，窄范围本身不再导致失败。
* 全程不排出时 `Q=0`，beta 无定义，输出 false/null 和 `no_observable_discharge`；不能将 `log(0)` 替成 epsilon，也不能捏造两个 beta 后评分。

建议可配置默认参数：

```text
derivative_window_sec = 0.25
min_independent_fit_points = 8
min_flow_snr = 3
error_half_score = 1.0
```

独立拟合点应来自不重叠或按相关长度抽样的窗口，不能把大量重叠平滑点当独立证据。默认参数不足以保证任意视频可测，实际适用性和采样限制必须报告。`fit_height_fraction_min`、`fit_height_fraction_max`、`min_height_ratio` 已从 P41 CLI/JSON 配置移除；旧配置需删除这三项。v3 继承 v2 的实际窗口拟合规则，保留原始指数误差与置信区间，仅将评分改为平滑映射。旧 `error_at_zero` 改为 `error_half_score`，含义为误差达到该值时得 0.5；新旧评分版本不可直接混用。

## 5. 输出格式

```json
{
  "task_id": "P41",
  "video_path": "g8/P41/output_videos/minimax_h3/sample_00.mp4",
  "image_path": "g8/P41/first_frames/provided/first_frame_01.png",
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
      "principle": "拟合 Q 与 h 的幂律指数，计算 abs(beta_water-0.5)+abs(beta_sand)。",
      "status": "extraction_failed",
      "measurements": {
        "beta_water": null,
        "beta_sand": null,
        "beta_water_ci": null,
        "beta_sand_ci": null,
        "raw_m1_error": null,
        "water_fit_window_sec": null,
        "sand_fit_window_sec": null,
        "volume_reconstruction_method": null
      },
      "score_details": {"version": "p41_soft_exponent_error_v3", "formula": "1 / (1 + (abs(beta_water - 0.5) + abs(beta_sand)) / error_half_score)", "range": [0, 1], "half_score_error": 1.0, "finite_error_cutoff": false, "fitting_policy": "observed positive-flow data; no relative-height window or minimum height ratio"},
      "thresholds": {},
      "evidence": [],
      "reason": "填写实际几何、拟合依据或失败原因"
    },
    "M2": null
  }
}
```

示例不是实际结果；运行时写入提示词内容、测量、窗口和参数。M2 仅兼容 null 占位，不计算。缺少依赖/权重使用 `status=environment_error` 且 false/null。未知值为 null，禁止 NaN、Infinity 和 JSON 注释。

## 6. 实现与交付要求

* 不硬编码液面位置、流完时间、漏斗尺寸或拟合指数。提供安装、设备、本地权重及几何配置说明，支持 CUDA。
* 默认在 `g8/P41/eval_results/minimax_h3/debug_sample_00/` 保存 `water.mp4`、`sand.mp4`、`water_mask.mp4`、`sand_mask.mp4`、`water_stream.mp4`、`sand_stream.mp4`、`segmentation_overlay.mp4`、`surface_tracking.mp4`。
* 保存无损 `masks.npz`、`timestamps.json`、`geometry.json`、`surfaces.json`、`height_volume_flow.csv`、`height_volume_flow.png`、`log_q_log_h.png`、`fit_details.json`，中间视频与输入逐帧对应并保留坐标变换。
* `debug_<sample_id>/calculation.md` 必须写出从 bench 图片转写的公式、体积恢复依据、求导方法、拟合区间、两个 beta、原始误差和具体分数；明确区分真实测量与轴对称等模型假设。
* 测试理想水/沙曲线、指数颠倒、恒定水流、零流量、窄高度范围、锥形漏斗几何、非水平沙面、曝光变化和遮挡。
* 实现后用本目录素材实际运行；无法可靠测量时如实报告，不得用“看起来水流慢了”等主观判断替代 beta。
