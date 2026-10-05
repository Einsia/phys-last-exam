# PhysScope: A Multi-Domain Benchmark for Proxy-Based Evaluation of Video Physics

## Abstract

Visually convincing videos can still violate basic physical laws, making reliable physical evaluation essential for assessing video generation models. Existing evaluations often rely on learned judgments or reference videos, while direct physical tests largely focus on mechanics. We introduce PhysScope, a measurement-based benchmark comprising 40 controlled tasks spanning mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface-tension effects. Each task pairs an initial image and a generation prompt with observable quantities and predefined physical criteria. Our evaluation first screens for temporal consistency, excluding videos whose unstable object identities or structures make physical measurements unreliable. Videos that pass are then assessed through task-specific measurements, such as oscillation periods, reflection angles, and liquid levels. We distinguish failures of physical tests from cases with insufficient measurement evidence. We evaluate eight video generation models on 1,278 generated videos. Although 98.36% pass the temporal-consistency check, only 159 of the 716 videos with sufficient measurement evidence (22.21%) pass all required physical checks. These findings reveal a substantial gap between temporal coherence and physical consistency, highlighting the need to report measurement coverage alongside physical performance. PhysScope provides an interpretable framework for evaluating diverse physical phenomena while making the limitations of measurement explicit.

摘要保留论文版本的实验快照。本文末尾的模型表使用 **2026-10-04 更新后的任务可观测性门控与证据复核结果**，与摘要中的旧门控统计分开报告。

## 仓库结构

```text
.
├── easy/                    # 15 个任务
├── medium/                  # 15 个任务
├── hard/                    # 10 个任务
├── evaluate.py              # 批量评测入口，自动发现全部 40 个题包
├── requirements.txt
├── benchmark_results.csv    # 文末模型表的完整精度数据
├── benchmark_results.json   # 汇总、评分口径及来源 SHA-256
└── tests/                   # 批量调度、错误处理和续跑校验
```

每题包含 `task.md`、`first_frame.png`、`prompt.txt` 和 `evaluator/evaluate.py`；依赖的共享实现封装在该题的 `evaluator/_shared/` 中。输入视频、模型权重和评测输出放在题包外。

## 安装与模型配置

使用 Python 3.12，先安装与机器 CUDA/CPU 匹配的 PyTorch 和 torchvision，然后在仓库根目录安装其余依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# 在这个环境中安装匹配的 PyTorch / torchvision 后：
python -m pip install -r requirements.txt
export FINAL_MODELS_DIR=/absolute/path/to/physics_models
```

根依赖文件包含已使用的公共科学计算版本，以及 SAM 2、CoTracker 的固定源码版本。P6 的历史精确锁与公共 OpenCV/NumPy/SciPy 版本有差异，说明保留在 `requirements.txt` 中；本仓库没有声称完成所有平台的全新环境安装验证。

`FINAL_MODELS_DIR` 使用以下目录布局，也可以用右列环境变量指定已有资源：

| 资源 | 模型根目录内路径 | 单独指定路径 |
| --- | --- | --- |
| Grounding DINO Tiny | `grounding-dino-tiny/` | `EVALUATOR_GROUNDING_DINO_MODEL` |
| 官方 SAM 2.1 Small 权重 | `sam2.1-hiera-small/sam2.1_hiera_small.pt` | `EVALUATOR_SAM2_SMALL_CHECKPOINT` |
| Transformers SAM 2.1 Small | `sam2.1-hiera-small-transformers/` | `EVALUATOR_SAM2_TRANSFORMERS_MODEL` |
| Transformers SAM 2.1 Large | `sam2.1-hiera-large-transformers/` | `EVALUATOR_SAM2_LARGE_TRANSFORMERS_MODEL` |
| CoTracker3 权重 | `cotracker3/scaled_offline.pth` | `EVALUATOR_COTRACKER_CHECKPOINT` |
| CoTracker 源码 | `cotracker3/source/` | `EVALUATOR_COTRACKER_CODE` |

官方 SAM 2 的 `.pt` 文件与 Transformers 的完整模型目录使用不同接口。权重需自行准备；`pip install` 不会下载它们。部分融冰任务需要 CUDA。`EVALUATOR_MEASUREMENT_PYTHON` 可指定物理测量环境，`FINAL_TRACKING_PYTHON` 可另外指定 P3/P9 跟踪环境。

一致性检查支持本地模型或兼容的多图 HTTP 服务。批量评测建议保持模型服务常驻，避免每条视频重新加载模型。以下为在独立 vLLM 环境中启动 Qwen3.6-27B 的示例，参数说明见 [vLLM serve 文档](https://docs.vllm.ai/en/latest/cli/serve/)：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /absolute/path/to/Qwen3.6-27B \
  --served-model-name physics-vlm \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":64,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

在评测终端中配置服务：

```bash
export VLM_BACKEND=http
export VLM_BASE_URL=http://127.0.0.1:8000/v1
export VLM_MODEL=physics-vlm
# 有认证的服务：export VLM_API_KEY=...
```

物理模型与 VLM 可以使用不同 GPU。例如服务占用 GPU 0 时，在评测终端设置 `CUDA_VISIBLE_DEVICES=1`，并传入 `--measurement-device cuda:0`。也可以不用 HTTP 服务：设置 `VLM_BACKEND=local`、`VLM_MODEL=/absolute/path/to/Qwen3.6-27B`，由每个评测进程直接加载本地模型。

## 准备视频与标注

目录扫描接受 `mp4/mov/mkv/avi/webm`，顶层子目录为生成模型名，文件名或父目录中须含任务 ID：

```text
videos/
├── model-a/
│   ├── g2_P19_seed42.mp4
│   ├── P3_gpt_01_modern_seed42.mp4
│   └── ...                  # 其余任务及 seeds
└── model-b/
    └── ...
```

默认使用对应题包的首帧与 `prompt.txt`；若视频旁存在 `VIDEO_STEM_config.json`，则使用其中的实际生成提示词。非标准文件名、不同首帧或逐视频提示词可通过 JSON 清单指定，路径相对于清单文件：

```json
[
  {
    "model": "model-a",
    "task": "P19",
    "seed": 42,
    "sample_id": "g2_P19_seed42",
    "video": "videos/model-a/g2_P19_seed42.mp4",
    "image": "inputs/P19/first_frame.png",
    "prompt": "inputs/P19/actual_prompt.txt"
  },
  {
    "model": "model-a",
    "task": "P37",
    "seed": 42,
    "video": "videos/model-a/g8_P37_seed42.mp4",
    "annotation": "annotations/model-a/P37/g8_P37_seed42.json"
  }
]
```

使用 `--manifest manifest.json` 替代 `--video-root`。`image`、`prompt` 可省略以采用题包默认值；`backend_args` 可为某条视频追加测量参数，例如 `["--config", "/absolute/path/to/config.json"]`。清单中的 `model` 是视频生成模型，不是 VLM 名称。

P3/P6/P9/P11 使用场景标定：对应文件名分别为 `P3_gpt_01_modern_seedN.mp4`、`P6_gpt_01_modern_seedN.mp4`、`P9_gpt_01_modern_seedN.mp4`、`P11_gpt_01_30deg_seedN.mp4`。P3/P9 的测量画布为 1344×768；其余尺度条件见各题 `task.md`。批量入口不会改名或改变视频画面；自定义 `sample_id` 不能替代后端的场景文件名匹配。

P37/P38/P39/P41/P43/P49 需要对应实际视频的初始化标注。题内 `first_frame_annotations.json` 是已有标注示例，包含来源图像/视频绑定，不能直接当作任意新视频的标注。准备当前视频的对象位置和哈希后，通过清单的 `annotation` 指定，或按 `ANNOTATION_ROOT/MODEL/TASK/SAMPLE_ID.json` 存放并传入 `--annotation-root`。

## 一键批量评测全部任务

完成上述环境、视频和标注准备后，在仓库根目录执行：

```bash
python evaluate.py \
  --video-root /absolute/path/to/videos \
  --annotation-root /absolute/path/to/annotations \
  --output runs/all_tasks \
  --require-all-tasks \
  --measurement-device cuda:0 \
  --workers 1 \
  --resume
```

该命令扫描所有模型与全部 40 个任务，逐条调用各题的公开入口，并生成逐视频、逐任务及逐模型结果。`--require-all-tasks` 检查每个输入模型是否覆盖全部任务；缺少某个 seed 的视频不会被补成零分。相同模型/任务/样本 ID 下，只有视频、首帧、提示词、标注和参数一致的副本才会去重；冲突会明确报错。

在命令末尾加 `--dry-run` 可只检查文件、哈希、任务覆盖和重复项，生成 `input_manifest.json`，不加载模型。它不验证视频几何标定、神经权重是否可用或测量质量。只测部分任务可使用 `--tasks P19 P1`；输入目录只包含单一模型时，可用 `--model my-model` 指定名称。

`--resume` 仅复用输入哈希、代码、参数一致且结果通过校验的已完成样本；失败样本会重新执行，旧尝试保留在 `attemptNNN/`。更换同路径下的模型权重或 HTTP 服务模型版本时应使用新的输出目录。默认每条视频超时 1800 秒，可用 `--timeout` 调整；退出码 0 表示全部选中样本完成，2 表示输入或运行错误，130 表示中断。单条失败不会中止其他视频，也不进入均分分母。

## 单视频评测

例如从仓库根目录评测 P19：

```bash
python easy/P19/evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image easy/P19/first_frame.png \
  --prompt easy/P19/prompt.txt \
  --output runs/single/result.json \
  --consistency-backend http \
  --consistency-base-url http://127.0.0.1:8000/v1 \
  --consistency-model physics-vlm
```

所有任务使用同一公共输出结构；物理测量参数按各题 `task.md` 和入口 `--help` 配置。

## 输出与评分

```text
runs/all_tasks/
├── input_manifest.json
├── configurations/          # 每次启动的设置和代码签名
├── results.json / results.csv
├── by_model.csv / by_task.csv
├── summary.json
└── MODEL/TASK/SAMPLE_ID/
    ├── batch_record.json
    └── attempt000/
        ├── request.json
        ├── result.json
        ├── evaluator.log
        └── debug/           # 门控输入、回复及物理测量证据
```

单视频 `result.json` 中，`consistency` 保存门控结果，`physics.score` 保存物理分，`metrics.M1/M2` 分开记录原始量与归一化分，`score.total` 保存综合分：

```text
C = 一致性/任务可观测性得分，默认通过阈值 0.8
P = 已定义物理指标的平均分
S = 0.15 × C + 0.85 × P    （门控通过）
S = 0.15 × C               （门控拒绝，不执行物理测量）
S = null                   （输入、模型调用或程序执行错误）
```

完成提取但未观测到可计分现象的指标采用显式零分规则，同时保留空原始量、原因和测量覆盖率。未定义的 M2 不进入物理分分母。批量汇总的 `physics_mean_attempted` 只覆盖实际尝试且无运行错误的物理结果；被门控拒绝的视频没有独立物理分。

## 验证

```bash
python -B -m unittest discover -s tests -v
```

批量入口已通过 40 个题包发现、原始 1278 条视频清单检查，以及错误隔离、超时、重复输入、结果绑定和续跑测试。P1/P19 已通过实际视频解码与物理后端的批量集成检查；该集成检查使用明确标识的固定 HTTP 门控回复，只验证调用链，不是新的 VLM 评测。本次发布整理没有重跑全量神经推理。

## 当前视频生成模型表现

以下为 **2026-10-04 原始提示词实验**：8 个模型、40 个任务、1278 条视频，所有分数均为 **0–1，越高越好**，按复核综合分排序。Cosmos 缺少 P13/seed42、P5/seed45，共 158 条；其余每个模型 160 条。均分按各模型实际视频等权计算，不将缺失文件补为零分，增强提示词视频未混入。

“物理分”来自独立测量全部视频的对照流程；“测量覆盖率”是每条视频已定义指标中实际测得比例的均值。“自动综合分”使用原始自动门控；“复核综合分”使用冻结的证据复核通过标记，仍保留原始自动一致性分。新运行的批量入口执行自动评测，不套用历史复核结论，因此不能把复核列当作一次自动运行的结果。

Qwen 直接物理分是同批视频的 Qwen3.6-27B 视觉评分对照；唯一证据不足样本在该全样本均分中显式计为 0，原始回复仍为 null。完整精度数据见 [CSV](benchmark_results.csv)，实验口径与来源哈希见 [JSON](benchmark_results.json)。

| 视频生成模型 | 视频数 | 自动门控通过率 | 平均测量覆盖率 | 独立物理分 | 自动综合分 | 复核综合分 | Qwen 直接物理分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seedance-2.5 | 160 | 98.12% | 88.44% | 0.5111 | 0.5776 | 0.5790 | 0.6228 |
| minimax-h3 | 160 | 91.25% | 88.12% | 0.4912 | 0.5489 | 0.5489 | 0.5259 |
| cosmos3-super-image2video | 158 | 91.14% | 66.77% | 0.3570 | 0.4290 | 0.4290 | 0.4978 |
| vbvr-wan2.2 | 160 | 70.00% | 74.69% | 0.3753 | 0.3779 | 0.3835 | 0.3441 |
| wan2.2-i2v-a14b | 160 | 76.25% | 63.12% | 0.3311 | 0.3566 | 0.3656 | 0.4003 |
| lingbot-video-moe-30b-a3b | 160 | 78.12% | 53.12% | 0.2584 | 0.3225 | 0.3297 | 0.4072 |
| hunyuan-video-1.5-i2v | 160 | 75.62% | 53.44% | 0.2825 | 0.3197 | 0.3159 | 0.4153 |
| cogvideox1.5-5b-i2v | 160 | 51.25% | 38.44% | 0.1781 | 0.1881 | 0.1902 | 0.1541 |
| **总体** | 1278 | 78.95% | 65.77% | 0.3480 | 0.3900 | 0.3927 | 0.4208 |
