# V3 视频物理评测

可靠性校准、四种测量状态、真值控制集和 Qwen-VL 对照流程见
[RELIABILITY.md](RELIABILITY.md)。

V3 先检查视频一致性，再决定是否运行物理评测。本地模式默认使用 Qwen3-VL-8B-Instruct，运行时只读本地权重；模型、虚拟环境和 MP4 不随源码仓库交付，需要单独准备。也可使用兼容的 HTTP VLM 服务。

- 默认通过阈值 **0.8**，一致性得分 **大于或等于 0.8** 才运行物理后端。
- 通过：`总分 = 0.15 × 一致性分 + 0.85 × 物理分`。
- 未通过：跳过分割、跟踪和物理后端，`总分 = 0.15 × 一致性分`。
- VLM 加载、解码、接口或响应格式出错：不运行物理后端，输出 `consistency_error`，总分为 `null`，避免把运行错误当成视频零分。

一致性检查采用中等严格标准：对视频内部持续的形状漂移、物体身份/部件数量变化、连接关系不稳定、重复性严重形变、复制/融合/分裂和异常瞬移进行扣分或拦截；纯运动模糊且主体结构稳定时不扣成失败。背景、场景、颜色、视角、物体布局变化，以及液面高低不合理、轨迹错误等不影响一致性，后者留给物理评测。判断使用覆盖视频首尾的 8 帧；参考首帧和生成提示词只存档，不发送给 VLM。

## 运行环境

下列公共命令在 `v3/` 目录执行。可使用已有 Python 环境；如果新建环境，可先运行：

```bash
python -m venv .venv
source .venv/bin/activate
export EVALUATOR_PYTHON="$(command -v python)"
```

按所选题目的 `evaluator/requirements.txt` 安装物理后端依赖，例如 G1/P1：

```bash
python -m pip install -r g1/P1/evaluator/requirements.txt
python -m pip install Pillow torch transformers accelerate
```

第二行用于本地一致性模型，需要支持 Qwen3-VL 的 Transformers 和与机器匹配的 PyTorch 环境。本地权重放在 `.models/Qwen3-VL-8B-Instruct/`，或通过 `VLM_MODEL` / `--consistency-model` 指定；HTTP 模式不需要本地 VLM 权重和加载栈，但仍需 OpenCV、Pillow 采样视频。SAM、CoTracker 等物理后端权重按所选题目单独配置。

批量 shell 脚本优先使用 `EVALUATOR_PYTHON`，否则使用 `v3/.venv/bin/python`。如果一致性模型和物理后端使用不同环境，可通过 `EVALUATOR_MEASUREMENT_PYTHON` 指定物理子进程的解释器。源码克隆后应按 `Pxx/data/metadata.json` 恢复视频和源文件名别名；登记输入会校验 SHA-256。metadata 中的路径以题目根目录为基准，不以 `data/` 为基准。

检查单视频和批量入口，不需要先加载模型或提供 MP4：

```bash
python g2/P19/evaluator/evaluate.py --help
EVALUATOR_PYTHON="$(command -v python)" bash g2/P19/scripts/run_eval.sh minimax_h3 --help
```

目录归档决策见 [RESTRUCTURE_NOTES.md](RESTRUCTURE_NOTES.md)，逐组的实际文件树和代码分类见 [SCHEMA_G1_G9.md](SCHEMA_G1_G9.md)。每题 README 给出本题输入、工作目录、当前入口和共享代码位置。

## 单视频

在 `v3/` 目录运行：

```bash
python g2/P19/evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image g2/P19/first_frames/provided/first_frame.png \
  --video_prompt_file g2/P19/prompts/video.txt \
  --output results/P19/result.json \
  --consistency-threshold 0.8 \
  --consistency-device cuda:0
```

40 个题目的 `evaluator/evaluate.py` 都使用该流程。阈值可调，例如 `--consistency-threshold 0.6` 会进一步放宽；模型、帧数、分辨率和 GPU 也可通过 `--consistency-*` 参数调整，详见入口的 `--help`。

## 批量

```bash
# 运行 data/metadata.json 中登记的视频；历史清单用于关联输入。
python scripts/run_all_eval.py \
  --tasks P19 --workers 1 --consistency-threshold 0.8
```

新结果默认写入 `v3/results/`，运行记录写入 `v3/work/v3_runs/`。默认单进程，避免同时加载多份 VLM 权重。可以通过 `--output-root`、`--log-dir` 改目录；无需调用历史 V2 的报告/重整脚本。现有的物理测量配置与输入关联规则继续适用，新视频需要提供正确的首帧、提示词及任务所需的标注/配置。

## 两个示例的完整验证

```bash
python scripts/smoke_consistency.py
```

该脚本需要额外准备其配置的 CogVideoX 和 Hunyuan `g2_P19_seed42.mp4`，这些视频未随源码交付。它执行真实 VLM 和物理流程，检查扭曲视频被拦截、连贯视频运行物理后端，以及最终分数满足 15%/85% 公式。输出位于 `work/consistency_smoke/`。视频文件名或生成模型名称不会作为 VLM 判分依据。

## 读取结果

- `verbose.M1._consistency`：一致性分、是否通过、阈值、理由、采样帧和模型信息。
- `verbose.M1._scoring_summary`：总分、一致性贡献、物理分及贡献、物理流程是否执行。
- `metrics.M1/M2.metric`：**纯物理分**，取值 0–1；不是 V2 中带识别奖励的分数。
- 调试目录下的 `consistency/`：输入采样帧、`request.json`、`response.json`、`consistency.json`。
- `scoring_calculation.md`：该视频的计分明细。

更详细的语义见 [SCORING_V3.md](SCORING_V3.md)。`SCORING_V2.md`、`OUTPUT_FORMAT_V2.md` 和已有 G1–G9 报告保留用于解释历史结果。

## 模型配置

历史运行使用 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 的 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 版本；本地模式默认使用 `cuda:0`。如需复用该版本，安装 Hugging Face CLI 后下载：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir .models/Qwen3-VL-8B-Instruct
```

如需替换模型，可设置 `VLM_MODEL` 或 `--consistency-model`。也支持兼容 Chat Completions 多图输入的 HTTP 服务：

```bash
python g2/P19/evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 --output results/P19/result.json \
  --consistency-backend http \
  --consistency-base-url http://localhost:8000/v1 \
  --consistency-model Qwen/Qwen3-VL-8B-Instruct
```

认证密钥从 `VLM_API_KEY` 环境变量读取，不写入评测记录。

## 测试

一致性阶段只根据视频画面判断时序视觉连贯性。参考图和完整生成提示词仅保留作审计，避免把背景/布景变化或“没有按要求运动”误判为失败；持续的形状/身份/连接关系不稳定和刚性装置拓扑改变等扭曲会扣分或拦截。物理规律仍由后续物理评测判断。

```bash
python -B -m unittest discover -s tests -p 'test*.py'
python -B -m unittest discover -s tests/refined -p 'test*.py'
```
