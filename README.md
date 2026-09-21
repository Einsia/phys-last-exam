# VDMBench V3 Evaluator

这是当前 V3 视频物理评测器的独立仓库。它包含 G1–G9 的 40 个物理任务、统一输出契约、视频内时序一致性门控、物理测量后端、首帧/任务配置素材、批量运行脚本、报告脚本和审计脚本。

仓库地址：`git@git-ssh.xiaoaojianghu.fun:yuno/vdmbench.git`

## V3 评测流程

每个视频先由本地 Qwen3-VL-8B-Instruct（或兼容的 HTTP VLM）抽取均匀分布的 8 帧，只判断视频内部的时序视觉连贯性。规则会忽略背景、场景布局、颜色、镜头变化、动作是否完成和物理规律本身；会对持续的主体形状/身份变化、部件复制、融合、分裂、刚性结构拓扑变化和稳定场景中的异常跳变扣分。

默认一致性阈值是 `0.80`。通过后才运行物理测量；总分为：

```text
总分 = 0.15 × consistency + 0.85 × physics
```

未通过一致性门控时跳过物理后端，只保留 `0.15 × consistency`。VLM 或程序运行错误不会伪装成 0 分，而会在结果的 `score_status`、`_consistency` 和 `_provenance` 中记录原因。

## 仓库内容

- `v3/unified_evaluators/`：V3 runtime、consistency gate、物理评分映射、JSON contract 和 backend executor。
- `v3/g1/` … `v3/g9/`：40 个任务的 evaluator、配置、metadata、任务 README 和首帧/模拟素材。
- `v3/scripts/run_all_test.py`：针对外部 `all_test` 视频目录的正式批量评测。
- `v3/scripts/report_all_test.py`：生成 `评测报告.md`、CSV 和 JSON 汇总；Markdown 中包含表格和嵌入的 HTML 总表。
- `v3/scripts/audit_all_test.py`：检查输入哈希、门控短路、权重公式和结果证据。
- `v3/tests/`：V3 一致性和统一输出契约的单元测试。
- `bench.md`：任务目录、难度、物理现象和 prompt 定义。
- `materials_manifest.json`：随仓库提供的首帧素材路径、大小和 SHA-256 清单。
- `scripts/download_models.py`：下载 Qwen、Grounding DINO Tiny 和 SAM 2.1 Small 到 evaluator 约定路径。

首帧图片和任务配置素材随仓库提供。生成视频、逐视频结果、debug 日志、虚拟环境和模型权重不提交到 Git；它们体积大且属于运行时数据。

## 安装

先安装目标机器对应 CUDA 版本的 PyTorch/torchvision，再安装其余依赖：

```bash
git clone git@git-ssh.xiaoaojianghu.fun:yuno/vdmbench.git
cd vdmbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
```

如果目标机使用其他 CUDA 或 CPU，请替换第一条 PyTorch 安装命令。安装完成后下载运行时模型：

```bash
# 一致性门控（约 8B VLM，需按机器显存准备）
python scripts/download_models.py --qwen

# P34 指南针任务的 Grounding DINO Tiny + SAM 2.1 Hiera Small
python scripts/download_models.py --p34
```

模型下载脚本只写入被 `.gitignore` 忽略的目录。也可以手动提供等价的本地模型路径。

## 输入目录和环境变量

正式批量脚本读取外部数据根目录，不会把视频复制进本仓库。数据根目录需要至少有：

```text
<VDMBENCH_DATA_ROOT>/data/videos/all_test/<model>/gpt/g1_P1_seed42.mp4
<VDMBENCH_DATA_ROOT>/data/g1_g9/g1/P1/first_frame.png
<VDMBENCH_DATA_ROOT>/data/g1_g9/g1/P1/video.txt
```

其中 `all_test/<model>/gpt/` 下的视频文件名应符合 `g[1-9]_P*_seed*.mp4`。设置：

```bash
export VDMBENCH_DATA_ROOT=/path/to/vdmbench-data
export V3_OUTPUT_ROOT=$PWD/runs/v3_evaluator
export VLM_MODEL=$PWD/v3/.models/Qwen3-VL-8B-Instruct
export VLM_DEVICE=cuda:0
```

`VDMBENCH_DATA_ROOT` 也可以指向包含 `data/` 的原始 vdmbench 根目录。`V3_OUTPUT_ROOT` 默认为仓库外的 `runs/v3_evaluator`，用于避免把结果写进源代码树。`EVALUATOR_MEASUREMENT_PYTHON` 可指定物理后端使用的 Python 解释器；未设置时使用当前解释器。

## 单视频运行

以下示例会先做一致性门控，再按门控结果决定是否运行 P19 物理后端：

```bash
python v3/g2/P19/evaluator/evaluate.py \
  --video /path/to/video/g2_P19_seed42.mp4 \
  --image "$VDMBENCH_DATA_ROOT/data/g1_g9/g2/P19/first_frame.png" \
  --video_prompt_file "$VDMBENCH_DATA_ROOT/data/g1_g9/g2/P19/video.txt" \
  --output "$V3_OUTPUT_ROOT/single/g2_P19_seed42.json" \
  --debug-dir "$V3_OUTPUT_ROOT/single/g2_P19_seed42" \
  --model demo-model --seed 42 \
  --consistency-model "$VLM_MODEL" \
  --consistency-device "$VLM_DEVICE" \
  --consistency-threshold 0.80
```

任务入口也支持 `--consistency-backend http`，配合 `VLM_BASE_URL`、`VLM_MODEL` 和可选的 `VLM_API_KEY` 使用 OpenAI-compatible `/chat/completions` 接口。默认 local backend 使用 `local_files_only=True`，不会偷偷下载另一个模型。

P34 还需要 Grounding DINO 和 SAM 2 checkpoint；下载到默认路径后可直接运行，也可以显式传 `--dino_model`、`--sam2_checkpoint` 和 `--sam2_config`。

## 正式批量运行、报告和审计

G3 的 P3/P9 需要先为每个外部视频生成首帧关联配置，G8/G9 的标注任务需要先生成标注输入。首次运行时先执行：

```bash
python v3/scripts/prepare_all_test_g3.py
python v3/scripts/prepare_all_test_annotations.py
```

这两个步骤只写入 `$V3_OUTPUT_ROOT/inputs/`，不会改写外部视频。

先只生成任务清单检查权限和输入关联：

```bash
python v3/scripts/run_all_test.py \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --threshold 0.80 \
  --prepare-only
```

确认清单后去掉 `--prepare-only` 正式运行。也可用 `--models model-a model-b`、`--tasks P19 P34`、`--seeds 42` 和 `--limit N` 缩小范围。结果写入 `$V3_OUTPUT_ROOT/v3_<model>/`，全局清单和状态写入 `$V3_OUTPUT_ROOT/`。

```bash
python v3/scripts/report_all_test.py
python v3/scripts/audit_all_test.py
```

报告脚本会根据仓库根目录的 `bench.md` 读取难度和物理现象，生成：

```text
$V3_OUTPUT_ROOT/评测报告.md
$V3_OUTPUT_ROOT/model_summary.csv
$V3_OUTPUT_ROOT/per_video.csv
$V3_OUTPUT_ROOT/summary.json
$V3_OUTPUT_ROOT/audit.json
```

审计通过的必要条件包括：每个输入结果的 SHA-256 对应、门控失败后没有 backend 调用、通过/失败样本的总分公式一致、VLM 采样证据存在，以及没有混用不同阈值或 rubric。

## 测试

不加载视频或模型即可运行静态测试：

```bash
python -m unittest discover -s v3/tests -p 'test_*.py' -v
python v3/scripts/run_all_test.py --help
```

`report_all_test.py` 读取正式运行产物后直接执行，不需要单独的 `--help` 模式。

需要 GPU 和 Qwen 权重的测试/评测不会在安装或导入阶段自动下载模型。若只想验证一致性短路逻辑，可在拥有示例视频后运行：

```bash
python v3/scripts/smoke_consistency.py --source-root "$VDMBENCH_DATA_ROOT"
```

## 输出契约和评分文档

- [V3 scoring specification](v3/SCORING_V3.md)
- [V2 output contract](v3/OUTPUT_FORMAT_V2.md)
- [Benchmark task catalog](bench.md)

结果 JSON 的公共顶层字段固定为 `task_id`、`video_path`、`image_path`、`video_prompt`、`model`、`seed`、`metrics`、`verbose`；V3 一致性、物理是否执行、权重计算和运行证据位于 `verbose.M1` 的 `_consistency`、`_scoring_summary` 和 `_provenance`。
