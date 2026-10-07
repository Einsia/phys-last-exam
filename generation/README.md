# Video generation

`generate.py` reads each benchmark task's image and prompt, calls a model, and exports videos and a manifest for `evaluate.py`. The eight built-in adapters use the existing generation implementations, with model-specific dependencies kept in their own Python environments.

## One-command generation

The launch scripts create isolated Python environments and prepare only the model you select. Use Linux and Python 3.12; set `PLE_PYTHON` if the executable is not named `python3.12`.

For Seedance, set the Videos API base URL and enter the key at the hidden prompt:

```bash
export SEEDANCE_BASE_URL="https://your-provider/v1"
bash scripts/generate_seedance.sh
```

For unattended execution, supply `SEEDANCE_API_KEY` in the environment. Keys are never written into the generated configuration. The default model is `doubao-seedance-2-5-260628`; override it with `SEEDANCE_MODEL` when necessary. The default `SEEDANCE_REQUEST_FORMAT=litellm_json` places Seedance fields in the gateway's `extra_body`. For a direct AiHubMix JSON route use `aihubmix_json`; for a provider accepting uploaded `input_reference` files use `multipart`. Five-second requests use seed values as sample labels; the API metadata records `seed_sent: false`.

For an open-source model:

```bash
bash scripts/generate_open.sh cogvideox1.5-5b-i2v
```

Use any of the seven names in the table below. The launcher installs each generator into its own environment, downloads pinned checkpoints under `models/generation/`, and creates its path configuration automatically. Select another checkpoint location with `--model-root /path/to/checkpoints`, or GPUs with `--devices 1` / `--devices 0,1,2,3`. The MiniMax, Cosmos, and LingBot defaults select four GPUs; other defaults select one. GPU IDs are physical IDs, so use `--devices` for generation rather than nesting a separate `CUDA_VISIBLE_DEVICES` setting.

For gated checkpoints, accept the upstream license and configure `HF_TOKEN` or Hugging Face login first. Hunyuan also downloads its separate text and vision encoders; its FLUX.1-Redux-dev encoder requires approved Hugging Face access. Wan2.2 builds FlashAttention and needs a CUDA toolkit (`nvcc`) and a C++ compiler.

Each command defaults to 40 tasks × four samples. Run one sample first, then evaluate it:

```bash
bash scripts/generate_seedance.sh --tasks P21 --seeds 42 --output videos/smoke
bash scripts/evaluate_all.sh videos/smoke --output runs/smoke
```

The same task, seed, output, and dry-run flags work with `generate_open.sh`. `--dry-run` creates the input plan without downloading generator checkpoints, loading a generator, or calling the API. It can install the lightweight controller dependencies. Interrupted runs resume automatically. A failed generation exits nonzero; review its per-video worker log under `runs/generation/`.

## Reuse an existing installation

To reuse model environments and checkpoints you already have, initialize the path template and edit its Python, source, checkpoint, and GPU paths once:

```bash
python generate.py --init-config --model-root /path/to/checkpoints
```

Open the generated `generation.local.json`. For each local model, check `options.python_bin` (its Python executable), `options.model_dir` or `options.ckpt_dir` (weights), and `options.proj` (source checkout or working directory). Relative paths are resolved against this configuration file. The initializer detects common `source/PROJECT/.venv/` and sibling `envs/` layouts. The file is ignored by Git.

Replace `/path/to/checkpoints` with your generation checkpoint root. The initializer writes a template and refuses to overwrite an existing file; it does not download weights or create Python environments. Edit the existing file to connect an installation in a different layout. Keep the controller's evaluation environment active when invoking `generate.py`; the adapter launches `options.python_bin` for model inference.

Pass `--config generation.local.json` to either generation launch script to use that installation and skip automatic generator provisioning. To reuse a controller environment too, set `PLE_CONTROLLER_PYTHON=/absolute/path/to/its/bin/python`.

| Model name | Runtime / upstream setup | Checkpoint directory under `--model-root` |
| --- | --- | --- |
| `seedance-2.5` | [Videos API](https://docs.aihubmix.com/en/api/Video-Gen), configured below | None |
| `minimax-h3` | [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), Diffusers modular pipeline | `MiniMax-H3` |
| `cosmos3-super-image2video` | [Cosmos3](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video), Diffusers | `Cosmos3-Super-Image2Video` |
| `vbvr-wan2.2` | [VBVR-Wan2.2](https://huggingface.co/vinesnt/VBVR-Wan2.2), Diffusers | `VBVR-Wan2.2` |
| `wan2.2-i2v-a14b` | [Wan2.2](https://github.com/Wan-Video/Wan2.2), `source/Wan2.2/generate.py` | `Wan2.2-I2V-A14B` |
| `lingbot-video-moe-30b-a3b` | [LingBot](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b), `source/lingbot-video/` | `Lingbot-Video-MoE-30B-A3B` |
| `hunyuan-video-1.5-i2v` | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), `source/HunyuanVideo-1.5/` | `HunyuanVideo-1.5-I2V` |
| `cogvideox1.5-5b-i2v` | [CogVideoX1.5](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V), Diffusers | `CogVideoX1.5-5B-I2V` |

For the bundled Diffusers runners, `proj` can be any existing working directory. MiniMax's runner is bundled too; its `model_dir` must include the locally usable modular pipeline and all referenced components. Hunyuan needs its separate text/vision encoders as described in the upstream download instructions.

`options.devices` selects visible GPUs for each local model. MiniMax's bundled full-precision recipe requires four GPUs; the Cosmos and LingBot presets also select four. Other presets select one GPU with offloading where supported. The evaluation quickstart's single-GPU recommendation applies to evaluation, not to every video generator. Runs are sequential, so different generators can reuse the same devices.

For Seedance, configure a gateway that supports the JSON Videos API used by the existing experiments, or choose `request_format: "multipart"` for a compatible multipart endpoint:

```bash
export SEEDANCE_BASE_URL="https://your-video-gateway/v1"
export SEEDANCE_API_KEY="your-api-key"
```

Set `SEEDANCE_MODEL` if your provider uses a different model ID. API keys are read from the environment. Seedance requests five seconds by default; seeds are **sample labels**, because this API contract does not send a seed. The exported metadata records `seed_sent: false`.

Check only the models you intend to run:

```bash
python generate.py --models cogvideox1.5-5b-i2v --check
```

This checks paths and API configuration without inference or paid submissions. It does not prove that every upstream package version is compatible or that GPU memory is sufficient. `--dry-run` only plans inputs, so it also works before models are installed.

## Generate and evaluate

First, run one configured model on one task and validate its inputs:

```bash
python generate.py --models cogvideox1.5-5b-i2v --check
python generate.py --models cogvideox1.5-5b-i2v --tasks P21 --seeds 42 --output videos --resume
python evaluate.py --manifest videos/manifest.json --tasks P21 --output runs/input_check --dry-run
python evaluate.py --manifest videos/manifest.json --tasks P21 --output runs/quick_test --resume
```

Only after all eight environments and the Seedance endpoint are configured, expand to the full benchmark:

```bash
python generate.py --models all --check
python generate.py --models all --output videos --resume
python evaluate.py --manifest videos/manifest.json --output runs/input_check_all --require-all-tasks --dry-run
python evaluate.py --manifest videos/manifest.json --output runs/all_tasks --require-all-tasks --resume
```

The full generation plan has 1,280 samples (eight models × 40 tasks × seeds 42–45). A generation dry run creates a plan, not a manifest; generate actual videos before running the evaluation commands. The generated manifest references the exact input images, prompts, and annotation snapshots. Evaluation weights can live elsewhere through `FINAL_MODELS_DIR`; this variable is separate from the generation checkpoint paths in the configuration.

Use `--models NAME NAME` to choose several models and `--tasks P21 P2` to choose tasks. Each model's `num_frames` and `options` are configurable; legal frame-count adjustments and the measured output frame count/FPS are recorded. Automatic prompt rewriting is disabled so the recorded task prompt is the supplied conditioning prompt.

Output layout:

```text
videos/
├── manifest.json
├── .inputs/                       # Frozen images/prompts/annotations
└── MODEL/
    ├── g2_P21_seed42.mp4
    ├── g2_P21_seed42_config.json
    └── P3_gpt_01_modern_seed42.mp4
runs/generation/OUTPUT_ID/           # Plans, raw videos, attempt logs, API job IDs
```

The controller assigns the special calibrated filenames for P3/P8/P14/P17. For P3/P14, it exports a 1344 × 768 evaluation copy with an explicit spatial resize if needed; it retains the native raw video, records both dimensions and scale factors, and checks that frame count and timing are unchanged. This format conversion does not establish that the generated scene matches the calibration. Other tasks retain the native video dimensions.

P33/P34/P30/P36/P10/P38 include reviewed annotations for their fixed task image. Generation snapshots these templates and any supplied per-video annotation overrides into `.inputs/` and adds them to the manifest. Move the complete `videos/` directory, including its hidden `.inputs/` directory, to keep those inputs portable. Evaluation loads the annotations automatically, checks the image hash, scales the coordinates to the output resolution, and validates correspondence with the decoded first frame. A successful check records the current video/frame hashes in the evaluation debug output. Standard benchmark inputs need no separate annotation directory.

The coordinate mapping supports full-image resizing. If a crop or changed layout fails the first-frame correspondence check, initialization fails explicitly. To override initialization, place reviewed video annotations at `annotations/MODEL/TASK/VIDEO_STEM.json` and use `--annotation-root annotations`; a matching file overrides the bundled template. This flag does not change the task image or prompt. When evaluating a different first frame, supply its actual `image` and `prompt` in a custom manifest as shown in the [main README](../README.md#prepare-your-videos). Explicit video annotations keep their video/image hash checks. This flag is optional for both generation and evaluation. A generation resume updates the manifest snapshot when an override changes; the annotation is an evaluation input and does not require regenerating the video. Direct evaluation with this flag reads the external annotation directory, so keep it available for that command.

`--resume` skips only files whose input/settings/controller-code signature and video hash match. Changed settings or modified videos require a new output directory. Use a new output directory when changing model weights, upstream environments, or custom inference code too; those external files are not hashed. Failed samples remain in the manifest and produce explicit evaluator input errors instead of disappearing from the sample count. Check `runs/generation/OUTPUT_ID/summary.json` for failures.

Interrupted Seedance polling/downloads reuse the saved job ID. If submission timed out before an ID was received, the next run stops that sample rather than creating a duplicate paid job. Inspect that sample's `remote.json` and the provider's job history; remove the state only after confirming no job was accepted, or recover the original ID. Failed remote jobs are not automatically resubmitted.

## Add a custom model

Save this as `custom-model.json`, replacing the three example paths:

```json
{
  "models": {
    "my-model": {
      "backend": "command",
      "num_frames": 81,
      "cwd": "/path/to/my-model",
      "command": [
        "/path/to/my-env/bin/python", "/path/to/infer.py",
        "--image", "{image}", "--prompt", "{prompt}",
        "--seed", "{seed}", "--output", "{output}"
      ]
    }
  }
}
```

Then run `python generate.py --config custom-model.json --models my-model --output videos --resume`. You can also add this model entry to the `models` object in `generation.local.json` to keep all generators in one configuration. The command must write a playable MP4 to `{output}` and return zero on success. Arguments are passed directly, without a shell; multiline prompts and spaces are preserved. Other placeholders are `{num_frames}`, `{model}`, `{task}`, and `{sample_id}`.

Alternatively pass `--request {request}` to your script. That JSON file contains `image`, `prompt`, `seed`, `num_frames`, `output`, `model`, and `task`, as well as provenance fields. Write the video to its `output` path; the controller handles naming, validation, metadata, calibrated export, and evaluator manifests.
