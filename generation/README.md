# Video generation

`generate.py` reads each benchmark task's image and prompt, calls a model, and exports videos and a manifest for `evaluate.py`. The eight built-in adapters use the existing generation implementations, with model-specific dependencies kept in their own Python environments.

## Connect your model environments once

Install the models you want to run using their upstream instructions, or reuse your existing installations. The evaluation `setup.sh` installs the controller and video validation dependencies; generation model weights and their separate environments are configured here.

```bash
python generate.py --init-config --model-root /path/to/checkpoints
```

Open the generated `generation.local.json`. For each local model, check `options.python_bin` (its Python executable), `options.model_dir` or `options.ckpt_dir` (weights), and `options.proj` (source checkout or working directory). Relative paths are resolved against this configuration file. The initializer detects common `source/PROJECT/.venv/` and sibling `envs/` layouts. The file is ignored by Git.

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

```bash
# One task and one seed first.
python generate.py --models cogvideox1.5-5b-i2v --tasks P19 --seeds 42 --output videos --resume

# All 8 models, all 40 tasks, seeds 42/43/44/45: 1,280 videos.
python generate.py --models all --output videos --resume

# The generated manifest includes the exact input images and prompts.
python evaluate.py --manifest videos/manifest.json --annotation-root annotations --output runs/all_tasks --require-all-tasks --resume
```

Use `--models NAME NAME` to choose several models and `--tasks P19 P1` to choose tasks. Each model's `num_frames` and `options` are configurable; legal frame-count adjustments and the measured output frame count/FPS are recorded. Automatic prompt rewriting is disabled so the recorded task prompt is the supplied conditioning prompt. These defaults do not reproduce every historical benchmark generation setting.

Output layout:

```text
videos/
├── manifest.json
├── .inputs/                       # Frozen images/prompts; keep with manifest
└── MODEL/
    ├── g2_P19_seed42.mp4
    ├── g2_P19_seed42_config.json
    └── P3_gpt_01_modern_seed42.mp4
runs/generation/OUTPUT_ID/           # Plans, raw videos, attempt logs, API job IDs
```

The controller assigns the special calibrated filenames for P3/P6/P9/P11. For P3/P9, it exports a 1344 × 768 evaluation copy with an explicit spatial resize if needed; it retains the native raw video, records both dimensions and scale factors, and checks that frame count and timing are unchanged. This format conversion does not establish that the generated scene matches the calibration. Other tasks retain the native video dimensions.

P37/P38/P39/P41/P43/P49 still need annotations tied to the generated video's first frame. Generation marks these tasks in its metadata; it does not invent annotations or reuse old video bindings. Put reviewed annotations at `annotations/MODEL/TASK/VIDEO_STEM.json` and pass `--annotation-root annotations` to evaluation. You may also pass this flag to generation to include existing annotations in the manifest.

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
