#!/usr/bin/env python3
"""Measure the rolling time of the sphere and ring in P7.

The pipeline is deliberately split into two stages:

1. SAM 3 gets two instance masks from text (and, when useful, an optional
   point/box prompt) on frame zero and propagates those masks through the
   video.
2. CoTracker3 tracks points sampled *inside each SAM mask*.  The median of
   the visible points is used as the object's centre, which is less noisy
   than following one arbitrary pixel on the rim.

The time measurement is made in image coordinates along the top-to-bottom
   incline direction.  For a reproducible experiment, pass ``--top-point``
   and ``--finish-point`` using pixel coordinates from the first frame.  If
   they are omitted, the direction and endpoints are estimated from the
   first and last valid centre positions.

SAM 3 and CoTracker are optional imports so that ``--help`` and static checks
   work on a machine without GPU dependencies.  Actual inference requires a
   CUDA installation of the official SAM 3 package and PyTorch/CoTracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class ObjectSpec:
    name: str
    prompt: str
    color: tuple[int, int, int]  # BGR for OpenCV
    point: Optional[tuple[float, float]] = None
    box: Optional[tuple[float, float, float, float]] = None  # xywh pixels
    sam_obj_id: Optional[int] = None
    seed_mask: Any = None


def _runtime_imports():
    """Import heavy dependencies with an actionable error message."""
    missing: list[str] = []
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None  # type: ignore[assignment]
        missing.append("opencv-python")
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore[assignment]
        missing.append("numpy")
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore[assignment]
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "The active environment is missing: " + ", ".join(missing) + ". "
            "Install requirements.txt (and a CUDA-enabled PyTorch build), "
            "then run the command again."
        )
    return cv2, np, torch


def _to_numpy(value: Any, np: Any) -> Optional[Any]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _first(outputs: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in outputs:
            return outputs[name]
    return None


def _mask_candidates(outputs: Any, np: Any) -> list[dict[str, Any]]:
    """Normalize SAM3 output variants into a list of mask candidates."""
    if outputs is None:
        return []
    if not isinstance(outputs, dict):
        # Some wrappers return the output dictionary as a one-element tuple.
        if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
            outputs = outputs[0]
        if not isinstance(outputs, dict):
            return []

    raw_masks = _first(
        outputs,
        ("out_binary_masks", "masks", "mask", "pred_masks", "out_masks"),
    )
    masks = _to_numpy(raw_masks, np)
    if masks is None:
        return []
    masks = np.asarray(masks)
    # SAM3 normally returns [N,H,W]. Be tolerant of [N,1,H,W] and [H,W].
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        return []
    if masks.dtype != np.bool_:
        # Binary masks are bool; logits/probabilities are accepted too.
        masks = masks > (0.5 if np.nanmin(masks) >= 0 else 0.0)

    raw_ids = _first(outputs, ("out_obj_ids", "obj_ids", "object_ids", "ids"))
    ids = _to_numpy(raw_ids, np)
    ids = np.asarray(ids).reshape(-1).tolist() if ids is not None else []
    raw_scores = _first(outputs, ("out_probs", "scores", "out_scores", "probabilities"))
    scores = _to_numpy(raw_scores, np)
    scores = np.asarray(scores).reshape(-1).tolist() if scores is not None else []
    raw_boxes = _first(outputs, ("out_boxes_xywh", "boxes", "pred_boxes"))
    boxes = _to_numpy(raw_boxes, np)
    if boxes is not None:
        boxes = np.asarray(boxes).reshape((-1, 4)) if np.asarray(boxes).size else []

    result: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        if index < len(ids):
            try:
                object_id: Optional[int] = int(ids[index])
            except (TypeError, ValueError):
                object_id = None
        else:
            object_id = None
        score = float(scores[index]) if index < len(scores) else 1.0
        if boxes is not None and len(boxes) > index:
            box = np.asarray(boxes[index], dtype=float).tolist()
        else:
            box = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        result.append(
            {
                "index": index,
                "obj_id": object_id,
                "score": score,
                "mask": mask,
                "center": (float(xs.mean()), float(ys.mean())),
                "box": box,
            }
        )
    return result


def _parse_xy(value: Optional[str]) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected x,y, got {value!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected numeric x,y, got {value!r}") from exc


def _parse_xywh(value: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Expected x,y,w,h, got {value!r}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected numeric x,y,w,h, got {value!r}") from exc


def _load_video(path: Path, cv2: Any, np: Any, max_frames: Optional[int] = None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames: list[Any] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise RuntimeError("The video must contain at least two readable frames")
    return frames, fps


def _build_sam3(args: argparse.Namespace):
    import torch  # type: ignore

    if args.device == "cpu":
        # Disable CUDA autocast *before importing* SAM3. Several SAM3 methods
        # are decorated with ``@torch.autocast(device_type="cuda")``; on a
        # CUDA-less host those decorators can still leave CPU bf16 activations.
        from contextlib import ContextDecorator

        class NoAutocast(ContextDecorator):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        # ``torch.autocast`` is used by SAM3's decorators and context managers;
        # replacing it before importing SAM3 prevents those decorators from
        # capturing a CUDA/bfloat16 mode on CPU.
        torch.autocast = lambda *a, **kw: NoAutocast()  # type: ignore[assignment]
        try:
            torch.amp.autocast = lambda *a, **kw: NoAutocast()  # type: ignore[assignment]
        except Exception:
            pass
        # `@torch.autocast(...)` uses the decorator implementation captured by
        # torch._dynamo at import time in some PyTorch versions. Disable the
        # CPU bf16 mode directly as a second safeguard.
        try:
            torch.set_autocast_enabled("cpu", False)
            torch.set_autocast_dtype("cpu", torch.float32)
        except Exception:
            pass
        try:
            # scikit-image is optional in SAM3 but is used by its CPU connected
            # components fallback. scipy is already present in mh3/track.
            import scipy.ndimage as ndi  # type: ignore
            import sam3.perflib.connected_components as cc  # type: ignore

            def scipy_connected_components_cpu_single(values):
                values_np = values.detach().cpu().numpy().astype("uint8", copy=False)
                labels_np, num = ndi.label(values_np)
                labels = torch.from_numpy(labels_np.astype("int64", copy=False))
                counts = torch.zeros_like(labels)
                if num:
                    sizes = torch.bincount(labels.reshape(-1), minlength=int(num) + 1)
                    counts = sizes[labels]
                return labels, counts

            cc.connected_components_cpu_single = scipy_connected_components_cpu_single
        except Exception:
            pass

    try:
        from sam3.model_builder import (  # type: ignore
            build_sam3_video_model,
            build_sam3_video_predictor,
        )
        from sam3.model.sam3_base_predictor import Sam3BasePredictor  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 is not installed. Clone https://github.com/facebookresearch/sam3, "
            "install it in the active environment, and request/access its checkpoint."
        ) from exc
    if args.device == "cpu":
        # SAM3's released CPU-incompatible CUDA decorators can still cast
        # activations to bf16 on some PyTorch builds. Add a lightweight
        # float32 guard around Linear layers so a CPU run remains executable.
        import torch.nn as nn  # type: ignore

        def _module_float32(module, inputs):
            target_dtype = getattr(getattr(module, "weight", None), "dtype", None)
            if target_dtype is None or not inputs:
                return inputs
            if hasattr(inputs[0], "dtype") and inputs[0].dtype != target_dtype:
                return (inputs[0].to(target_dtype),) + tuple(inputs[1:])
            return inputs

        linear_hook = _module_float32
    else:
        linear_hook = None
    kwargs: dict[str, Any] = {}
    if args.sam3_checkpoint:
        kwargs["checkpoint_path"] = str(args.sam3_checkpoint)
    if args.sam3_bpe:
        kwargs["bpe_path"] = str(args.sam3_bpe)
    # The official predictor assumes CUDA and calls ``.cuda()`` internally.
    # Keep a small CPU wrapper for smoke tests and CPU-only hosts; production
    # measurements should use ``--device cuda`` on a CUDA machine.
    if args.device == "cpu":
        class CpuSam3VideoPredictor(Sam3BasePredictor):
            def __init__(self):
                super().__init__()
                # The released SAM3 code has two constructor-time positional
                # caches hard-coded to ``device="cuda"`` and a few tracker
                # paths call Tensor.cuda() even when state is CPU-offloaded.
                # Redirect those calls only for this CPU smoke-test process.
                real_zeros = torch.zeros
                real_arange = torch.arange
                factory_names = ("full", "empty", "ones", "tensor", "as_tensor", "rand", "randn")
                real_factories = {name: getattr(torch, name) for name in factory_names}

                def cpu_device(kwargs: dict[str, Any]) -> dict[str, Any]:
                    out = dict(kwargs)
                    if str(out.get("device", "")) == "cuda":
                        out["device"] = "cpu"
                    return out

                torch.zeros = lambda *a, **kw: real_zeros(*a, **cpu_device(kw))  # type: ignore[assignment]
                torch.arange = lambda *a, **kw: real_arange(*a, **cpu_device(kw))  # type: ignore[assignment]
                for factory_name, factory in real_factories.items():
                    setattr(
                        torch,
                        factory_name,
                        lambda *a, _factory=factory, **kw: _factory(*a, **cpu_device(kw)),
                    )
                if not getattr(torch.Tensor, "_p7_cpu_cuda_patch", False):
                    original_cuda = torch.Tensor.cuda
                    original_pin_memory = torch.Tensor.pin_memory
                    original_to = torch.Tensor.to

                    def cpu_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
                        return self

                    cpu_cuda._p7_original = original_cuda  # type: ignore[attr-defined]
                    torch.Tensor.cuda = cpu_cuda  # type: ignore[assignment]
                    torch.Tensor.pin_memory = lambda self, device=None: self  # type: ignore[assignment]
                    def cpu_to(self, *args, **kwargs):
                        args = list(args)
                        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
                            args[0] = torch.device("cpu")
                        if str(kwargs.get("device", "")).startswith("cuda"):
                            kwargs["device"] = torch.device("cpu")
                        return original_to(self, *args, **kwargs)
                    torch.Tensor.to = cpu_to  # type: ignore[assignment]
                    torch.Tensor._p7_cpu_cuda_patch = True  # type: ignore[attr-defined]
                model_kwargs = dict(kwargs)
                model_kwargs["device"] = "cpu"
                model_kwargs["load_from_HF"] = False
                self.model = build_sam3_video_model(**model_kwargs).eval()
                # A few SAM3 forward paths were decorated at import time. CPU
                # execution must be float32 end-to-end.
                self.model.float()
                if linear_hook is not None:
                    for module in self.model.modules():
                        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            module.register_forward_pre_hook(linear_hook)
                tracker = getattr(self.model, "tracker", None)
                bf16_context = getattr(tracker, "bf16_context", None)
                if bf16_context is not None:
                    # SAM3's tracker enters a CUDA bf16 context at
                    # construction time; on CPU this silently becomes a CPU
                    # bf16 autocast and mismatches the float32 checkpoint.
                    bf16_context.__exit__(None, None, None)
                    tracker.bf16_context = None
                self.async_loading_frames = False
                self.video_loader_type = "cv2"

            def start_session(self, *args, **kwargs):
                response = super().start_session(*args, **kwargs)
                # SAM3's video loader stores normalized frames as float16 for
                # GPU inference. CPU linear layers require matching float32
                # weights and activations.
                state = self._all_inference_states[response["session_id"]]["state"]
                image_batch = getattr(state.get("input_batch"), "img_batch", None)
                if image_batch is not None and image_batch.dtype != torch.float32:
                    state["input_batch"].img_batch = image_batch.float()
                return response

        try:
            return CpuSam3VideoPredictor()
        except Exception as exc:
            raise RuntimeError(
                "SAM3 CPU loading/inference failed. The checkpoint may be too "
                "large for this host; use a CUDA node with --device cuda. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but PyTorch cannot see a CUDA driver. "
            "Run this on a GPU node or use --device cpu for a very slow smoke test."
        )
    try:
        return build_sam3_video_predictor(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "SAM3 could not be loaded. Check the checkpoint/Hugging Face access and "
            "that this process has a CUDA GPU."
        ) from exc


def _prompt_request(
    session_id: str,
    spec: ObjectSpec,
    frame_index: int = 0,
    frame_shape: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    # SAM3's point prompt is a tracker prompt and cannot be combined with a
    # semantic text prompt. If a positive point is supplied, use it alone.
    request: dict[str, Any] = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_index,
        "text": None if spec.point is not None else spec.prompt,
        # Explicit IDs make matching masks during propagation deterministic.
        "obj_id": spec.sam_obj_id,
        "output_prob_thresh": 0.5,
    }
    if spec.point is not None:
        request["points"] = [list(spec.point)]
        request["point_labels"] = [1]
        request["rel_coordinates"] = False
    if spec.box is not None:
        if frame_shape is None:
            raise ValueError("frame_shape is required when using a box prompt")
        height, width = frame_shape
        x, y, w, h = spec.box
        # SAM3's semantic box interface expects normalized xywh, while this
        # evaluator's CLI intentionally accepts pixels for user convenience.
        request["bounding_boxes"] = [[x / width, y / height, w / width, h / height]]
        request["bounding_box_labels"] = [1]
        request["rel_coordinates"] = False
    return request


def _select_candidate(
    candidates: list[dict[str, Any]],
    spec: ObjectSpec,
    previous_center: Optional[tuple[float, float]] = None,
) -> Optional[dict[str, Any]]:
    if not candidates:
        return None
    if spec.sam_obj_id is not None:
        exact = [c for c in candidates if c["obj_id"] == spec.sam_obj_id]
        if exact:
            return max(exact, key=lambda c: c["score"])
    target = spec.point
    if target is None and spec.box is not None:
        x, y, w, h = spec.box
        target = (x + w / 2.0, y + h / 2.0)
    if previous_center is not None:
        target = previous_center
    if target is not None:
        return min(
            candidates,
            key=lambda c: (c["center"][0] - target[0]) ** 2 + (c["center"][1] - target[1]) ** 2,
        )
    return max(candidates, key=lambda c: c["score"] * math.sqrt(float(c["mask"].sum())))


def _run_sam3(
    video_path: Path,
    frame_count: int,
    frame_shape: tuple[int, int],
    specs: list[ObjectSpec],
    args: argparse.Namespace,
    np: Any,
):
    predictor = _build_sam3(args)
    masks: dict[str, list[Optional[Any]]] = {
        spec.name: [None] * frame_count for spec in specs
    }
    try:
        # A semantic text prompt resets the SAM3 video state. Use one session
        # per object so the sphere prompt cannot erase the ring prompt.
        for spec in specs:
            start = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": bool(args.offload_sam_state),
                }
            )
            session_id = start["session_id"]
            response = predictor.handle_request(
                _prompt_request(session_id, spec, frame_shape=frame_shape)
            )
            candidates = _mask_candidates(response.get("outputs"), np)
            selected = _select_candidate(candidates, spec)
            if selected is None:
                raise RuntimeError(
                    f"SAM3 found no mask for {spec.name!r}. Try --{spec.name}-point "
                    "or --{spec.name}-box on frame zero."
                )
            # The requested ID is normally echoed by SAM3. Preserve the actual
            # returned ID too, because older checkpoints may ignore obj_id.
            spec.sam_obj_id = selected["obj_id"] if selected["obj_id"] is not None else spec.sam_obj_id
            spec.seed_mask = selected["mask"]
            masks[spec.name][0] = selected["mask"]

            if args.sam3_propagate:
                stream_request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": frame_count,
                    "output_prob_thresh": 0.5,
                }
                previous = _mask_center(spec.seed_mask, np)
                for item in predictor.handle_stream_request(stream_request):
                    frame_index = int(item["frame_index"])
                    if not 0 <= frame_index < frame_count:
                        continue
                    chosen = _select_candidate(
                        _mask_candidates(item.get("outputs"), np), spec, previous
                    )
                    if chosen is not None:
                        masks[spec.name][frame_index] = chosen["mask"]
                        previous = chosen["center"]
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
    finally:
        shutdown = getattr(predictor, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
    return masks


def _mask_center(mask: Any, np: Any) -> Optional[tuple[float, float]]:
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _sample_mask_points(mask: Any, count: int, np: Any, cv2: Any) -> Any:
    """Sample deterministic interior points for CoTracker queries."""
    if mask is None:
        return np.empty((0, 2), dtype=np.float32)
    work = np.asarray(mask, dtype=np.uint8)
    # Erosion removes unstable edge pixels. A thin ring can disappear, so
    # fall back to the original mask if erosion leaves too few pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(work, kernel, iterations=1)
    ys, xs = np.where(eroded > 0)
    if len(xs) < max(4, count // 4):
        ys, xs = np.where(work > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    # Evenly spaced deterministic selection avoids random/non-reproducible runs.
    order = np.linspace(0, len(xs) - 1, min(count, len(xs)), dtype=int)
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)


def _run_cotracker(frames: list[Any], specs: list[ObjectSpec], masks: dict[str, list[Optional[Any]]], args: argparse.Namespace, np: Any, torch: Any, cv2: Any):
    try:
        if args.cotracker_checkpoint:
            from cotracker.predictor import CoTrackerPredictor  # type: ignore

            model = CoTrackerPredictor(
                checkpoint=str(args.cotracker_checkpoint), offline=True, window_len=60
            )
        else:
            model = torch.hub.load("facebookresearch/co-tracker", args.cotracker_model)
    except Exception as exc:
        raise RuntimeError(
            "CoTracker3 could not be loaded. Pass --cotracker-checkpoint to a "
            "local scaled_offline.pth, or allow the first-run checkpoint download."
        ) from exc
    device = torch.device(args.device)
    model = model.to(device).eval()

    query_points: list[list[float]] = []
    slices: dict[str, slice] = {}
    for spec in specs:
        points = _sample_mask_points(spec.seed_mask, args.points_per_object, np, cv2)
        begin = len(query_points)
        query_points.extend([[0.0, float(x), float(y)] for x, y in points])
        slices[spec.name] = slice(begin, len(query_points))
    if not query_points:
        raise RuntimeError("No points could be sampled from the SAM3 seed masks")

    # CoTracker3 expects H and W divisible by its stride (4). Edge padding does
    # not alter any original pixel coordinate or the measured trajectory.
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    height, width = rgb.shape[1:3]
    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.tensor(query_points, dtype=torch.float32, device=device)[None]
    with torch.inference_mode():
        try:
            tracks, visibility = model(video, queries=queries, grid_size=0)
        except TypeError:
            tracks, visibility = model(video, queries=queries)
    tracks = tracks.detach().float().cpu().numpy()[0]
    visibility = visibility.detach().float().cpu().numpy()
    if visibility.ndim == 4:
        visibility = visibility[..., 0]
    visibility = visibility[0] > float(args.visibility_threshold)

    centres: dict[str, Any] = {}
    visible_counts: dict[str, Any] = {}
    for spec in specs:
        track_slice = tracks[:, slices[spec.name], :]
        vis_slice = visibility[:, slices[spec.name]]
        centre = np.full((len(frames), 2), np.nan, dtype=np.float32)
        count = np.zeros(len(frames), dtype=np.int32)
        for frame_index in range(len(frames)):
            valid = vis_slice[frame_index] & np.isfinite(track_slice[frame_index]).all(axis=1)
            points = track_slice[frame_index][valid]
            # If a propagated SAM mask exists, reject points that have clearly
            # jumped to the other object/background. A 5px dilation handles
            # small segmentation/tracking discrepancies.
            mask = masks[spec.name][frame_index]
            if mask is not None and len(points):
                dilated = cv2.dilate(np.asarray(mask, dtype=np.uint8), np.ones((11, 11), np.uint8), 1)
                inside = []
                h, w = dilated.shape[:2]
                for point in points:
                    x, y = int(round(float(point[0]))), int(round(float(point[1])))
                    inside.append(0 <= x < w and 0 <= y < h and bool(dilated[y, x]))
                gated = points[np.asarray(inside, dtype=bool)]
                if len(gated) >= 2:
                    points = gated
            count[frame_index] = len(points)
            if len(points) >= 1:
                centre[frame_index] = np.median(points, axis=0)
        centres[spec.name] = centre
        visible_counts[spec.name] = count
    return centres, visible_counts, tracks, visibility, query_points, slices


def _mask_centres(masks: dict[str, list[Optional[Any]]], np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, sequence in masks.items():
        arr = np.full((len(sequence), 2), np.nan, dtype=np.float32)
        for i, mask in enumerate(sequence):
            center = _mask_center(mask, np)
            if center is not None:
                arr[i] = center
        result[name] = arr
    return result


def _summarize_masks(masks: dict[str, list[Optional[Any]]], np: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, sequence in masks.items():
        areas: list[int] = []
        boxes: list[Optional[list[int]]] = []
        for mask in sequence:
            if mask is None:
                areas.append(0)
                boxes.append(None)
                continue
            ys, xs = np.where(np.asarray(mask, dtype=bool))
            areas.append(int(len(xs)))
            boxes.append(
                [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if len(xs)
                else None
            )
        valid = [i for i, area in enumerate(areas) if area > 0]
        summary[name] = {
            "frames_with_mask": len(valid),
            "first_frame_area_px": areas[0] if areas else 0,
            "last_valid_frame": valid[-1] if valid else None,
            "min_area_px": min((areas[i] for i in valid), default=0),
            "max_area_px": max((areas[i] for i in valid), default=0),
            "areas_px": areas,
            "boxes_xyxy": boxes,
        }
    return summary


def _summarize_cotracker(
    specs: list[ObjectSpec],
    tracks: Any,
    visibility: Any,
    query_points: list[list[float]],
    slices: Optional[dict[str, slice]],
    np: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "track_array_shape": list(tracks.shape),
        "visibility_array_shape": list(visibility.shape),
        "query_count": len(query_points),
        "query_points_txy": query_points,
    }
    # The caller passes the per-object slices through a compact reconstruction:
    # points are concatenated in the same order as specs.
    offset = 0
    for spec in specs:
        count = int(round(len(query_points) / len(specs))) if len(specs) else 0
        # This is only a diagnostic summary; exact per-object counts are also
        # present in tracks.csv as visible-point counts.
        if slices is not None and spec.name in slices:
            sl = slices[spec.name]
            count = max(0, sl.stop - sl.start)
        chunk = tracks[:, offset : offset + count]
        chunk_vis = visibility[:, offset : offset + count]
        offset += count
        valid = np.isfinite(chunk).all(axis=-1) & (chunk_vis > 0)
        summary[spec.name] = {
            "query_points": count,
            "visible_samples": int(valid.sum()),
            "visible_fraction": float(valid.mean()) if valid.size else 0.0,
            "first_frame_median_xy": np.median(chunk[0], axis=0).tolist() if count else None,
            "last_frame_median_xy": np.median(chunk[-1], axis=0).tolist() if count else None,
        }
    return summary


def _fuse_centres(track_centres: dict[str, Any], mask_centres: dict[str, Any], visible_counts: dict[str, Any], np: Any) -> dict[str, Any]:
    fused: dict[str, Any] = {}
    for name, tracks in track_centres.items():
        out = np.array(tracks, copy=True)
        fallback = mask_centres[name]
        for i in range(len(out)):
            if not np.isfinite(out[i]).all() or visible_counts[name][i] < 2:
                out[i] = fallback[i]
        fused[name] = out
    return fused


def _valid_rows(points: Any, np: Any) -> Any:
    return np.isfinite(points).all(axis=1)


def _estimate_endpoints(centres: dict[str, Any], top: Optional[tuple[float, float]], finish: Optional[tuple[float, float]], np: Any):
    if (top is None) != (finish is None):
        raise ValueError("Pass both --top-point and --finish-point, or pass neither")
    if top is not None and finish is not None:
        return np.asarray(top, dtype=float), np.asarray(finish, dtype=float)
    starts: list[Any] = []
    ends: list[Any] = []
    for points in centres.values():
        valid = points[_valid_rows(points, np)]
        if len(valid) < 2:
            continue
        k = max(1, min(5, len(valid) // 10))
        starts.append(np.median(valid[:k], axis=0))
        ends.append(np.median(valid[-k:], axis=0))
    if not starts or not ends:
        raise RuntimeError("Could not estimate incline endpoints from tracked centres")
    return np.median(np.stack(starts), axis=0), np.median(np.stack(ends), axis=0)


def _crossing_frame(points: Any, top: Any, finish: Any, start_frame: int, tolerance_px: float, np: Any):
    axis = finish - top
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        raise ValueError("Top and finish points are identical")
    unit = axis / length
    s = np.full(len(points), np.nan, dtype=float)
    valid = _valid_rows(points, np)
    s[valid] = np.sum((points[valid] - top) * unit, axis=1)
    threshold = max(0.0, length - float(tolerance_px))
    first = max(1, int(start_frame))
    for i in range(first, len(s)):
        if not np.isfinite(s[i]):
            continue
        previous = s[i - 1]
        if np.isfinite(previous) and previous < threshold <= s[i]:
            denominator = s[i] - previous
            fraction = (threshold - previous) / denominator if abs(denominator) > 1e-9 else 0.0
            return (i - 1) + float(np.clip(fraction, 0.0, 1.0)), s, "crossed_finish"
    valid_after = np.where(np.isfinite(s) & (np.arange(len(s)) >= first))[0]
    if len(valid_after):
        return float(valid_after[-1]), s, "finish_not_reached; using_last_valid"
    return float("nan"), s, "no_valid_track"


def _detect_release(points: Any, threshold_px: float, np: Any) -> int:
    valid = _valid_rows(points, np)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0
    # The first visible centre is the physical top position. Using the median
    # of the first few frames would incorrectly move the release time forward
    # when the video starts with the bodies already rolling.
    first_valid = int(valid_indices[0])
    base = points[first_valid]
    displacement = np.linalg.norm(points - base, axis=1)
    for i in valid_indices[1:]:
        window = displacement[i : min(i + 3, len(displacement))]
        finite = window[np.isfinite(window)]
        if len(finite) and float(np.min(finite)) >= threshold_px:
            # The crossing lies between the previous frame and i. Reporting
            # the previous frame is a conservative frame-time estimate.
            return max(first_valid, int(i) - 1)
    return first_valid


def _write_overlay(path: Path, frames: list[Any], fps: float, specs: list[ObjectSpec], masks: dict[str, list[Optional[Any]]], centres: dict[str, Any], top: Any, finish: Any, results: dict[str, Any], cv2: Any, np: Any):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {path}")
    try:
        for frame_index, original in enumerate(frames):
            image = original.copy()
            for spec in specs:
                mask = masks[spec.name][frame_index]
                if mask is not None:
                    color = np.zeros_like(image)
                    color[:, :] = spec.color
                    alpha_mask = np.asarray(mask, dtype=bool)
                    image[alpha_mask] = (0.65 * image[alpha_mask] + 0.35 * color[alpha_mask]).astype(np.uint8)
                    contours, _ = cv2.findContours(alpha_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, spec.color, 2)
                point = centres[spec.name][frame_index]
                if np.isfinite(point).all():
                    cv2.circle(image, tuple(np.round(point).astype(int)), 5, spec.color, -1)
                valid = centres[spec.name][: frame_index + 1]
                valid = valid[np.isfinite(valid).all(axis=1)]
                if len(valid) > 1:
                    cv2.polylines(image, [np.round(valid).astype(np.int32)], False, spec.color, 2)
            cv2.circle(image, tuple(np.round(top).astype(int)), 7, (255, 255, 255), -1)
            cv2.circle(image, tuple(np.round(finish).astype(int)), 7, (0, 255, 255), -1)
            cv2.putText(image, f"frame={frame_index}  t={frame_index / fps:.3f}s", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y = 54
            for spec in specs:
                value = results["objects"][spec.name]["elapsed_s"]
                label = f"{spec.name}: {value:.4f}s" if value is not None and math.isfinite(value) else f"{spec.name}: n/a"
                cv2.putText(image, label, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spec.color, 2, cv2.LINE_AA)
                y += 25
            writer.write(image)
    finally:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    default_video = here.parent / "continuation.mp4"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=default_video)
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument("--sam3-bpe", type=Path, default=None)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=None)
    parser.add_argument("--offload-sam-state", action="store_true", help="Offload SAM3 state to CPU to reduce VRAM")
    parser.add_argument("--sam3-propagate", action="store_true", help="Propagate SAM3 masks through all frames (slow on CPU; CoTracker does not require this)")
    parser.add_argument("--device", default="cuda", help="CoTracker device, e.g. cuda or cpu")
    parser.add_argument("--cotracker-model", default="cotracker3_offline", choices=("cotracker3_offline",), help="Use the full-video CoTracker3 model")
    parser.add_argument("--points-per-object", type=int, default=32)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--sphere-prompt", default="solid sphere rolling on an inclined ramp")
    parser.add_argument("--ring-prompt", default="thin circular ring rolling on an inclined ramp")
    parser.add_argument("--sphere-point", type=_parse_xy, default=None, help="Optional frame-0 positive point: x,y")
    parser.add_argument("--ring-point", type=_parse_xy, default=None, help="Optional frame-0 positive point: x,y")
    parser.add_argument("--sphere-box", type=_parse_xywh, default=None, help="Optional frame-0 box: x,y,w,h")
    parser.add_argument("--ring-box", type=_parse_xywh, default=None, help="Optional frame-0 box: x,y,w,h")
    parser.add_argument("--top-point", type=_parse_xy, default=None, help="Centre position at ramp top: x,y")
    parser.add_argument("--finish-point", type=_parse_xy, default=None, help="Centre position at common finish: x,y")
    parser.add_argument("--finish-tolerance-px", type=float, default=4.0)
    parser.add_argument("--start-displacement-px", type=float, default=3.0)
    parser.add_argument("--release-frame", type=int, default=None, help="Override release frame; default detects first movement")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.points_per_object < 1:
        raise ValueError("--points-per-object must be positive")
    cv2, np, torch = _runtime_imports()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = _load_video(video_path, cv2, np, args.max_frames)
    specs = [
        ObjectSpec("solid_sphere", args.sphere_prompt, (0, 220, 255), args.sphere_point, args.sphere_box, 1),
        ObjectSpec("thin_ring", args.ring_prompt, (255, 100, 0), args.ring_point, args.ring_box, 2),
    ]
    print(f"Loaded {len(frames)} frames at {fps:.3f} FPS: {video_path}")
    print("Running SAM3 segmentation and propagation...")
    masks = _run_sam3(video_path, len(frames), frames[0].shape[:2], specs, args, np)
    mask_centres = _mask_centres(masks, np)
    print("Running CoTracker3 on points sampled from the two masks...")
    (
        track_centres,
        visible_counts,
        cotracker_tracks,
        cotracker_visibility,
        cotracker_queries,
        cotracker_slices,
    ) = _run_cotracker(frames, specs, masks, args, np, torch, cv2)
    centres = _fuse_centres(track_centres, mask_centres, visible_counts, np)

    sam3_summary = _summarize_masks(masks, np)
    cotracker_summary = _summarize_cotracker(
        specs,
        cotracker_tracks,
        cotracker_visibility,
        cotracker_queries,
        cotracker_slices,
        np,
    )
    with (args.output_dir / "sam3_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(sam3_summary, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        args.output_dir / "cotracker_tracks.npz",
        tracks=cotracker_tracks,
        visibility=cotracker_visibility.astype(np.uint8),
        queries=np.asarray(cotracker_queries, dtype=np.float32),
    )
    with (args.output_dir / "cotracker_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(cotracker_summary, handle, ensure_ascii=False, indent=2)
    print("SAM3 intermediate summary:")
    print(json.dumps(sam3_summary, ensure_ascii=False, indent=2))
    print("CoTracker3 intermediate summary:")
    print(json.dumps(cotracker_summary, ensure_ascii=False, indent=2))

    top, finish = _estimate_endpoints(centres, args.top_point, args.finish_point, np)
    path_length = float(np.linalg.norm(finish - top))
    if args.release_frame is not None:
        if not 0 <= args.release_frame < len(frames):
            raise ValueError("--release-frame is outside the video")
        release_frame = int(args.release_frame)
    else:
        release_candidates = [_detect_release(centres[s.name], args.start_displacement_px, np) for s in specs]
        # Both bodies are released together; the earliest reliable movement is
        # preferable when one mask is temporarily less visible.
        release_frame = min(release_candidates)

    positions: dict[str, Any] = {}
    objects: dict[str, Any] = {}
    for spec in specs:
        finish_frame, s, status = _crossing_frame(centres[spec.name], top, finish, release_frame, args.finish_tolerance_px, np)
        positions[spec.name] = s
        elapsed = (finish_frame - release_frame) / fps if math.isfinite(finish_frame) else float("nan")
        objects[spec.name] = {
            "release_frame": release_frame,
            "release_time_s": release_frame / fps,
            "finish_frame_float": finish_frame if math.isfinite(finish_frame) else None,
            "finish_time_s": finish_frame / fps if math.isfinite(finish_frame) else None,
            "elapsed_s": elapsed if math.isfinite(elapsed) else None,
            "status": status,
            "valid_frames": int(np.isfinite(centres[spec.name]).all(axis=1).sum()),
        }

    # Arrays are intentionally kept in simple CSV/NPZ artefacts; JSON contains
    # the human-readable summary and all geometry needed to reproduce it.
    result = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": len(frames),
        "top_point_xy": [float(x) for x in top],
        "finish_point_xy": [float(x) for x in finish],
        "path_length_px": path_length,
        "objects": objects,
        "method": (
            "SAM3 frame-0 visual prompts -> SAM3 propagated masks -> CoTracker3 median point tracks"
            if args.sam3_propagate
            else "SAM3 frame-0 visual prompts -> CoTracker3 median point tracks"
        ),
        "warning": (
            "Automatic endpoints are estimates; pass --top-point/--finish-point for benchmark timing."
            if args.top_point is None
            else "Endpoint centres were supplied explicitly in pixel coordinates."
        ),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        args.output_dir / "sam3_masks.npz",
        **{name: np.asarray([m if m is not None else np.zeros_like(specs[0].seed_mask, dtype=bool) for m in sequence], dtype=np.uint8) for name, sequence in masks.items()},
    )
    # Keep the path length outside the ndarray without changing the CSV API.
    csv_positions = {name: np.asarray(values[: len(frames)], dtype=float) for name, values in positions.items()}
    for name in csv_positions:
        csv_positions[name].setflags(write=False)
    with (args.output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["frame", "time_s"]
        for spec in specs:
            header.extend([f"{spec.name}_x", f"{spec.name}_y", f"{spec.name}_distance_px", f"{spec.name}_distance_norm", f"{spec.name}_visible_points"])
        writer.writerow(header)
        for i in range(len(frames)):
            row: list[Any] = [i, i / fps]
            for spec in specs:
                point = centres[spec.name][i]
                distance = csv_positions[spec.name][i]
                row.extend([
                    "" if not np.isfinite(point[0]) else float(point[0]),
                    "" if not np.isfinite(point[1]) else float(point[1]),
                    "" if not np.isfinite(distance) else float(distance),
                    "" if not np.isfinite(distance) else float(distance / path_length),
                    int(visible_counts[spec.name][i]),
                ])
            writer.writerow(row)
    if not args.no_overlay:
        _write_overlay(args.output_dir / "overlay.mp4", frames, fps, specs, masks, centres, top, finish, result, cv2, np)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
