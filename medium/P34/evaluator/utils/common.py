"""Dependency-free result and threshold definitions."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import re
import tempfile


class ExtractionError(Exception):
    pass


class EnvironmentError(Exception):
    pass


def path_component(value, label):
    """Validate metadata keys used as a single output path component."""
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'{label} must be a filename-safe identifier: {value!r}')
    return value


@dataclass
class Thresholds:
    min_deflection_deg: float = 5.0
    symmetry_abs_tol_deg: float = 5.0
    symmetry_rel_tol: float = 0.20
    stable_window_sec: float = 0.5
    stable_angle_range_deg: float = 3.0
    initial_reference_tolerance_deg: float = 5.0
    min_valid_fraction: float = 0.95
    max_gap_sec: float = 0.10
    max_step_deg: float = 35.0
    max_speed_deg_sec: float = 360.0
    max_camera_rotation_deg: float = 0.5
    max_camera_shift_px: float = 3.0
    max_ellipse_residual: float = 0.035
    boundary_localization_px: float = 3.0
    dial_white_saturation_max: int = 40
    dial_white_value_min: int = 135
    max_pivot_offset_fraction: float = 0.15
    max_pivot_drift_fraction: float = 0.035
    max_needle_line_residual_px: float = 3.0
    min_red_pixels: int = 12
    detection_threshold: float = 0.25
    text_threshold: float = 0.20

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.min_valid_fraction > 1 or self.max_step_deg >= 180:
            raise ValueError('min_valid_fraction must be <= 1; max_step_deg must be < 180')


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a crashed sample leaving half a JSON document.
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(clean_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def new_result(video, image=None, prompt=None, model=None, seed=None, thresholds=None):
    keys = ('theta_1_pre_deg', 'theta_2_pre_deg', 'theta_1_post_deg',
            'theta_2_post_deg', 'delta_1_deg', 'delta_2_deg', 'opposite_direction',
            'absolute_deflection_sum_deg', 'applied_symmetry_tolerance_deg')
    return {
        'task_id': 'P34', 'video_path': str(video),
        'image_path': str(image) if image else None, 'video_prompt': prompt,
        'model': model, 'seed': seed,
        'metrics': {'M1': {'extract_success': False, 'metric': None},
                    'M2': {'extract_success': None, 'metric': None}},
        'verbose': {'M1': {
            'principle': '计算两根指南针相对于各自初始方向的有符号偏转，判断方向是否相反、幅度是否接近。',
            'status': 'extraction_failed', 'measurements': dict.fromkeys(keys),
            'thresholds': asdict(thresholds or Thresholds()), 'windows': {},
            'conditions': {}, 'evidence': [], 'reason': None}, 'M2': None}}
