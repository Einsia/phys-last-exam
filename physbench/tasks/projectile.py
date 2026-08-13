"""P2: ball launched at a stated angle. Judge-free measurables.

What is measured, and why each survives unknown calibration:

  M1  H / R  vs  tan(theta)/4
      H and R are both pixel lengths, so the unknown metres-per-pixel s cancels in
      the ratio. Neither depends on the frame rate. Requires H and R to be measured
      between points at equal height, which is why both ends are level crossings of
      one reference line.

  M2  N_up / N_down  vs  1
      Both are frame counts, so the unknown seconds-per-frame tau cancels. This is
      the purest invariant in the task: it needs no angle, no length, and no fit.

  M3  (x_apex - x_launch) / (x_land - x_apex)  vs  1
      Spatial mirror symmetry about the apex. Independent of M2 -- a clip can trace a
      perfect arc while traversing it at the wrong speed, or trace a lopsided arc at
      symmetric timing, and these two catch the two cases separately.

  M4  parabolicity: normalised RMS of (x linear in n, y quadratic in n)
      Constant horizontal velocity and constant vertical acceleration. Scale-free
      after normalising by each axis's own extent. Weak against smooth low-order
      error by construction -- see M5.

  M5  v_x(descent) / v_x(ascent)  vs  1
      Conservation of horizontal momentum, as a ratio of two px/frame quantities, so
      s and tau cancel. This is first-order where M4 is a residual, which matters more
      than it sounds: drag bends x(t) into a gentle parabola, and a parabola is close
      to its own best-fit line in RMS terms, so a clip that loses 21% of its horizontal
      speed registers only ~1.4% on M4 and passes. M5 reads the same clip as 18% off.

  M6  a_y(descent) / a_y(ascent)  vs  1
      Gravity is the same going up and coming down. M2 tests only the integrated
      version of this; the accelerations can differ while the two durations match.
      Skipped when both halves are near zero, where the ratio is two zeros and says
      nothing -- that case is M7's to catch.

  M7  a_y * N^2 / H  vs  8   (diagnostic only -- see below)
      Ties the fitted dynamics to the measured geometry: with H = v0^2/(2g) and
      N = 2 v0/g the combination is exactly 8 for any projectile, whatever g, v0, and
      the two calibrations are. a_y is px/frame^2, N is frames, H is px, so it is fully
      calibration-free.

      It is kept as a diagnostic and NOT allowed to gate, because measurement shows it
      is insensitive for a reason that does not go away. On a triangular
      (constant-speed-up-then-down) path the least-squares quadratic returns
      a_y = 15 v0 / 4N, giving a_y N^2 / H_true = 7.5 -- but the apex fit also rounds
      off the sharp peak and shrinks measured H by ~6%, and the two biases very nearly
      cancel, putting the observed value back at 7.97. Both errors are driven by the
      same non-parabolicity, so they move together rather than compounding. The tent is
      caught decisively by M1 (0.63) and M4 (0.07) instead.

M1 is computed twice. Against the *prompt's* angle it tests instruction-following
and shape together. Against the angle *measured* from the launch direction it tests
shape alone. The gap between the two is the instruction-following error, so
reporting both keeps them from being confused for each other.
"""

from __future__ import annotations

import numpy as np

from ..config import TaskConfig
from ..first_frame import BallSeed
from ..fits import fit_parabola, half_fits, launch_slope
from ..io_video import Clip
from ..kinematics import compensate, extract_flight
from ..metrics import SampleResult, Violation, make_abs_residual, make_residual
from ..tracking import estimate_camera_shift, run_tracks, track_disagreement


def _near_border(xy: np.ndarray, w: int, h: int, margin: float) -> bool:
    if not np.isfinite(xy).all():
        return False
    x, y = xy
    return bool(x < margin or y < margin or x > w - margin or y > h - margin)


def _max_gap(track) -> int:
    """Longest run of consecutive missing detections inside the tracked span.

    This is the observability signal for "the ball was not continuously in view", and it
    is more reliable than testing whether a sighting is near a border. A steeply launched
    ball crosses the border band in a single step -- y = 60 in one frame, y = -30 in the
    next -- so it is never observed inside the margin, yet its apex is off-canvas and H is
    unmeasurable. The gap is what gives that away. A ball genuinely in view the whole time
    has gaps of at most a frame or two from tracker noise.

    Deliberately agnostic about *why* the ball vanished: from pixels alone, "left the
    frame" and "stopped existing" are the same observation, and neither is measurable.
    That is what PMR is for.
    """
    found = track.found
    idx = np.flatnonzero(found)
    if len(idx) < 2:
        return len(found)
    inner = found[idx[0]:idx[-1] + 1]
    best = run = 0
    for ok in inner:
        run = 0 if ok else run + 1
        best = max(best, run)
    return best


def _still_moving_at_end(track, radius: float, k: int = 4) -> bool:
    """Is the ball still in motion at its last sighting?

    Separates a clip that simply ran out before the ball landed (truncated flight, an
    observability limit) from one where the ball stopped in mid-air (a physics failure).
    Both surface as `no_descent_observed`, and they must not be scored the same way.
    """
    idx = np.flatnonzero(track.found)
    if len(idx) < k + 1:
        return False
    tail = track.xy[idx[-(k + 1):]]
    step = float(np.median(np.linalg.norm(np.diff(tail, axis=0), axis=1)))
    return step > 0.15 * radius


def _never_moved(track, radius: float) -> bool:
    """Did the ball's centre stay within its own radius for the entire clip?

    Geometric, not tuned: a ball whose centre never displaced by even one radius did
    not launch, at any frame rate and any scale. Tested on displacement rather than on
    extract_flight's reason string because a static ball does not reliably report
    `no_motion_observed` -- sub-pixel tracker noise keeps the peak speed above the
    exact-zero test, and the run then terminates as `no_ascent_observed` instead.
    """
    d = track.xy[track.found]
    if len(d) < 2:
        return False
    return float(np.linalg.norm(d.max(axis=0) - d.min(axis=0))) < radius


def _radius_cv(radius: np.ndarray, lo: int, hi: int) -> float:
    """Coefficient of variation of the apparent ball radius over the flight.

    Inflated by legitimate motion blur -- a fast ball smears into a streak whose
    equivalent-disc radius is larger -- so the gate on this is deliberately loose and
    the raw value is always reported rather than only its verdict.
    """
    seg = radius[lo:hi + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 5:
        return float("nan")
    m = float(np.mean(seg))
    return float(np.std(seg) / m) if m > 1e-6 else float("nan")


def evaluate_projectile(clip: Clip, seed: BallSeed, theta_deg: float,
                        sample_id: str, model: str,
                        cfg: TaskConfig | None = None) -> tuple[SampleResult, dict]:
    cfg = cfg or TaskConfig()
    res = SampleResult(sample_id=sample_id, task=cfg.task, model=model, measurable=False)
    dbg: dict = {}

    from ..neural_track import TrackerBackendConfig

    tcfg = cfg.tracking
    ts = run_tracks(clip.frames, seed, backends=tuple(tcfg.backends),
                    backend_cfg=TrackerBackendConfig(devices=tcfg.device,
                                                     dtype=tcfg.dtype,
                                                     sam3_src=tcfg.sam3_src,
                                                     sam3_checkpoint=tcfg.sam3_checkpoint,
                                                     sam3_prompt=tcfg.sam3_prompt))
    ta, tb = ts.pair
    primary = ts.primary
    # CoTracker's background grid measures translation directly from point
    # correspondences, with the ball's own neighbourhood excluded by construction.
    # Phase correlation has to be told to mask the ball and can still be pulled by it,
    # so the tracker's estimate is preferred when one exists.
    if tcfg.camera_shift_from_tracker and ts.camera_shift is not None:
        shift = ts.camera_shift
        shift_src = "cotracker_bg_grid"
    else:
        shift = estimate_camera_shift(clip.frames, primary, seed)
        shift_src = "phase_correlation"
    disagree = track_disagreement(ta, tb, seed.radius)
    drift = float(np.max(np.linalg.norm(shift, axis=1)) / clip.diag)
    gap = _max_gap(primary)

    res.qc = {
        "n_frames": clip.n, "width": clip.width, "height": clip.height,
        "backends": list(ts.tracks),
        **{f"coverage_{b}": t.coverage for b, t in ts.tracks.items()},
        "primary_backend": primary.backend,
        "tracker_disagreement_radii": disagree,
        "camera_drift_norm": drift,
        "camera_shift_source": shift_src,
        "max_detection_gap": gap,
        "container_fps": clip.fps,
        "tracker_seconds": ts.seconds,
    }
    dbg.update({"tracks": ts.tracks, "shift": shift, "primary": primary})

    # ---- observability gates. Physics correctness is not consulted here. ----
    gates: list[str] = []
    if primary.coverage < cfg.gates.min_coverage_overall:
        gates.append(f"low_coverage({primary.coverage:.2f})")
    if disagree > cfg.gates.max_tracker_disagree:
        gates.append(f"tracker_disagreement({disagree:.2f})")
    if drift > cfg.gates.max_camera_drift:
        gates.append(f"camera_drift({drift:.3f})")
    if gap > cfg.gates.max_detection_gap:
        gates.append(f"detection_gap({gap})")
    if gates:
        res.gate_reasons = gates
        return res, dbg

    xy = compensate(primary.xy, shift)
    flight, reason = extract_flight(xy, seed.radius, cfg.gates.min_flight_frames)

    if flight is None:
        # A clip with no rise-apex-fall is either unobservable or wrong, and merging the
        # two would let a model trade one for the other. Three reasons are ours, not the
        # video's -- too few detections to fit, a flight shorter than the sampling
        # requirement, or a ball that left the view -- and those lower PMR. Anything else
        # means the ball was tracked cleanly, stayed in frame, and still failed to rise
        # and come back down: that is measurable, and wrong.
        res.qc["flight_extract_reason"] = reason
        our_limit = reason == "too_few_detections" or reason.startswith("flight_too_short")
        # Still airborne at the last sighting: the clip ended before the ball landed, so
        # the flight was never contained in the view. Covers both ways a truncated clip
        # presents -- cut off before any descent, or cut off mid-descent so the launch
        # level is never reached. A ball that came to REST without landing is a different
        # thing entirely and falls through to the violation below.
        truncated = (reason in ("no_descent_observed", "no_ascent_observed")
                     or reason.startswith("landing_level_not_reached")) and \
            _still_moving_at_end(primary, seed.radius)
        if our_limit or truncated:
            res.gate_reasons = [f"flight_not_contained({reason})"]
            return res, dbg
        res.measurable = True
        # A ball tracked at full coverage that never moved at all is its own failure,
        # and naming it "no measurable flight" would read as a measurement problem when
        # it is the plainest possible instruction failure: the prompt says the ball
        # launches immediately and it never left the ground.
        #
        # Reaching this branch depends on the tracker. The classic pair calls a static
        # clip unmeasurable, but only because bgsub cannot see a stationary ball at all
        # -- it becomes its own temporal-median background, coverage collapses, and the
        # disagreement gate fires. That is the right verdict from a blind spot. The
        # learned pair tracks it perfectly, both backends agree, and the clip lands
        # here, where README's own rule puts it: tracked cleanly, stayed in frame, and
        # still wrong.
        if _never_moved(primary, seed.radius):
            d = primary.xy[primary.found]
            res.violations.append(Violation(
                "ball_never_moved",
                f"centre moved {float(np.linalg.norm(d.max(axis=0) - d.min(axis=0))):.2f} px "
                f"over {int(primary.found.sum())} tracked frames "
                f"(radius {seed.radius:.1f} px)"))
        else:
            res.violations.append(Violation("no_measurable_flight", reason))
        return res, dbg

    if flight.inner_coverage < cfg.gates.min_coverage_flight:
        res.gate_reasons = [f"low_flight_coverage({flight.inner_coverage:.2f})"]
        return res, dbg

    rcv = _radius_cv(primary.radius, flight.n_launch, flight.n_land)
    res.qc["radius_cv"] = rcv
    if np.isfinite(rcv) and rcv > cfg.gates.max_radius_cv:
        res.gate_reasons = [f"ball_shape_unstable({rcv:.2f})"]
        return res, dbg

    res.measurable = True
    dbg["flight"] = flight
    _measure(res, dbg, xy, flight, seed, theta_deg, cfg)
    return res, dbg


def _measure(res: SampleResult, dbg: dict, xy: np.ndarray, flight, seed: BallSeed,
             theta_deg: float, cfg: TaskConfig) -> None:
    """Fill in residuals and violations for a flight that passed the gates."""
    found = np.isfinite(xy[:, 0])
    inside = np.flatnonzero(found & (np.arange(len(xy)) >= flight.n_launch)
                            & (np.arange(len(xy)) <= flight.n_land))
    t = inside.astype(float)
    x, y = xy[inside, 0], xy[inside, 1]

    fit = fit_parabola(t, x, y, flight.t_launch, radius=seed.radius)
    slope = launch_slope(t, x, y, flight.t_launch, flight.n_up)
    halves = half_fits(t, x, y, flight.t_apex)
    dbg["fit"] = fit
    dbg["halves"] = halves
    theta_nom = float(np.radians(theta_deg))
    hr = flight.H / flight.R
    tsym = flight.n_up / flight.n_down if flight.n_down > 1e-9 else float("inf")
    ssym = ((flight.x_apex - flight.x_launch) / (flight.x_land - flight.x_apex)
            if abs(flight.x_land - flight.x_apex) > 1e-9 else float("inf"))

    res.measured = {
        "H_px": flight.H, "R_px": flight.R, "H_over_R": hr,
        "n_up": flight.n_up, "n_down": flight.n_down,
        "n_up_int": flight.n_apex - flight.n_launch,
        "n_down_int": flight.n_land - flight.n_apex,
        "t_launch": flight.t_launch, "t_apex": flight.t_apex, "t_land": flight.t_land,
        "x_launch": flight.x_launch, "x_apex": flight.x_apex, "x_land": flight.x_land,
        "y_ref": flight.y_ref, "y_apex": flight.y_apex,
        "theta_nominal_deg": theta_deg,
        "flight_frames": int(flight.n_land - flight.n_launch),
        "inner_coverage": flight.inner_coverage,
        "used_rest_level": flight.used_rest_level,
        "land_extrapolated": flight.land_extrapolated,
        "theta_from_HR_deg": float(np.degrees(np.arctan(4.0 * hr))),
    }
    if fit is not None:
        res.measured.update({"fit_vx_px_per_frame": fit.vx, "fit_ay_px_per_frame2": fit.ay,
                             "fit_vy0_px_per_frame": fit.vy0,
                             "fit_rms_x_norm": fit.rms_x_norm,
                             "fit_rms_y_norm": fit.rms_y_norm})

    # M1 against the prompt's angle: instruction-following and shape together.
    res.residuals.append(make_residual(
        "M1_HR_nominal", hr, np.tan(theta_nom) / 4.0, cfg.tol.hr_nominal,
        primary=True, note="H/R vs tan(theta_prompt)/4"))

    # M2: the one invariant that needs no angle and no length.
    res.residuals.append(make_residual(
        "M2_time_symmetry", tsym, 1.0, cfg.tol.time_symmetry,
        primary=True, note="N_up/N_down vs 1"))

    # M1 against the measured launch direction: shape alone. Must use the *local*
    # launch fit -- against the global fit's own angle this identity is vacuous.
    if slope is not None:
        s, npts = slope
        res.measured.update({"launch_slope_local": s, "launch_slope_npts": npts,
                             "theta_measured_deg": float(np.degrees(np.arctan(abs(s))))})
        res.residuals.append(make_residual(
            "M1_HR_selfconsistent", hr, abs(s) / 4.0, cfg.tol.hr_selfconsistent,
            primary=False, note="H/R vs tan(theta_measured_local)/4"))

    res.residuals.append(make_residual(
        "M3_space_symmetry", ssym, 1.0, cfg.tol.space_symmetry,
        primary=False, note="apex-centred horizontal mirror symmetry"))

    if halves is not None:
        res.measured.update({
            "vx_ascent": halves.vx_ascent, "vx_descent": halves.vx_descent,
            "ay_ascent": halves.ay_ascent, "ay_descent": halves.ay_descent,
        })
        res.residuals.append(make_residual(
            "M5_vx_conservation", halves.vx_ratio, 1.0, cfg.tol.vx_conservation,
            primary=True, note="v_x(descent)/v_x(ascent) vs 1"))
        # An ideal clip has a_y ~ 8H/N^2 on each half. Below a small fraction of that
        # there is no acceleration to compare and the ratio is noise over noise.
        ay_scale = 8.0 * flight.H / max((flight.t_land - flight.t_launch) ** 2, 1e-9)
        if min(abs(halves.ay_ascent), abs(halves.ay_descent)) > 0.15 * ay_scale:
            res.residuals.append(make_residual(
                "M6_gravity_symmetry", halves.ay_ratio, 1.0, cfg.tol.gravity_symmetry,
                primary=False, note="a_y(descent)/a_y(ascent) vs 1"))
        else:
            res.measured["M6_skipped"] = "per-half a_y below 15% of 8H/N^2"
        if halves.vx_ascent * halves.vx_descent < 0:
            res.violations.append(Violation(
                "horizontal_velocity_reversal", "v_x sign differs between halves"))

    if fit is not None:
        n_flight = flight.t_land - flight.t_launch
        m7 = fit.ay * n_flight ** 2 / flight.H if flight.H > 1e-9 else float("nan")
        res.measured["ay_N2_over_H"] = m7
        res.residuals.append(make_residual(
            "M7_accel_geometry", m7, 8.0, cfg.tol.accel_geometry,
            primary=False, note="a_y*N^2/H vs 8; diagnostic, biases cancel (see docstring)"))

    if fit is not None:
        para = float(max(fit.rms_x_norm, fit.rms_y_norm))
        res.residuals.append(make_abs_residual(
            "M4_parabolicity", para, cfg.tol.parabolicity, primary=True,
            note="max normalised RMS of (x linear, y quadratic) in frame index"))
        if para > cfg.viol.parabolicity_gross:
            res.violations.append(Violation("not_a_parabola", f"norm RMS {para:.3f}"))
        if fit.ay <= 0:
            res.violations.append(Violation(
                "no_downward_acceleration", f"fitted a_y {fit.ay:.3f} px/frame^2"))

    if np.isfinite(tsym) and (tsym >= cfg.viol.time_symmetry_factor
                              or tsym <= 1.0 / cfg.viol.time_symmetry_factor):
        res.violations.append(Violation("time_asymmetry_gross", f"N_up/N_down = {tsym:.2f}"))
    theory_hr = np.tan(theta_nom) / 4.0
    if np.isfinite(hr) and theory_hr > 0 and (
            hr / theory_hr >= cfg.viol.hr_factor or hr / theory_hr <= 1.0 / cfg.viol.hr_factor):
        res.violations.append(Violation("hr_gross", f"H/R off by {hr / theory_hr:.2f}x"))
