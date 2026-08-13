"""Diagnostic overlay for one evaluated sample.

Judge-free numbers are only trustworthy if the geometry behind them can be checked by
eye, so every panel shows a quantity that enters a residual: where the reference level
and the two crossings landed, whether the apex vertex sits where the track turns over,
and whether the fit residuals are structured (a real model error) or scattered (noise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_overlay(path: str | Path, clip, seed, res, dbg, theta_deg: float) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track = dbg.get("primary")
    if track is None:
        return None
    flight = dbg.get("flight")
    fit = dbg.get("fit")
    shift = dbg.get("shift")
    xy = track.xy if shift is None else track.xy - shift
    n = np.arange(clip.n)

    # The reference is an ideal projectile with the prompt angle, anchored to the
    # measured launch/landing positions and reference level. Its horizontal range is
    # therefore the observed R, while its expected height is
    # tan(theta_prompt) / 4 * R -- exactly the geometry tested by M1. It is a visual
    # target, not hidden ground-truth trajectory data.
    ref_t = ref_x = ref_y = None
    if flight is not None:
        ref_t = np.linspace(flight.t_launch, flight.t_land, 240)
        u = (ref_t - flight.t_launch) / max(flight.t_land - flight.t_launch, 1e-9)
        ref_x = flight.x_launch + u * (flight.x_land - flight.x_launch)
        ref_h = np.tan(np.radians(theta_deg)) / 4.0 * flight.R
        ref_y = flight.y_ref - 4.0 * ref_h * u * (1.0 - u)

    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    ax = axes[0, 0]
    mid = flight.n_apex if flight else clip.n // 2
    ax.imshow(clip.frames[mid])
    good = np.isfinite(track.xy[:, 0])
    ax.plot(track.xy[good, 0], track.xy[good, 1], ".-", color="#2ec4ff", ms=3, lw=0.9,
            label=f"track ({track.backend})")
    for other in (dbg.get("tracks") or {}).values():
        if other.backend == track.backend:
            continue
        g2 = np.isfinite(other.xy[:, 0])
        ax.plot(other.xy[g2, 0], other.xy[g2, 1], ".", color="#ffb000", ms=2, alpha=0.6,
                label=f"cross-check ({other.backend})")
    if flight:
        # Convert the compensated reference back to the raw image coordinates used by
        # imshow. The correction is tiny for locked-off clips, but this keeps the
        # overlay correct when CoTracker reports camera translation.
        if shift is not None:
            ref_x_img = ref_x + np.interp(ref_t, n, shift[:, 0])
            ref_y_img = ref_y + np.interp(ref_t, n, shift[:, 1])
        else:
            ref_x_img, ref_y_img = ref_x, ref_y
        ax.plot(ref_x_img, ref_y_img, color="#20a464", lw=2.0, ls="--",
                label=f"reference ({theta_deg:g}° prompt)")
        ax.axhline(flight.y_ref, color="#ff4d4d", lw=1.0, ls="--", label="reference level")
        ax.plot([flight.x_launch, flight.x_land], [flight.y_ref] * 2, "r|", ms=14, mew=2)
        ax.plot([flight.x_apex], [flight.y_apex], "r*", ms=13, label="apex")
        ax.plot([flight.x_apex] * 2, [flight.y_apex, flight.y_ref], "r-", lw=1.0)
        ax.set_title(f"{res.sample_id}   H={flight.H:.1f}px  R={flight.R:.1f}px  "
                     f"H/R={flight.H / flight.R:.4f}  (theory {np.tan(np.radians(theta_deg)) / 4:.4f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, clip.width); ax.set_ylim(clip.height, 0)

    ax = axes[0, 1]
    ax.plot(n[good], xy[good, 1], ".", color="#2ec4ff", ms=4, label="y (drift-compensated)")
    if flight is not None:
        ax.plot(ref_t, ref_y, color="#20a464", lw=2.0, ls="--",
                label=f"reference ({theta_deg:g}° prompt)")
    if flight and fit is not None:
        seg = np.linspace(flight.n_launch, flight.n_land, 200)
        inside = np.flatnonzero(good & (n >= flight.n_launch) & (n <= flight.n_land))
        py = np.polyfit(inside.astype(float), xy[inside, 1], 2)
        ax.plot(seg, np.polyval(py, seg), "-", color="#ff4d4d", lw=1.3, label="quadratic fit")
        ax.set_title(f"y vs frame   N_up={flight.n_up:.2f}  N_down={flight.n_down:.2f}  "
                     f"ratio={flight.n_up / flight.n_down:.4f} (theory 1)")
    if flight:
        ax.axhline(flight.y_ref, color="#888", ls="--", lw=0.9)
        for tt, lab in ((flight.t_launch, "launch"), (flight.t_apex, "apex"),
                        (flight.t_land, "land")):
            ax.axvline(tt, color="#7a5cff", lw=0.9, ls=":")
            ax.text(tt, ax.get_ylim()[0], f" {lab}", fontsize=7, color="#7a5cff",
                    rotation=90, va="bottom")
        if fit is None:
            ax.set_title(f"y vs frame   reference: {theta_deg:g}° prompt, measured R")
    ax.invert_yaxis(); ax.set_xlabel("frame"); ax.set_ylabel("y (px, down)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(n[good], xy[good, 0], ".", color="#2ec4ff", ms=4, label="x")
    if flight is not None:
        ax.plot(ref_t, ref_x, color="#20a464", lw=2.0, ls="--",
                label=f"reference ({theta_deg:g}° prompt)")
    if flight and fit is not None:
        inside = np.flatnonzero(good & (n >= flight.n_launch) & (n <= flight.n_land))
        px = np.polyfit(inside.astype(float), xy[inside, 0], 1)
        seg = np.linspace(flight.n_launch, flight.n_land, 200)
        ax.plot(seg, np.polyval(px, seg), "-", color="#ff4d4d", lw=1.3,
                label=f"linear fit, v_x={fit.vx:.2f} px/frame")
        ax.set_title(f"x vs frame   rms_x/extent={fit.rms_x_norm:.4f}  "
                     f"rms_y/extent={fit.rms_y_norm:.4f}")
    ax.set_xlabel("frame"); ax.set_ylabel("x (px)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"model: {res.model}    theta_prompt: {theta_deg:g} deg",
             f"measurable: {res.measurable}"]
    if flight is not None:
        lines.append(f"reference curve: ideal {theta_deg:g}° parabola at measured R")
    if res.gate_reasons:
        lines.append("gate failures: " + ", ".join(res.gate_reasons))
    if res.measured.get("theta_measured_deg") is not None:
        lines.append(f"theta measured at launch: {res.measured['theta_measured_deg']:.1f} deg"
                     f"    theta implied by H/R: {res.measured['theta_from_HR_deg']:.1f} deg")
    lines.append("")
    lines.append(f"{'invariant':<24}{'measured':>11}{'theory':>11}{'residual':>11}{'tol':>8}  verdict")
    for r in res.residuals:
        lines.append(f"{r.name:<24}{r.measured:>11.4f}{r.theory:>11.4f}"
                     f"{r.residual:>11.4f}{r.tol:>8.3f}  {'pass' if r.passed else 'FAIL'}"
                     f"{'' if r.primary else '  (diagnostic)'}")
    if res.violations:
        lines.append("")
        lines += [f"HARD VIOLATION  {v.name}: {v.detail}" for v in res.violations]
    lines.append("")
    q = res.qc
    cov = "  ".join(f"{b}={q.get(f'coverage_{b}', float('nan')):.3f}"
                    for b in q.get("backends", []))
    lines.append(f"coverage: {cov}"
                 f"    disagreement: {q.get('tracker_disagreement_radii', float('nan')):.3f} radii")
    lines.append(f"camera drift: {q.get('camera_drift_norm', float('nan')):.4f} of diagonal"
                 f" ({q.get('camera_shift_source', '?')})"
                 f"    radius CV: {q.get('radius_cv', float('nan')):.3f}")
    if res.measured:
        lines.append(f"flight frames: {res.measured.get('flight_frames')}"
                     f"    inner coverage: {res.measured.get('inner_coverage', float('nan')):.3f}"
                     f"    landing extrapolated: {res.measured.get('land_extrapolated')}")
    sr = res.scene_residual
    lines.append(f"scene residual e_i: {sr:.4f}" if sr is not None else "scene residual e_i: n/a")
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.5,
            va="top", ha="left", transform=ax.transAxes)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return str(path)
