"""P40 - two droplets in air merge into one.

Benchmark metrics, taken verbatim:
  M1  |r_f^3 / (r_1^3 + r_2^3) - 1|: liquid volume is conserved through the
      merge, and a sphere's volume goes as r^3, so the cube of the merged radius
      equals the sum of the cubes of the two initial radii.
  M2  the change in roundness across the merge, i.e. how close to spherical the
      bodies are before and after.

All radii are pixel lengths in one frame series, and the metric is a ratio of
cubes, so no calibration enters.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import viz
from ..context import Context
from ..schema import Result
from ..track import background, camera_drift
from ..video import Clip

MIN_AREA = 60
SETTLE_TAIL = 0.15      # fraction of the clip used for the merged droplet


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "|r_f^3/(r_1^3 + r_2^3) - 1|", principle=(
        "Coalescence conserves liquid volume and both states are near-spherical, "
        "so r_f^3 = r_1^3 + r_2^3. The metric is the absolute fractional "
        "departure from that, which is zero when no liquid is created or lost. "
        "Radii come from a circle fitted to each droplet outline."), tol=0.1)
    M2 = res.add("M2", "roundness change across the merge", principle=(
        "Roundness is 1 minus the RMS radial deviation of the outline from its "
        "fitted circle, over the radius; it is 1 for a perfect circle. The metric is "
        "the absolute difference between the merged droplet's roundness and the "
        "mean roundness of the two initial droplets, so it reports whether the "
        "merge left a near-spherical body as it should."), tol=0.1)

    bg = background(clip)
    blobs = [_blobs(clip[i], bg) for i in range(clip.n)]
    counts = np.array([len(b) for b in blobs])

    # Two droplets before, one after: the merge frame is where the count drops
    # to one and stays there.
    merged_from = _first_stable_single(counts)
    two_frames = [i for i in range(clip.n)
                  if len(blobs[i]) == 2 and (merged_from is None
                                             or i < merged_from)]
    res.scene = {
        "camera_drift_frac_diag": camera_drift(clip),
        "blob_counts_head": counts[:12].tolist(),
        "blob_counts_tail": counts[-12:].tolist(),
        "frames_with_two_droplets": len(two_frames),
        "merge_frame": None if merged_from is None else int(merged_from),
        "extractor": "temporal-median background difference, then connected "
                     "components; radius = sqrt(area/pi)",
    }
    common = [
        "detect droplets by fitting circles to the edge contours and keeping "
        "those whose edge points stay equidistant from the centre",
        "count connected components per frame: two before the merge, one after",
        "merge frame = the first frame from which the count stays at one",
    ]
    M1.steps = common + [
        "r_1, r_2 = fitted radii on the last frame that still shows two "
        "droplets",
        f"r_f = median fitted radius over the final "
        f"{int(SETTLE_TAIL * 100)}% of the clip",
        "M1 = |r_f^3/(r_1^3 + r_2^3) - 1|",
    ]
    M2.steps = common + [
        "roundness = 1 - (RMS radial deviation of the outline from the "
        "fitted circle) / radius, which is 1 for a perfect circle",
        "M2 = |roundness_merged - mean(roundness_1, roundness_2)|",
    ]

    if not two_frames:
        res.fail_all("no frame shows two separate droplets, so the initial "
                     "volumes cannot be measured")
        return _finish(res, ctx, clip, blobs, None, None, None)
    if merged_from is None:
        res.fail_all("the droplets never settle into a single body, so there is "
                     "no merged droplet to measure")
        return _finish(res, ctx, clip, blobs, two_frames[-1], None, None)

    pre_i = two_frames[-1]
    pre = sorted(blobs[pre_i], key=lambda b: -b["area"])[:2]
    tail_start = max(merged_from, int((1 - SETTLE_TAIL) * clip.n))
    tail = [blobs[i][0] for i in range(tail_start, clip.n) if len(blobs[i]) == 1]
    if not tail:
        res.fail_all("no single-droplet frame in the final part of the clip")
        return _finish(res, ctx, clip, blobs, pre_i, merged_from, None)

    r1, r2 = pre[0]["radius"], pre[1]["radius"]
    rf = float(np.median([b["radius"] for b in tail]))
    denom = r1 ** 3 + r2 ** 3
    if denom <= 0:
        M1.fail("degenerate initial droplet radii", r1_px=r1, r2_px=r2)
    else:
        M1.succeed(abs(rf ** 3 / denom - 1.0), r1_px=r1, r2_px=r2,
                   r_merged_px=rf, frame_before_merge=int(pre_i),
                   merge_frame=int(merged_from),
                   merged_frames_used=len(tail),
                   volume_ratio=rf ** 3 / denom,
                   r_merged_px_series=[round(b["radius"], 2) for b in tail])

    round_pre = float(np.mean([b["roundness"] for b in pre]))
    round_post = float(np.median([b["roundness"] for b in tail]))
    M2.succeed(abs(round_post - round_pre),
               roundness_before=round_pre, roundness_after=round_post,
               roundness_droplet_1=pre[0]["roundness"],
               roundness_droplet_2=pre[1]["roundness"])
    return _finish(res, ctx, clip, blobs, pre_i, merged_from, tail)


MIN_RADIUS = 16.0
MAX_RADIUS = 300.0
# Loose enough that a merged droplet still wobbling out of round is found,
# since reporting that deformation is exactly what M2 is for. Clutter is
# excluded by the radius range, the closed-outline requirement and the
# in-frame requirement rather than by demanding a perfect circle.
MAX_RADIAL_RMS_FRAC = 0.28


def _blobs(frame: np.ndarray, bg: np.ndarray) -> list[dict]:
    """Droplets in one frame, found by fitting circles to their outlines.

    The outline is the one cue both routes share. A brightness split does not
    work: in the photographic route the droplets are almost as dark as the wall
    behind them and are betrayed only by a thin bright rim, while the rendered
    route has bright droplets on a dark ground. Both, however, present a clean
    circular edge, before and after the merge alike, so a circle fit to the edge
    contours reads every state of the clip with one rule.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = g.shape

    found: list[dict] = []
    for c in cnts:
        pts = c.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 40:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (MIN_RADIUS <= r <= MAX_RADIUS):
            continue
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        # A circular outline keeps every edge point at the same distance from
        # the centre; a wall crack or a highlight streak does not.
        rms = float(np.sqrt(np.mean((d - r) ** 2)))
        if rms / r > MAX_RADIAL_RMS_FRAC:
            continue
        # The benchmark keeps every droplet fully inside the frame.
        if cx - r <= 1 or cy - r <= 1 or cx + r >= w - 1 or cy + r >= h - 1:
            continue
        found.append({"radius": float(r), "cx": float(cx), "cy": float(cy),
                      "roundness": float(1.0 - rms / r),
                      "radial_rms_px": rms, "contour": c,
                      "area": float(np.pi * r * r)})

    # One droplet produces an inner and an outer rim contour; keep the roundest
    # detection per location.
    found.sort(key=lambda b: -b["roundness"])
    keep: list[dict] = []
    for b in found:
        if all(np.hypot(b["cx"] - k["cx"], b["cy"] - k["cy"])
               > 0.6 * max(b["radius"], k["radius"]) for k in keep):
            keep.append(b)
    return sorted(keep, key=lambda b: b["cx"])


def _first_stable_single(counts: np.ndarray, run: int = 6,
                         purity: float = 0.8) -> int | None:
    """First index from which the clip is a single droplet for the rest of it.

    A single stray frame - a transient satellite drop, one noisy threshold -
    should not hide a merge that plainly happened, so the test is that the
    remainder of the clip is overwhelmingly single-bodied rather than perfectly
    so, and that the clip ends single-bodied.
    """
    n = counts.size
    if n < run or counts[-1] != 1:
        return None
    for i in range(n - run):
        rest = counts[i:]
        if float(np.mean(rest == 1)) >= purity:
            return i
    return None


def _finish(res, ctx, clip, blobs, pre_i, merged_from, tail):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, blobs, pre_i, merged_from, tail, res)
    return res


def _debug(clip, ctx, blobs, pre_i, merged_from, tail, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)

    if pre_i is not None:
        viz.show_frame(ax[0], clip[pre_i], f"frame {pre_i}: before the merge")
        for b, col in zip(sorted(blobs[pre_i], key=lambda b: -b["area"])[:2],
                          ("#ff3b30", "#0a84ff")):
            c = b["contour"].reshape(-1, 2)
            ax[0].plot(c[:, 0], c[:, 1], "-", color=col, lw=1.6,
                       label=f"r = {b['radius']:.1f} px")
        ax[0].legend(loc="lower left", fontsize=7)
    else:
        viz.show_frame(ax[0], clip[0], "two droplets never seen")

    if tail:
        j = clip.n - 1
        viz.show_frame(ax[1], clip[j], f"frame {j}: merged droplet")
        c = tail[-1]["contour"].reshape(-1, 2)
        ax[1].plot(c[:, 0], c[:, 1], "-", color="#34c759", lw=1.8,
                   label=f"r_f = {tail[-1]['radius']:.1f} px")
        ax[1].legend(loc="lower left", fontsize=7)
    else:
        viz.show_frame(ax[1], clip[clip.n - 1], "no merged droplet measured")

    rad = [[b["radius"] for b in bl] for bl in blobs]
    for i, rs in enumerate(rad):
        for r in rs:
            ax[2].plot(i, r, ".", ms=2.5, color="#0a84ff")
    if merged_from is not None:
        ax[2].axvline(merged_from, color="#ffd400", ls="--",
                      label=f"merge at frame {merged_from}")
    q = res.metrics["M1"].quantities
    if q.get("r_merged_px"):
        target = (q["r1_px"] ** 3 + q["r2_px"] ** 3) ** (1 / 3)
        ax[2].axhline(target, color="#ff3b30", ls=":",
                      label=f"volume-conserving r_f = {target:.1f} px")
        ax[2].axhline(q["r_merged_px"], color="#34c759", ls="-", lw=1.0,
                      label=f"measured r_f = {q['r_merged_px']:.1f} px")
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("droplet radius  [px]")
    ax[2].set_title("radii over the clip")
    ax[2].legend(fontsize=7)

    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 |r_f^3/(r1^3+r2^3) - 1| = {fmt('M1')}"
                    f"   |   M2 roundness change = {fmt('M2')}")
