#!/usr/bin/env python3
"""Transfer existing image annotations only after exact still hash and frame-0 checks.

Use the P34 virtualenv. The output is a candidate until the contact sheets have
been visually reviewed; --accept publishes reviewed input annotations. v2 still
enforces its own image correspondence threshold and rejects failing inputs.
No motion, physical measurements or scores are
copied from the original video.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import cv2
import numpy as np
from all_test_common import ROOT, V3, SOURCE, ANNOTATED, digest, write_json, now


def transform(annotation, width, height):
    a = deepcopy(annotation)
    sx, sy = width / a["size_wh"][0], height / a["size_wh"][1]
    point = lambda p: [round(p[0] * sx), round(p[1] * sy)]
    box = lambda b: point(b[:2]) + point(b[2:])
    for obj in a["objects"]:
        obj["box"] = box(obj["box"])
        for key in ("positive", "negative"):
            if key in obj:
                obj[key] = [point(p) for p in obj[key]]
    geom = a["geometry"]
    for key in ("fixed_reference_boxes", "stream_boxes"):
        if key in geom:
            geom[key] = [box(b) for b in geom[key]]
    for key in ("gap_evidence_box", "lamp_emission_box", "lamp_background_box"):
        if key in geom:
            geom[key] = box(geom[key])
    for key in ("pivot_points", "entry_plane", "exit_plane", "outlet_points", "ground_points"):
        if key in geom:
            geom[key] = [point(p) for p in geom[key]]
    for key in ("bottom_y", "liquid_surface_y"):
        if key in geom:
            geom[key] = round(geom[key] * sy)
    if "tank_walls_x" in geom:
        geom["tank_walls_x"] = [round(x * sx) for x in geom["tank_walls_x"]]
    # Old rollout observations are deliberately not propagated as input evidence.
    geom.pop("release_reference", None)
    a.pop("review", None)
    a["size_wh"] = [width, height]
    a["registration_scale_xy"] = [sx, sy]
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--review-subdir", default="")
    args = ap.parse_args()
    cv2.setNumThreads(2)
    models = sorted(p for p in (SOURCE / "data/videos/all_test").iterdir() if p.is_dir() and (not args.models or p.name in args.models) and all(os.access(v, os.R_OK) for v in (p/"gpt").glob("*.mp4")))
    records = []
    for task in sorted(ANNOTATED):
        original = next(V3.glob(f"g[1-9]/{task}/annotations/first_frame_annotations.json"))
        template = json.loads(original.read_text())
        still_path = next((V3).glob(f"g[1-9]/{task}/first_frames/provided/first_frame.png"))
        if digest(still_path) != template["source_image_sha256"]:
            raise ValueError(f"{task}: source still does not match the reviewed template")
        still = cv2.imread(str(still_path))
        # The existing template coordinates refer to its decoded video canvas,
        # while load_annotation resizes this same hashed still to that canvas.
        sheet = np.full((4 * 244, len(models) * 384, 3), 245, np.uint8)
        for col, model in enumerate(models):
            for row, video in enumerate(sorted((model / "gpt").glob(f"g*_{task}_seed*.mp4"))):
                cap = cv2.VideoCapture(str(video)); ok, frame = cap.read(); cap.release()
                if not ok:
                    raise ValueError(f"Cannot decode first frame: {video}")
                height, width = frame.shape[:2]
                a = transform(template, width, height)
                ref = cv2.resize(still, (width, height))
                delta = abs(cv2.GaussianBlur(ref, (5, 5), 0).astype(float) - cv2.GaussianBlur(frame, (5, 5), 0).astype(float))
                errors = []
                canvas = frame.copy()
                for i, obj in enumerate(a["objects"]):
                    x1, y1, x2, y2 = obj["box"]
                    errors.append(float(np.mean(delta[y1:y2, x1:x2])))
                    color = [(40, 210, 40), (255, 160, 20), (60, 80, 255)][i % 3]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    for pt in obj["positive"]:
                        cv2.circle(canvas, tuple(pt), 3, color, -1)
                accepted = max(errors) < 15.0
                a.update(source_video_sha256=digest(video), annotation_type="source_image_annotation_registered_to_frame0",
                         coordinate_frame="decoded_video_frame_0", source_video=str(video),
                         annotation_source=str(original), annotation_source_sha256=digest(original),
                         created_at_utc=now(), review="Exact annotated source-image SHA-256 matched; geometry scaled to decoded frame 0; correspondence checked per object. See contact sheet.",
                         registration_validation={"whole_frame_mean_absolute_difference": float(np.mean(delta)),
                                                  "object_crop_errors": errors, "v2_threshold": 15.0, "accepted": accepted},
                         note="Initialization geometry and identities only; no trajectory, velocity, peak, physical metric or score reused.")
                out = ROOT / ("v3_" + model.name) / "inputs"
                write_json(out / "annotation_candidates" / (video.stem + ".json"), a)
                cv2.imwrite(str(out / "annotation_candidates" / (video.stem + ".png")), canvas)
                if args.accept:
                    a['visual_review'] = {'reviewed_at_utc': now(), 'contact_sheet': str(ROOT / 'input_review' / args.review_subdir / (task + '.jpg')),
                                          'scope': 'Frame-zero object identities and scaled initialization boxes visually inspected across all model/seed panels; no motion inspected.'}
                    write_json(out / "annotations" / (video.stem + ".json"), a)
                tile = np.full((244, 384, 3), 245, np.uint8)
                cv2.putText(tile, model.name[:31], (4, 15), 0, .42, (0, 0, 0), 1)
                cv2.putText(tile, f"{video.stem} cropMAD={max(errors):.2f} {'OK' if accepted else 'REJECT'}", (4, 31), 0, .4, (0, 0, 0), 1)
                tile[36:244] = cv2.resize(canvas, (384, 208))
                sheet[row*244:(row+1)*244, col*384:(col+1)*384] = tile
                records.append({"model": model.name, "sample_id": video.stem, "task": task,
                                "max_crop_mad": max(errors), "whole_frame_mad": float(np.mean(delta)), "accepted": accepted,
                                "size_wh": [width, height], "candidate": str(out / "annotation_candidates" / (video.stem + ".json"))})
        review = ROOT / "input_review" / args.review_subdir
        review.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(review / (task + ".jpg")), sheet)
    write_json(ROOT / "input_review" / args.review_subdir / "annotation_validation.json", {"created_at_utc": now(), "published": args.accept, "samples": records})
    print(json.dumps({"total": len(records), "accepted": sum(r["accepted"] for r in records),
                      "rejected": [r for r in records if not r["accepted"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
