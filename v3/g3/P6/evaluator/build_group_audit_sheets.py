#!/usr/bin/env python3
"""Stack four all-frame seed sheets per first-frame family for manual QA."""

from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent / "results_v1"
SOURCE = ROOT / "dense_audit_sheets"
OUTPUT = ROOT / "group_audit_sheets"
OUTPUT.mkdir(parents=True, exist_ok=True)
groups = defaultdict(list)
for path in sorted(SOURCE.glob("P6_*.jpg")):
    group = path.stem.rsplit("_seed", 1)[0]
    groups[group].append(path)

for group, paths in sorted(groups.items()):
    blocks = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((1024, 529), Image.Resampling.LANCZOS)
        blocks.append((path.stem, image))
    sheet = Image.new("RGB", (1024, 545 * len(blocks)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(blocks):
        y = index * 545
        draw.text((4, y + 2), f"{name} | all 124 consecutive frames", fill="black")
        sheet.paste(image, (0, y + 16))
    sheet.save(OUTPUT / f"{group}.jpg", quality=76, optimize=True)
    print(group)
