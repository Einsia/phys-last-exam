#!/usr/bin/env python3
"""Create a small labelled contact sheet and list likely pendulum bobs."""
from pathlib import Path
import argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    tiles = []
    for path in sorted(Path(args.inputs).glob("*.png")):
        im = cv2.imread(str(path))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 35,
                              param1=110, param2=35, minRadius=10, maxRadius=55)
        circles = [] if cs is None else np.round(cs[0]).astype(int).tolist()
        circles = [c for c in circles if 80 < c[1] < im.shape[0] - 40]
        print(path.stem, circles)
        small = cv2.resize(im, (672, 384), interpolation=cv2.INTER_AREA)
        for x, y, r in circles:
            cv2.circle(small, (x // 2, y // 2), max(4, r // 2), (0, 255, 0), 2)
            cv2.putText(small, f"{x},{y},{r}", (x // 2 + 5, y // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(small, (0, 0), (672, 28), (0, 0, 0), -1)
        cv2.putText(small, path.stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(small)
    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(args.output, sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main()
