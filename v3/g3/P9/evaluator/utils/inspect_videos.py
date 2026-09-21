#!/usr/bin/env python3
from pathlib import Path
import argparse
import av
import cv2
import numpy as np


def decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--videos',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); rows=[]
    for path in sorted(Path(a.videos).glob('*seed42.mp4')):
        fs=decode(path); ids=np.linspace(0,len(fs)-1,6).round().astype(int); tiles=[]
        for i in ids:
            x=cv2.resize(fs[i],(336,192),interpolation=cv2.INTER_AREA)
            cv2.rectangle(x,(0,0),(336,24),(0,0,0),-1)
            cv2.putText(x,f'{path.stem} f{i}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
            tiles.append(x)
        rows.append(np.hstack(tiles))
    cv2.imwrite(a.output,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,80])
if __name__=='__main__': main()
