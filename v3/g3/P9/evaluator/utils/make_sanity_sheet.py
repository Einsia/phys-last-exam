#!/usr/bin/env python3
"""Build labelled key-frame contact sheets from evaluator overlays."""
import argparse
from pathlib import Path
import av, cv2, numpy as np

def read(path):
    with av.open(str(path)) as c: return [f.to_ndarray(format='bgr24') for f in c.decode(video=0)]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',required=True); p.add_argument('--pattern',default='**/overlay.mp4'); p.add_argument('--frames',type=int,default=3); p.add_argument('--width',type=int,default=280); a=p.parse_args()
    rows=[]
    for path in sorted(Path(a.root).glob(a.pattern)):
        fs=read(path); ids=np.linspace(0,len(fs)-1,a.frames).round().astype(int); tiles=[]
        for i in ids:
            height=round(a.width*768/1344); im=cv2.resize(fs[i],(a.width,height),interpolation=cv2.INTER_AREA)
            cv2.rectangle(im,(0,0),(a.width,20),(0,0,0),-1); cv2.putText(im,f'{path.parent.name} f{i}',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA); tiles.append(im)
        rows.append(np.hstack(tiles))
    if not rows: raise SystemExit('no overlays found')
    cv2.imwrite(a.output,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,82])
if __name__=='__main__': main()
