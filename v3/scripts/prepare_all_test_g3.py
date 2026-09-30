#!/usr/bin/env python3
"""Freeze new P3/P9 initialization from source images, with frame-zero panels."""
import argparse
import os
from copy import deepcopy
import cv2
import numpy as np
import yaml
from all_test_common import ROOT, V3, SOURCE, digest, write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--review-subdir", default="")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    models = sorted(p for p in (SOURCE / 'data/videos/all_test').iterdir() if p.is_dir() and (not args.models or p.name in args.models) and all(os.access(v, os.R_OK) for v in (p/"gpt").glob("*.mp4")))
    for task in ['P3', 'P9']:
        image = V3 / 'g3' / task / 'first_frames' / 'gpt' / 'gpt_01.png'
        still = cv2.imread(str(image)); sh, sw = still.shape[:2]
        hsv = cv2.cvtColor(still, cv2.COLOR_BGR2HSV)
        balls = {}
        for color in ['red', 'blue']:
            hue = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) if color == 'red' else ((hsv[:,:,0] > 100) & (hsv[:,:,0] < 140))
            mask = (hue & (hsv[:,:,1] > 100) & (hsv[:,:,2] > 90)).astype('uint8')
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
            choices = [i for i in range(1,n) if stats[i,4] > 100]
            if len(choices) != 1:
                raise ValueError(f'{task} {color}: ambiguous first-frame colored object')
            i = choices[0]
            balls[color] = {'center': centers[i].tolist(), 'radius': float(np.sqrt(stats[i,4] / np.pi))}
        # Attachment points visually reviewed on the original 1659x948 P9 image.
        pivots = {'red': [554., 113.], 'blue': [1085., 113.]}
        basefile = V3 / 'g3' / task / 'evaluator/config.yaml'
        base = yaml.safe_load(basefile.read_text())
        sheet = np.full((4*254, len(models)*384,3),245,np.uint8)
        for col, model in enumerate(models):
            for row, video in enumerate(sorted((model/'gpt').glob(f'g3_{task}_seed*.mp4'))):
                cap=cv2.VideoCapture(str(video));ok,frame=cap.read();cap.release()
                if not ok: raise ValueError(f'Cannot decode {video}')
                h,w=frame.shape[:2];scale=np.array([w/sw,h/sh]); sample={'first_frame':str(image)}
                if task == 'P3':
                    sample.update(method='gpt', balls={})
                    for slot,color,angle in [('upper','blue',60.),('lower','red',30.)]:
                        c=np.array(balls[color]['center'])*scale;r=balls[color]['radius']*float(np.mean(scale))
                        sample['balls'][slot]={'cx':float(c[0]),'cy':float(c[1]),'radius':r,'angle_deg':angle}
                else:
                    for slot,color in [('short','red'),('long','blue')]:
                        sample[slot]={'bob':(np.array(balls[color]['center'])*scale).tolist(),
                                      'pivot':(np.array(pivots[color])*scale).tolist(),
                                      'radius':balls[color]['radius']*float(np.mean(scale))}
                cfg=deepcopy(base);cfg['samples'][f'g3_{task}']=sample
                out=ROOT/('v3_'+model.name)/'inputs/configs';out.mkdir(parents=True,exist_ok=True)
                config=out/(video.stem+'.yaml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
                crop_errors=[];diff=abs(cv2.GaussianBlur(cv2.resize(still,(w,h)),(5,5),0).astype(float)-cv2.GaussianBlur(frame,(5,5),0).astype(float))
                for color in ['red','blue']:
                    c=np.rint(np.array(balls[color]['center'])*scale).astype(int);r=round(balls[color]['radius']*float(np.mean(scale)))
                    crop_errors.append(float(np.mean(diff[c[1]-r:c[1]+r+1,c[0]-r:c[0]+r+1])))
                    cv2.circle(frame,tuple(c),r+2,(0,255,0),2);cv2.circle(frame,tuple(c),2,(0,255,0),-1)
                    if task=='P9':
                        pivot=np.rint(np.array(pivots[color])*scale).astype(int)
                        cv2.circle(frame,tuple(pivot),4,(0,255,255),-1)
                association={'created_at_utc':now(),'task':task,'video':str(video),'video_sha256':digest(video),
                             'source_image':str(image),'source_image_sha256':digest(image),'source_size_wh':[sw,sh],
                             'video_size_wh':[w,h],'source_color_components':balls,'scale_xy':scale.tolist(),
                             'base_config':str(basefile),'base_config_sha256':digest(basefile),'config_sha256':digest(config),
                             'changed_fields':[f'samples.g3_{task}'],'physics_thresholds_unchanged':True,
                             'identity_evidence':'P3: upper blue 60 degrees, lower red 30 degrees, fixed from source prompt/image. P9: left red short string, right blue long string, fixed from frame zero.',
                             'first_frame_object_crop_mad':crop_errors,'visual_review_scope':'Frame-zero initialization only; all model/seed panels reviewed before tracking. No motion or physics scores used.'}
                write_json(config.with_suffix('.association.json'),association)
                tile=np.full((254,384,3),245,np.uint8)
                cv2.putText(tile,model.name[:31],(4,15),0,.42,(0,0,0),1)
                cv2.putText(tile,video.stem,(4,31),0,.4,(0,0,0),1)
                tile[36:254]=cv2.resize(frame,(384,218))
                sheet[row*254:(row+1)*254,col*384:(col+1)*384]=tile
        cv2.imwrite(str(ROOT/'input_review'/args.review_subdir/(task+'_initialization.jpg')),sheet)
    # P6 uses the same source image as the modern fixture. Only the filename
    # association is new; video_size and every physical threshold stay frozen.
    import json
    from pathlib import Path
    basefile = V3 / 'g3/P6/evaluator/config_v1.yaml'
    base = yaml.safe_load(basefile.read_text())
    base['sources']['g3_P6'] = deepcopy(base['sources']['P6_gpt_01_modern'])
    for model in models:
        out = ROOT / ('v3_' + model.name) / 'inputs/configs'
        out.mkdir(parents=True, exist_ok=True)
        for video in (model / 'gpt').glob('g3_P6_seed*.mp4'):
            (out / (video.stem + '.yaml')).write_text(yaml.safe_dump(base, sort_keys=False))
    print('Prepared P3/P9 first-frame profiles and the P6 input filename alias; physical settings unchanged.',flush=True)


if __name__=='__main__':main()
