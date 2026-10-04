"""Observed ice segmentation and a visible-melting prerequisite.

Task-local symlinks expose this source to P21/P21b. Frozen task snapshots copy
and hash the resolved contents. No final frame or desired outcome seeds ice.
"""
from __future__ import annotations
from pathlib import Path
import gc
import hashlib
import inspect
import os
import tempfile
import cv2
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _solid_color_masks(frames,vessel,seed_box):
    """Flat, opaque schematic bodies have a directly observable color ID."""
    x0,y0,x1,y1=map(int,vessel);sx,sy,ex,ey=map(int,seed_box)
    cx,cy=(sx+ex)//2,(sy+ey)//2
    patch=frames[0][max(0,cy-5):cy+6,max(0,cx-5):cx+6].astype(float)
    if not patch.size:return None
    color=np.median(patch.reshape(-1,3),axis=0)
    if np.percentile(np.linalg.norm(patch-color,axis=2),95)>8:return None
    crop=frames[0][y0:y1,x0:x1];side=crop[:,max(2,int(.05*(x1-x0))):max(4,int(.15*(x1-x0)))]
    if not side.size:return None
    if np.linalg.norm(color-np.median(side.reshape(-1,3),axis=0))<35:return None
    masks=[]
    for frame in frames:
        mask=np.zeros(frame.shape[:2],np.uint8)
        mask[y0:y1,x0:x1]=(np.linalg.norm(frame[y0:y1,x0:x1].astype(float)-color,axis=2)<18).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        masks.append(mask.astype(bool))
    ys,xs=np.where(masks[0])
    if len(xs)<50 or xs.min()<sx-3 or xs.max()>ex+3 or ys.min()<sy-3 or ys.max()>ey+3:return None
    return masks,{'method':'resolved_uniform_first_frame_ice_color','seed_bgr':color.tolist()}


def _neural_masks(frames,vessel,seed_box):
    """Local Grounding DINO frame-zero identity, official SAM 2 video masks."""
    import fcntl
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2_video_predictor
    import sys
    root=next(candidate for parent in Path(__file__).resolve().parents for candidate in (parent, parent / "_shared") if (candidate / "task_catalog.json").is_file() and (candidate / "shared/model_paths.py").is_file())
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from shared.model_paths import model_path
    dino=model_path('grounding-dino-tiny','EVALUATOR_GROUNDING_DINO_MODEL')
    checkpoint=model_path('sam2.1-hiera-small/sam2.1_hiera_small.pt','EVALUATOR_SAM2_SMALL_CHECKPOINT')
    if not torch.cuda.is_available():raise RuntimeError('Ice segmentation requires the installed SAM 2 CUDA environment')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    with open('/tmp/vdmbench-v4-'+str(os.getuid())+'-gpu-'+gpu+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        torch.set_num_threads(2);torch.cuda.set_device(0);device='cuda:0'
        processor=AutoProcessor.from_pretrained(str(dino),local_files_only=True)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(str(dino),local_files_only=True).to(device).eval()
        with torch.inference_mode():
            inputs=processor(images=Image.fromarray(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB)),text='ice cube.',return_tensors='pt').to(device)
            output=model(**inputs)
            method=processor.post_process_grounded_object_detection
            kw={'threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold':.25}
            found=method(output,inputs.input_ids,**kw,text_threshold=.20,target_sizes=[frames[0].shape[:2]])[0]
            choices=sorted(zip(found['boxes'].cpu().tolist(),found['scores'].cpu().tolist()),key=lambda x:-x[1])
        del model,output,inputs;gc.collect();torch.cuda.empty_cache()
        vx,vy,ve,vb=vessel;sx,sy,se,sb=seed_box
        choices=[(box,score) for box,score in choices if vx<=(box[0]+box[2])/2<=ve and vy<=(box[1]+box[3])/2<=vb
                 and box[2]-box[0]<.85*(ve-vx) and box[3]-box[1]<.70*(vb-vy)
                 and max(0,min(box[2],se)-max(box[0],sx))*max(0,min(box[3],sb)-max(box[1],sy))>0]
        if not choices:return [],{'method':'grounding_dino_sam2','initial_detection_failed':True}
        box,confidence=choices[0]
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(checkpoint),device=device)
        predictor.fill_hole_area=0
        masks=[None]*len(frames)
        with tempfile.TemporaryDirectory(prefix='vdmbench_ice_') as directory:
            for i,frame in enumerate(frames):
                if not cv2.imwrite(str(Path(directory)/f'{i:06d}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,100]):raise RuntimeError('Cannot stage ice video')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=predictor.init_state(directory,offload_video_to_cpu=True,offload_state_to_cpu=True)
                predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,box=np.array(box,np.float32))
                for i,ids,logits in predictor.propagate_in_video(state):masks[i]=(logits[0,0]>0).cpu().numpy()
        del state,predictor;gc.collect();torch.cuda.empty_cache()
        if any(m is None for m in masks):raise RuntimeError('SAM 2 did not return every ice frame')
        return masks,{'method':'grounding_dino_frame_zero_and_official_sam2_video',
                      'initial_box_xyxy':box,'initial_detection_score':confidence,
                      'sam2_checkpoint_sha256':_sha(checkpoint),
                      'dino_files_sha256':{p.name:_sha(p) for p in dino.iterdir() if p.suffix in ('.json','.safetensors')},
                      'initialization_frame':0,'future_frames_used_for_initialization':False}


def _median_filter_1d(values, radius=1):
    values=np.asarray(values,float)
    if len(values)<=2*radius:return values.copy()
    padded=np.pad(values,(radius,radius),mode='edge')
    return np.asarray([np.median(padded[i:i+2*radius+1]) for i in range(len(values))],float)


def _visible_shrink_event(observations,areas,baseline):
    """Find sustained visible shrinkage with tolerance for mask quantization.

    The old implementation required three exact area bands in sequence. That
    rejected monotonic clips when SAM skipped one band in a single frame. The
    revised rule smooths the projected-area curve, tolerates one moderate
    segmentation jump, and accepts two of three checkpoints when the overall
    reduction and persistence are still clear. A skipped checkpoint is kept as
    an explicit low-confidence flag; identity loss and long gaps remain hard
    failures. Area is still projected-area evidence, not a mass measurement.
    """
    runs=[];run=[];breaks=[]
    for i,(obs,area) in enumerate(zip(observations,areas)):
        reason=None
        if not np.isfinite(area) or area<=0:reason='not_observed'
        elif not obs.get('boundary_clear',False):reason='vessel_boundary_not_clear'
        elif run:
            previous=run[-1]
            if obs.get('frame',i)!=observations[previous].get('frame',previous)+1:
                reason='nonconsecutive_frames'
            else:
                smaller=min(area,areas[previous])
                # Generated-video masks often quantize a smooth boundary into
                # one larger step. Keep continuity for moderate jumps, while
                # still breaking on gross migration to another object.
                if abs(area-areas[previous])>max(.35*smaller,3.5*np.sqrt(smaller)):
                    reason='abrupt_area_change'
        if reason:
            if run:runs.append(run);run=[]
            if reason not in ('not_observed','vessel_boundary_not_clear'):
                breaks.append({'frame':obs.get('frame',i),'reason':reason})
            if reason in ('not_observed','vessel_boundary_not_clear'):continue
        run.append(i)
    if run:runs.append(run)
    if baseline<=0:return None,breaks
    for run in runs:
        # Anchor the identity to its initial scale; a small reappearing mask
        # cannot inherit shrink evidence across an occlusion or tracking jump.
        if len(run)<8 or not .75<=areas[run[0]]/baseline<=1.25:continue
        raw_scaled=areas[run]/baseline
        scaled=_median_filter_1d(raw_scaled,1)
        minimum_ratio=float(np.min(scaled))
        reduction=float(scaled[0]-minimum_ratio)
        if reduction<.30 or minimum_ratio>.72:continue
        steps=np.diff(scaled)
        downward_fraction=float(np.mean(steps<=.035)) if len(steps) else 0.0
        largest_rebound=float(np.max(steps)) if len(steps) else 0.0
        if downward_fraction<.70 or largest_rebound>.12:continue

        # A frame can jump across a checkpoint. Require two checkpoints plus
        # persistence, instead of insisting on all three exact bands.
        cursor=0;stage_hits=[]
        for stage_index,(low,high,hold) in enumerate(((.80,.95,2),(.65,.80,2),(0.,.70,4))):
            found=None
            for j in range(cursor,len(run)-hold+1):
                if np.all((scaled[j:j+hold]>=low)&(scaled[j:j+hold]<high)):
                    found=j;break
            if found is not None:
                stage_hits.append({'stage':stage_index,'start':found,'end':found+hold-1})
                cursor=found+hold
        if len(stage_hits)<2:continue
        confirmation_index=stage_hits[-1]['end']
        # The final checkpoint itself already requires a four-frame hold.
        # Do not require an additional post-confirmation tail: many clips end
        # immediately after the visible shrink event.
        confidence='high' if len(stage_hits)==3 else 'low'
        stage_frames=[observations[run[item['start']]].get('frame',run[item['start']]) for item in stage_hits]
        end=run[confirmation_index]
        return {'start_frame':observations[run[0]].get('frame',run[0]),
                'confirmed_frame':observations[end].get('frame',end),
                'continuous_run_end_frame':observations[run[-1]].get('frame',run[-1]),
                'intermediate_stage_frames':stage_frames,
                'stage_count':len(stage_hits),'required_stage_count':2,
                'skipped_stage_count':3-len(stage_hits),
                'confidence':confidence,
                'confirmation_area_ratio':float(np.median(scaled[max(0,confirmation_index-3):confirmation_index+1])),
                'minimum_area_ratio':minimum_ratio,
                'minimum_resolved_area_reduction':reduction,
                'monotonic_decrease_fraction':downward_fraction,
                'largest_smoothed_rebound':largest_rebound,
                'continuous_observed_frames':len(run)},breaks
    return None,breaks


def melting_from_observations(observations):
    n=len(observations);window=max(3,min(12,n//8))
    areas=np.array([o['area'] if o['observed'] else np.nan for o in observations],dtype=float)
    first=areas[:window];last=areas[-window:]
    baseline=float(np.nanmedian(first)) if np.isfinite(first).any() else 0.
    ratio=float(np.nanmedian(last)/baseline) if baseline>0 and np.isfinite(last).sum()>=window//2 else None
    evidence,breaks=_visible_shrink_event(observations,areas,baseline)
    progress=evidence is not None
    confidence=evidence.get('confidence') if progress else None
    observed=np.flatnonzero(np.isfinite(areas))
    disappeared=False
    if progress and ratio is None and len(observed):
        i=int(observed[-1]);frame=observations[i].get('frame',i)
        disappeared=bool(areas[i]<.25*baseline and observations[i]['boundary_clear'] and i<n-window
                         and evidence['start_frame']<=frame<=evidence['continuous_run_end_frame'])
    return {'initial_envelope_area_px':baseline,'final_to_initial_area_ratio':ratio,
            'initial_identity_resolved':baseline>0,'observed_coverage':float(np.mean(np.isfinite(areas))) if n else 0.,
            'final_ice_detected':bool(ratio is not None and ratio>.03),
            'gradual_disappearance_observed':disappeared,'melting_observed':progress,
            'melting_confidence':confidence,
            'melting_prerequisite_state':'pass' if confidence=='high' else 'low_confidence' if confidence=='low' else 'fail',
            'reason':None if progress else 'initial_ice_identity_unresolved' if baseline<=0 else 'visible_melting_not_observed' if ratio is not None else 'ice_track_lost_without_observed_melting',
            'minimum_resolved_area_reduction':.30,
            'melting_evidence':evidence,'observation_discontinuities':breaks,
            'melting_detection_rule':'smoothed_monotonic_shrink_with_two_of_three_intermediate_checkpoints',
            'melting_evidence_limit':'Projected area cannot independently exclude gradual internal occlusion or a changing viewpoint.'}


def observe_melting(frames,vessel,seed_box,*,rgb=False,debug_dir=None):
    bgr=[cv2.cvtColor(f,cv2.COLOR_RGB2BGR) for f in frames] if rgb else frames
    segmented=_solid_color_masks(bgr,vessel,seed_box)
    if segmented is None:segmented=_neural_masks(bgr,vessel,seed_box)
    masks,identity=segmented;observations=[];vx,vy,ve,vb=vessel
    for i in range(len(frames)):
        mask=masks[i] if masks else np.zeros(frames[0].shape[:2],bool)
        ys,xs=np.where(mask);ok=len(xs)>=15
        row={'frame':i,'observed':ok,'area':float(len(xs)) if ok else None,'box':None,'boundary_clear':False}
        if ok:
            x,y,xe,ye=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
            row.update(box=[x,y,xe-x,ye-y],boundary_clear=bool(vx+3<x<xe<ve-3 and vy+3<y<ye<vb-3))
        observations.append(row)
    result={**identity,'quantity':'segmented projected ice area, not ice mass','observations':observations,
            **melting_from_observations(observations)}
    if debug_dir and masks:
        path=Path(debug_dir);path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'ice_masks.npz',masks=np.array(masks),initialization_frame=np.array(0))
        panels=[]
        for i in np.linspace(0,len(frames)-1,6).astype(int):
            image=bgr[i].copy();image[masks[i]]=(.55*image[masks[i]]+.45*np.array([0,220,255])).astype(np.uint8)
            panels.append(cv2.resize(image,(384,round(image.shape[0]*384/image.shape[1]))))
        cv2.imwrite(str(path/'ice_segmentation_contact.jpg'),np.hstack(panels))
        result['mask_artifact']=str((path/'ice_masks.npz').resolve())
    return result
