"""Image-reviewed initialization, real SAM 2 masks and CoTracker3 material tracks."""
from pathlib import Path
from contextlib import nullcontext
import json, sys, tempfile, gc
import cv2
import numpy as np
from .common import ExtractionError, EnvironmentError, write_json
from .media import fingerprint, frame_fingerprint
from .resources import resource
from .annotations import TEMPLATE_TYPE, bind_task_template
ROOT=Path(__file__).resolve().parents[1]


def load_annotation(args, frames, directory):
    path=Path(args.annotation) if args.annotation else Path(args.video_path).parent/'first_frame_annotations.json'
    if not path.is_file():
        raise ExtractionError('Missing bundled first-frame annotations; supply --annotation for custom inputs.')
    args.annotation=str(path)
    a=json.loads(path.read_text())
    if a.get('annotation_type') == TEMPLATE_TYPE:
        a=bind_task_template(a,args,frames[0],directory,path)
    else:
        if a['source_video_sha256']!=fingerprint(args.video_path):
            raise ExtractionError('Annotation video hash mismatch; do not reuse fixture coordinates on another video.')
        if args.image_path and a['source_image_sha256']!=fingerprint(args.image_path):
            raise ExtractionError('Annotation first-frame image hash mismatch')
        if a['size_wh']!=list(frames[0].shape[1::-1]):
            raise ExtractionError('Annotation dimensions differ from decoded video')
        if a.get('annotation_type') == 'decoded_video_frame_0':
            # A generation input image can differ in color, light or layout from
            # the actual video. Explicit frame-zero annotations are validated
            # against their decoded pixels, rather than against that input image.
            if a.get('frame0_sha256') != frame_fingerprint(frames[0]):
                raise ExtractionError('Annotation decoded first-frame hash mismatch')
            if a.get('coordinate_frame') != 'decoded_video_frame_0':
                raise ExtractionError('Annotation coordinate frame is not video frame zero')
            review=a.get('frame0_review', {})
            if review.get('accepted') is not True or not review.get('evidence'):
                raise ExtractionError('Actual first-frame annotation has no accepted identity review')
            height,width=frames[0].shape[:2]
            for obj in a['objects']:
                x1,y1,x2,y2=obj['box']
                if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                    raise ExtractionError('First-frame object box lies outside the video')
                if not obj.get('positive'):
                    raise ExtractionError('First-frame object has no observed material point')
                if any(not (0<=x<width and 0<=y<height) for x,y in obj['positive']+obj.get('negative',[])):
                    raise ExtractionError('First-frame material point lies outside the video')
            a['image_correspondence']={
                'accepted':True,'method':'exact_decoded_video_frame_hash_and_identity_review',
                'generation_image_used_for_initialization':False,
                'frame0_sha256':a['frame0_sha256']}
        elif args.image_path:
            still=cv2.imread(str(args.image_path))
            if still is None: raise ExtractionError('First-frame image unreadable')
            still=cv2.resize(still,frames[0].shape[1::-1])
            # Images may differ in JPEG noise but must show the same initial object geometry.
            blurred=[cv2.GaussianBlur(x,(5,5),0).astype(float) for x in (still,frames[0])]
            a['image_correspondence']={'mean_absolute_difference':float(np.mean(abs(blurred[0]-blurred[1]))),'object_crop_errors':[]}
            for obj in a['objects']:
                x1,y1,x2,y2=obj['box']
                a['image_correspondence']['object_crop_errors'].append(float(np.mean(abs(blurred[0][y1:y2,x1:x2]-blurred[1][y1:y2,x1:x2]))))
            a['image_correspondence']['accepted']=bool(max(a['image_correspondence']['object_crop_errors'])<15)
            if not a['image_correspondence']['accepted']: raise ExtractionError('Supplied image and video initial object geometry differ')
    canvas=frames[0].copy()
    for j,obj in enumerate(a['objects']):
        x1,y1,x2,y2=obj['box']; color=COLORS[j%len(COLORS)]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        cv2.putText(canvas,obj['name'],(x1,max(25,y1-8)),0,.6,color,2)
        for pt in obj['positive']: cv2.circle(canvas,tuple(pt),4,(0,255,0),-1)
        for pt in obj.get('negative',[]): cv2.circle(canvas,tuple(pt),4,(0,0,255),-1)
    cv2.imwrite(str(directory/'first_frame_identity.png'),canvas)
    if a['geometry'].get('gap_evidence_box'):
        x1,y1,x2,y2=a['geometry']['gap_evidence_box']
        cv2.imwrite(str(directory/'gap_evidence.png'),cv2.resize(frames[0][y1:y2,x1:x2],None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST))
    write_json(directory/'initial_reference.json',a)
    return a

COLORS=[(40,210,40),(255,160,20),(60,80,255),(180,20,230)]


def segment(frames,args,a,directory):
    import torch
    signature={'video':fingerprint(args.video_path),'annotation':fingerprint(args.annotation or Path(args.video_path).parent/'first_frame_annotations.json'),
               'checkpoint':fingerprint(args.sam2_checkpoint),'config':args.sam2_config,'version':3}
    cache=directory/'sam2_masks.npz'
    if args.reuse_masks and cache.is_file():
        with np.load(cache) as x:
            if json.loads(str(x['signature']))!=signature: raise ExtractionError('SAM cache signature mismatch')
            if x['masks'].shape != (len(frames),len(a['objects']),*frames[0].shape[:2]):raise ExtractionError('Invalid SAM cache shape')
            args.sam2_cache_used=True
            return x['masks']
    from sam2.build_sam import build_sam2_video_predictor
    model=build_sam2_video_predictor(args.sam2_config,args.sam2_checkpoint,device=args.device)
    model.fill_hole_area=0
    masks=np.zeros((len(frames),len(a['objects']),*frames[0].shape[:2]),dtype=bool)
    with tempfile.TemporaryDirectory(prefix='refined_sam_') as staging:
        for i,f in enumerate(frames): cv2.imwrite(str(Path(staging)/f'{i:06d}.jpg'),f,[cv2.IMWRITE_JPEG_QUALITY,100])
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
        with torch.inference_mode(),amp:
            state=model.init_state(staging,offload_video_to_cpu=True,offload_state_to_cpu=True)
            for j,obj in enumerate(a['objects']):
                pts=obj['positive']+obj.get('negative',[]); labels=[1]*len(obj['positive'])+[0]*len(obj.get('negative',[]))
                model.add_new_points_or_box(state,frame_idx=0,obj_id=j,box=np.array(obj['box'],dtype=np.float32),points=np.array(pts,dtype=np.float32),labels=np.array(labels))
            seen=[]
            for i,ids,logits in model.propagate_in_video(state):
                for j,oid in enumerate(ids): masks[i,oid]=(logits[j,0]>0).cpu().numpy()
                seen.append(i)
            if len(seen)!=len(frames): raise ExtractionError('Incomplete SAM propagation')
    np.savez_compressed(cache,masks=masks,signature=json.dumps(signature,sort_keys=True))
    del state,model;gc.collect()
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    return masks


def refine_masks(frames,masks,a):
    """Use actual object masks, without inferring material from a fixed hue."""
    out=masks.copy()
    # Blue conducting plates and differently painted magnets retain their
    # independently initialized SAM identities; copper/red hue is not physics.
    if a['task_id'] in ('P34','P30'):return out
    if a['task_id']=='P36':
        references=[]
        for obj in a['objects']:
            colors=[]
            for x,y in obj['positive']:
                x=int(x);y=int(y);patch=frames[0][max(0,y-3):y+4,max(0,x-3):x+4]
                if patch.size:colors.append(np.median(patch.reshape(-1,3),axis=0))
            references.append(colors)
        for i,frame in enumerate(frames):
            for j,colors in enumerate(references):
                if not colors:continue
                distance=np.min([np.linalg.norm(frame.astype(float)-color,axis=2) for color in colors],axis=0)
                out[i,j]&=distance<55.0
    return out


def material_points(frame,mask,count=32):
    eroded=cv2.erode(mask.astype('uint8'),np.ones((5,5),np.uint8))
    pts=cv2.goodFeaturesToTrack(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),count,.002,5,mask=eroded)
    found=[] if pts is None else pts[:,0,:].tolist()
    # Evenly distributed interior points for low-texture metal. Never query holes.
    yy,xx=np.where(eroded)
    if len(xx)==0: raise ExtractionError('Empty initial material mask')
    for k in np.linspace(0,len(xx)-1,min(len(xx),200)).astype(int):
        p=[int(xx[k]),int(yy[k])]
        if all(np.linalg.norm(np.array(p)-q)>6 for q in found): found.append(p)
        if len(found)>=count: break
    return np.array(found[:count],dtype=np.float32)


def track(frames,masks,args,a,directory):
    import torch
    cache=directory/'cotracker_tracks.npz'
    sig={'video':fingerprint(args.video_path),'masks':fingerprint(directory/'masks.npz'),'weights':fingerprint(args.cotracker_checkpoint),'version':3}
    if args.reuse_tracks and cache.exists():
        with np.load(cache) as d:
            if json.loads(str(d['signature']))!=sig: raise ExtractionError('CoTracker cache signature mismatch')
            if d['tracks'].shape[0] != len(frames) or d['visibility'].shape != d['tracks'].shape[:2]:raise ExtractionError('Invalid CoTracker cache shape')
            args.cotracker_cache_used=True
            return d['tracks'],d['visibility'],json.loads(str(d['groups']))
    sys.path.insert(0,str(resource('shared_vendor/co-tracker')))
    from cotracker.predictor import CoTrackerPredictor
    points=[];groups={}
    for j,obj in enumerate(a['objects']):
        p=material_points(frames[0],masks[0,j]); start=len(points);points.extend(p.tolist());groups[obj['name']]=list(range(start,len(points)))
    fixed=np.zeros(frames[0].shape[:2],np.uint8)
    for x1,y1,x2,y2 in a['geometry'].get('fixed_reference_boxes',[]):fixed[y1:y2,x1:x2]=1
    if fixed.any():
        p=material_points(frames[0],fixed.astype(bool),48);start=len(points);points.extend(p.tolist());groups['fixed_reference']=list(range(start,len(points)))
    model=CoTrackerPredictor(checkpoint=args.cotracker_checkpoint,offline=True,window_len=60).to(args.device).eval()
    # Model rescales internally to 384x512. Resize input ourselves to bound host/GPU memory.
    h,w=frames[0].shape[:2];size=(768,round(h*768/w));scale=np.array([(size[0]-1)/(w-1),(size[1]-1)/(h-1)])
    video=np.stack([cv2.cvtColor(cv2.resize(f,size),cv2.COLOR_BGR2RGB) for f in frames])
    v=torch.from_numpy(video).permute(0,3,1,2)[None].float().to(args.device)
    q=np.c_[np.zeros(len(points)),np.array(points)*scale]
    with torch.inference_mode(): xy,vis=model(v,queries=torch.tensor(q,dtype=torch.float32,device=args.device)[None])
    xy=xy[0].cpu().numpy()/scale;vis=vis[0].cpu().numpy().astype(bool)
    np.savez_compressed(cache,tracks=xy,visibility=vis,queries=np.c_[np.zeros(len(points)),points],groups=json.dumps(groups),signature=json.dumps(sig,sort_keys=True))
    del model,v;gc.collect()
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return xy,vis,groups


def camera_motion(xy,vis,groups):
    ids=groups.get('fixed_reference',[])
    if not ids:return np.zeros((len(xy),2)),np.zeros(len(xy))
    d=xy[:,ids]-xy[0,ids]; d[~vis[:,ids]]=np.nan
    offset=np.nanmedian(d,axis=1)
    residual=np.nanmedian(np.linalg.norm(d-offset[:,None,:],axis=2),axis=1)
    if np.any(~np.isfinite(offset)): raise ExtractionError('Static references lost')
    return offset,residual


def detect_context(frame,args,a,directory):
    """Retain DINO proposals as context; never relabel objects by their motion."""
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from PIL import Image
    import torch,inspect
    path=directory/'detections.json'
    query={'P33':'aluminum ring. metal rod. coil.','P34':'metal plate. pendulum.','P30':'bar magnet. coil. light bulb.','P36':'funnel. water. sand.','P10':'rectangular block.','P38':'metal ball.'}[a['task_id']]
    signature={'video':a['source_video_sha256'],'dino_model':str(Path(args.dino_model).resolve()),'query':query,'threshold':.2}
    if args.reuse_masks and path.exists():
        d=json.loads(path.read_text())
        if d.get('signature')==signature:return d
    proc=AutoProcessor.from_pretrained(args.dino_model,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model,local_files_only=True).to(args.device).eval()
    with torch.inference_mode():
        x=proc(images=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),text=query,return_tensors='pt').to(args.device);y=model(**x)
        method=proc.post_process_grounded_object_detection;key='threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
        r=method(y,x.input_ids,**{key:.2},text_threshold=.2,target_sizes=[frame.shape[:2]])[0]
    d={'signature':signature,'boxes_xyxy':r['boxes'].cpu().tolist(),'scores':r['scores'].cpu().tolist(),'labels':r.get('text_labels',[]),'initialization_source':'Reviewed first-frame object boxes/positive/negative points, not potentially whole-apparatus DINO boxes','reviewed_boxes':[o['box'] for o in a['objects']]}
    write_json(path,d);del model,proc,x,y;gc.collect()
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return d
