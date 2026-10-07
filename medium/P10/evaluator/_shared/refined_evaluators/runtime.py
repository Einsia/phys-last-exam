"""Single-video CLI for the refined g8/g9 task-specific evaluators."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,time,traceback,math
from .common import write_json,ExtractionError,EnvironmentError,path_component
from .resources import ROOT, resource, model_resource, task_resource
from .output import finalize
ROOT=Path(__file__).resolve().parents[1]


def parser(task):
    from .definitions import DEFAULTS
    p=argparse.ArgumentParser(description=f'{task}: reviewed first-frame identity, SAM2 + CoTracker3, task-specific M1')
    annotation=task_resource(task)/'first_frame_annotations.json'
    p.set_defaults(task_id=task)
    p.add_argument('--video_path',required=True);p.add_argument('--image_path');g=p.add_mutually_exclusive_group();g.add_argument('--video_prompt');g.add_argument('--video_prompt_file')
    p.add_argument('--model');p.add_argument('--sample_id');p.add_argument('--seed',type=int);p.add_argument('--output',required=True);p.add_argument('--debug_dir');p.add_argument('--annotation',default=str(annotation) if annotation.is_file() else None,help='Override the bundled task-image annotation');p.add_argument('--device',default='auto');p.add_argument('--config');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dino_model',default=str(model_resource('grounding-dino-tiny')),help='Context detector path; reviewed first-frame annotation overrides unreliable whole-apparatus detections')
    p.add_argument('--sam2_checkpoint',default=str(model_resource('sam2.1-hiera-small/sam2.1_hiera_small.pt')));p.add_argument('--sam2_config',default='configs/sam2.1/sam2.1_hiera_s.yaml');p.add_argument('--cotracker_checkpoint',default=str(model_resource('cotracker3/scaled_offline.pth')));p.add_argument('--reuse_masks',action='store_true');p.add_argument('--reuse_tracks',action='store_true')
    for k,v in DEFAULTS[task].items():p.add_argument('--'+k,type=type(v),default=None)
    return p


def new_result(task,video,image=None,prompt=None,model=None,seed=None,sample_id=None):
    from .definitions import PRINCIPLES
    return {'task_id':task,'video_path':str(video),'image_path':str(image) if image else None,'video_prompt':prompt,'model':model,'seed':seed,'sample_id':sample_id,'metrics':{'M1':{'extract_success':False,'metric':None},'M2':{'extract_success':None,'metric':None}},'verbose':{'M1':{'principle':PRINCIPLES.get(task),'status':'extraction_failed','measurements':{},'score_details':{},'thresholds':{},'evidence':[],'reason':None},'M2':None}}


def run(task,argv=None):
    if task == 'P39':
        from .bubbles import run as run_bubbles
        return run_bubbles(argv)
    args=parser(task).parse_args(argv);started=time.monotonic();sid=path_component(args.sample_id or Path(args.video_path).stem,'sample_id');out=Path(args.debug_dir) if args.debug_dir else Path(args.output).parent/('debug_'+sid);out.mkdir(parents=True,exist_ok=True)
    result=new_result(task,args.video_path,args.image_path,args.video_prompt,args.model,args.seed,sid);m=result['verbose']['M1'];c=None;code=0
    try:
        if args.video_prompt_file:result['video_prompt']=Path(args.video_prompt_file).read_text().strip()
        from .tasks import TASKS,DEFAULTS
        cfg=dict(DEFAULTS[task])
        if args.config:
            overrides=json.loads(Path(args.config).read_text())
            if not isinstance(overrides,dict) or set(overrides)-set(cfg):raise ValueError('Unknown threshold config keys')
            cfg.update(overrides)
        cfg.update({k:getattr(args,k) for k in cfg if getattr(args,k) is not None})
        for k,v in cfg.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or (v<0 if k=='onset_tolerance_sec' else v<=0):raise ValueError(f'{k} is outside its finite allowed range')
        if cfg.get('margin',1)>1:raise ValueError('margin must satisfy 0 < margin <= 1')
        m['thresholds']=cfg
        import cv2,numpy as np,torch
        from .media import decode
        from .vision import load_annotation,segment,refine_masks,track,detect_context
        from .export import export
        if args.threads<1:raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads);cv2.setNumThreads(args.threads)
        if args.device=='auto':args.device='cuda:'+str(max(range(torch.cuda.device_count()),key=lambda i:torch.cuda.mem_get_info(i)[0])) if torch.cuda.is_available() else 'cpu'
        if args.device.startswith('cuda'):torch.cuda.set_device(args.device)
        for key in ['sam2_checkpoint','cotracker_checkpoint']:
            if not Path(getattr(args,key)).is_file():raise EnvironmentError(f'Missing {key}: {getattr(args,key)}')
        print(f'{task}: decoding and validating first-frame identities on {args.device}',flush=True)
        frames,t,pts=decode(args.video_path);write_json(out/'timestamps.json',[dict(x,frame_index=i) for i,x in enumerate(pts)]);a=load_annotation(args,frames,out)
        detect_context(frames[0],args,a,out)
        masks=refine_masks(frames,segment(frames,args,a,out),a)
        # Only rewrite this cache when its actual contents differ. This avoids invalidating tracks merely because of ZIP timestamps.
        maskfile=out/'masks.npz';rewrite=True
        if maskfile.exists():
            with np.load(maskfile) as d:rewrite=not np.array_equal(d['masks'],masks)
        if rewrite:np.savez_compressed(maskfile,masks=masks,object_names=np.array([o['name'] for o in a['objects']]))
        c=SimpleNamespace(task=task,out=out,frames=frames,t=t,pts=pts,a=a,masks=masks,m=m,cfg=cfg,xy=None,vis=None,groups={},annotations={},extra_masks={},calculation=[])
        print(f'{task}: tracking actual material points',flush=True);c.xy,c.vis,c.groups=track(frames,masks,args,a,out)
        m['evidence']=[{'stage':'first_frame_identity','path':str(out/'first_frame_identity.png'),'source':a['annotation_type'],'identities':{o['name']:o['identity_evidence'] for o in a['objects']}},{'stage':'segmentation','method':'SAM2.1 Hiera Small; actual pixel masks and documented color refinement','path':str(maskfile)},{'stage':'tracking','method':'CoTracker3 visible material points and static-reference drift compensation','path':str(out/'cotracker_tracks.npz')}]
        m['runtime']={'device':args.device,'torch':torch.__version__,'opencv':cv2.__version__,'numpy':np.__version__}
        m['extraction_run']={'video_decoded':True,'physical_measurements_recomputed':True,
                             'sam2_cache_reused':getattr(args,'sam2_cache_used',False),
                             'cotracker_cache_reused':getattr(args,'cotracker_cache_used',False),
                             'cache_validation':'SHA-256 of video, annotations, masks and weights plus extraction configuration'}
        score=TASKS[task](c);result['metrics']['M1']={'extract_success':True,'metric':float(score)};m['status']='scored'
    except ExtractionError as e:m.update(status='extraction_failed',reason=str(e));code=1
    except (ImportError,EnvironmentError) as e:m.update(status='environment_error',reason=str(e));code=2
    except (ValueError,FileNotFoundError) as e:m.update(status='configuration_error',reason=str(e));code=2
    except Exception as e:m.update(status='runtime_error',reason=f'{type(e).__name__}: {e}');write_json(out/'runtime_failure.json',{'traceback':traceback.format_exc()});code=2
    finally:
        if c is not None:
            try:
                from .export import export
                m['artifacts']=export(c)
            except Exception as e:
                m['export_error']=f'{type(e).__name__}: {e}';m.update(status='runtime_error',reason='Artifact export failed: '+str(e));result['metrics']['M1']={'extract_success':False,'metric':None};write_json(out/'export_failure.json',{'traceback':traceback.format_exc()});code=2
        lines=[f'# {task} M1 calculation',f'Status: {m["status"]}',f'Reason: {m["reason"]}',f'Parameters: {json.dumps(m["thresholds"],ensure_ascii=False)}']
        if c is not None:lines+=c.calculation
        lines += ["## Measurement and scoring",json.dumps({'measurements':m['measurements'],'score_details':m['score_details'],'M1':result['metrics']['M1']},ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),"M2 is not computed. Annotations provide object identity and initial geometry only; prompts do not enter measurement or scoring."]
        (out/'calculation.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8');m['elapsed_sec']=round(time.monotonic()-started,3)
        result=finalize(result,task_resource(task),out)
        if code == 0 and result['metrics']['M1']['extract_success'] is not True:code=1
        write_json(args.output,result)
    print(f'{task}: {m["status"]}; M1={result["metrics"]["M1"]}; {m["reason"]}',flush=True)
    return code
