"""Observed P28 ball/thread tracking, initialized by frame-zero search boxes.

Boxes select identities only. Later positions and string directions are read
from pixels; no equilibrium angle, symmetry or expected score guides tracking.
"""
import itertools
import cv2
import numpy as np


def circles(frame):
    height,width=frame.shape[:2]
    scale=min(1.,640/width)
    gray=cv2.cvtColor(cv2.resize(frame,None,fx=scale,fy=scale),cv2.COLOR_RGB2GRAY)
    h=gray.shape[0]
    proposals=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,1.2,
        minDist=max(8,h*.032),param1=90,param2=16,
        minRadius=max(4,int(h*.015)),maxRadius=max(8,int(h*.075)))
    if proposals is None:return []
    out=[]
    for x,y,radius in proposals.reshape(-1,3)/scale:
        if y<height*.20 or x-radius<0 or x+radius>=width or y+radius>=height:continue
        out.append({'x':float(x),'y':float(y),'radius':float(radius),
                    'width':float(2*radius),'height':float(2*radius),
                    'area':float(np.pi*radius**2),'method':'observed_circular_outline'})
    return out


def match_pair(candidates,boxes,previous,initial,missing,shape):
    h,w=shape[:2];choices=[]
    for side in ('left','right'):
        reference=previous.get(side)
        box=boxes[side+'_ball'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        scored=[]
        for index,item in enumerate(candidates):
            x,y,r=item['x'],item['y'],item['radius']
            if reference is None:
                if not (box[0]<=x<=box[2] and box[1]<=y<=box[3]):continue
                cost=((x-cx)**2+(y-cy)**2)**.5/max(1,r)
            else:
                expected_r=initial[side]['radius']
                if not .55*expected_r<=r<=1.8*expected_r:continue
                distance=np.hypot(x-reference['x'],y-reference['y'])
                if distance>max(4*expected_r,.07*h)*(1+min(missing[side],5)):continue
                cost=distance/expected_r+abs(np.log(r/expected_r))
            scored.append((cost,index,item))
        choices.append(sorted(scored,key=lambda x:x[0]))
    pairs=[(a[0]+b[0],a,b) for a,b in itertools.product(*choices) if a[1]!=b[1]]
    if pairs:
        _,left,right=min(pairs,key=lambda x:x[0]);return {'left':left[2],'right':right[2]}
    # Keep independently found observations without assigning the same circle
    # to two identities. Missing coordinates remain missing.
    result={};used=set()
    for side,options in zip(('left','right'),choices):
        choice=next((x for x in options if x[1] not in used),None)
        result[side]=choice[2] if choice else None
        if choice:used.add(choice[1])
    return result


def observed_thread(frame,ball,box,support_box,anchor=None):
    if ball is None:return None
    h,w=frame.shape[:2];gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # The top search box is an initial apparatus region; the lower search area
    # follows the actually detected ball instead of clipping its displacement.
    xlo=max(0,int(min(box[0],ball['x']-2*ball['radius'])))
    xhi=min(w,int(max(box[2],ball['x']+2*ball['radius'])))
    ylo=max(0,support_box[1]);yhi=min(h,int(ball['y']-.65*ball['radius']))
    if yhi-ylo<25:return None
    crop=gray[ylo:yhi,xlo:xhi]
    edge=cv2.Canny(crop,40,120)
    # Subpixel line segments preserve long strings whose slopes fall between
    # Hough accumulator bins. Segment endpoints remain image observations.
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(crop)[0]
    if raw is None:return None
    center=np.array([ball['x'],ball['y']]);candidates=[]
    yy,xx=np.where(edge>0);edge_points=np.c_[xx+xlo,yy+ylo].astype(float)
    for values in raw.reshape(-1,4):
        a=np.array(values[:2],float)+[xlo,ylo];b=np.array(values[2:],float)+[xlo,ylo]
        if a[1]>b[1]:a,b=b,a
        direction=b-a;length=np.linalg.norm(direction)
        if length<max(20,.20*(yhi-ylo)) or direction[1]<.5*length:continue
        offset=center-a
        distance=abs(direction[0]*offset[1]-direction[1]*offset[0])/length
        if distance>max(3.,.4*ball['radius']):continue
        direction/=length
        normal=np.array([-direction[1],direction[0]])
        points=edge_points[np.abs((edge_points-a)@normal)<max(3.,.25*ball['radius'])]
        if len(points)<25:continue
        # Join only observed edge pixels along the candidate direction. Short
        # detector segments may share a continuous string; large pixel gaps
        # remain missing, rather than being filled by an assumed straight line.
        rows=np.unique(points[:,1]);cuts=np.where(np.diff(rows)>max(4.,.01*h))[0]+1
        spans=np.split(rows,cuts)
        spans=[r for r in spans if len(r)>0 and r[0]<=support_box[3]
               and r[-1]>=center[1]-2.5*ball['radius']]
        if not spans:continue
        span=max(spans,key=len)
        if len(span)/max(1,span[-1]-span[0]+1)<.80:continue
        points=points[(points[:,1]>=span[0])&(points[:,1]<=span[-1])]
        vx,vy,px,py=cv2.fitLine(points.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel()
        fitted=np.array([float(vx),float(vy)])
        if fitted[1]<0:fitted=-fitted
        origin=np.array([float(px),float(py)]);projected=(points-origin)@fitted
        a=origin+projected.min()*fitted;b=origin+projected.max()*fitted
        residual=float(np.mean(np.abs((points-origin)@np.array([-fitted[1],fitted[0]]))))
        if residual>max(2.,.1*ball['radius']):continue
        if not box[0]<=a[0]<=box[2]:continue
        anchor_error=0. if anchor is None else np.linalg.norm(a-anchor)
        if anchor is not None and anchor_error>max(8.,.04*h):continue
        length=float(np.linalg.norm(b-a))
        if length<.6*(center[1]-support_box[3]):continue
        cost=distance+residual+.05*anchor_error-.015*length
        candidates.append((cost,{'a':a,'b':b,'length':length,
            'residual_px':residual,'points':points,'method':'observed_thread_edges',
            'observed_row_fraction':float(len(span)/(span[-1]-span[0]+1))}))
    if not candidates:return None
    return min(candidates,key=lambda x:x[0])[1]


def track(frames,boxes):
    balls={s:[] for s in ('left','right')};threads={s:[] for s in balls}
    previous={};initial={};anchors={};initial_frames={};anchor_frames={};missing={s:0 for s in balls}
    for frame_index,frame in enumerate(frames):
        pair=match_pair(circles(frame),boxes,previous,initial,missing,frame.shape)
        for side in balls:
            ball=pair[side];balls[side].append(ball)
            if ball is not None:
                previous[side]=ball;initial.setdefault(side,ball);initial_frames.setdefault(side,frame_index);missing[side]=0
            else:missing[side]+=1
            line=observed_thread(frame,ball,boxes[side+'_thread'],boxes['support'],anchors.get(side))
            threads[side].append(line)
            if line is not None:
                anchors.setdefault(side,line['a']);anchor_frames.setdefault(side,frame_index)
    return balls,threads,{'method':'Initial search-box identity + circular outlines + independently observed thread edges',
                          'initial_balls':initial,'initial_observed_anchors':anchors,
                          'ball_initialization_frames':initial_frames,'anchor_initialization_frames':anchor_frames,
                          'future_truth_used':False,'missing_positions_interpolated':False}
