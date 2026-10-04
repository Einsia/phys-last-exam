"""P45 clear-liquid observations from independently reviewed frame-zero geometry.

No identities, heights, expected ordering, scores or future trajectories are
inferred from filenames. Geometry selects two bores and one shared reservoir;
every accepted meniscus/free surface is read again from that frame's pixels.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def parse_annotation(clip,annotation):
    if annotation.get('annotation_type')!='reviewed_first_frame_geometry':raise ValueError('P45 requires a reviewed first-frame geometry annotation')
    if annotation.get('source_video_sha256')!=_sha(clip.path):raise ValueError('P45 geometry/video SHA-256 mismatch')
    if annotation.get('size_wh')!=[clip.w,clip.h]:raise ValueError('P45 geometry dimensions do not match decoded frames')
    pixels=hashlib.sha256(np.ascontiguousarray(clip[0]).tobytes()).hexdigest()
    if annotation.get('frame_pixels_sha256')!=pixels:raise ValueError('P45 geometry/decoded frame-zero pixel hash mismatch')
    geometry=annotation['geometry']
    def box(value):
        if len(value)!=4 or not all(np.isfinite(value)):raise ValueError('Invalid P45 geometry box')
        x0,y0,x1,y1=map(float,value)
        if not (0<=x0<x1<=clip.w and 0<=y0<y1<=clip.h):raise ValueError('P45 geometry box outside decoded frame')
        return [x0,y0,x1,y1]
    tubes=[box(x) for x in geometry['tube_boxes']]
    if len(tubes)!=2 or min(tubes[0][2],tubes[1][2])>max(tubes[0][0],tubes[1][0]):raise ValueError('P45 must have two separate independently identified bores')
    surface=box(geometry['reservoir_surface_search_band'])
    if not all(surface[0]<(t[0]+t[2])/2<surface[2] for t in tubes):raise ValueError('P45 shared surface search must span both bores')
    result={'tube_boxes':tubes,'reservoir_surface_search_band':surface,'exclude_boxes':[box(x) for x in geometry.get('exclude_boxes',[])],'frame_pixels_sha256':pixels,'annotation_source_video_sha256':annotation['source_video_sha256']}
    result['wall_references']=[_wall_reference(clip[0],t,result) for t in tubes]
    return result

def _wall_reference(frame,box,geometry):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32);edges=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3));a,y0,b,y1=box;width=b-a
    rows=np.arange(max(0,int(y0+width+4)),min(len(gray),int(min(y1-width,geometry['reservoir_surface_search_band'][1]))))
    for x0,top,x1,bottom in geometry['exclude_boxes']:
        if x0< b and x1>a:rows=rows[(rows<top-12)|(rows>bottom+12)]
    if len(rows)<20:return None
    profile=np.median(edges[rows],axis=0);observed=[]
    for lo,hi in ((a-.6*width,a+.2*width),(b-.2*width,b+.6*width)):
        lo=max(0,int(lo));hi=min(len(profile),int(hi)+1)
        x=lo+int(np.argmax(profile[lo:hi]));observed.append({'x':x,'strength':float(profile[x])})
    return {'rows':rows.tolist(),'edges':observed}

def wall_identity(frame,tube,geometry):
    boxes=geometry['tube_boxes'];idx=next(i for i,b in enumerate(boxes) if b[0]==tube['x_left'] and b[2]==tube['x_right']);ref=geometry.get('wall_references',[None]*len(boxes))[idx]
    if ref is None:
        return 'wall_references' not in geometry,{'status':'first-frame tube-wall reference unavailable; no independent geometry-route identity assertion'}
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32);edges=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3));profile=np.median(edges[np.asarray(ref['rows'],int)],axis=0);width=tube['bore'];checks=[]
    for edge in ref['edges']:
        radius=max(3,int(np.ceil(.25*width)));lo=max(0,edge['x']-radius);hi=min(len(profile),edge['x']+radius+1);strength=float(np.max(profile[lo:hi]));checks.append({'reference_x':edge['x'],'current_strength_near_reference':strength,'reference_strength':edge['strength'],'valid':bool(edge['strength']>=4 and strength>=max(4.,.25*edge['strength']))})
    return all(c['valid'] for c in checks),{'status':'checked static first-frame tube-wall identity','checks':checks}

def tubes_from_geometry(geometry):
    return [{'x_left':b[0],'x_right':b[2],'bore':b[2]-b[0],'y_start':b[1],'y_end':b[3],'support':1.0,'identity_method':'reviewed frame-zero inner-bore geometry; no motion or level labels','width_method':'reviewed visible inner walls'} for b in geometry['tube_boxes']]

def _profile(gray,cols):
    return np.median(gray[:,np.asarray(cols,int)],axis=1)

def _response(profile):
    # Four-pixel finite difference retains clear monochrome step/stroke edges.
    out=np.zeros(len(profile),float);out[2:-2]=np.abs(profile[4:]-profile[:-4]);return out

def _peaks(response,lo,hi,count=3,separation=7):
    candidates=[]
    for y in sorted(range(max(2,lo),min(len(response)-2,hi)),key=lambda y:-response[y]):
        if response[y]<4.:break
        if all(abs(y-p)>separation for p in candidates):candidates.append(y)
        if len(candidates)>=count:break
    return candidates

def _interface(profile,proposal):
    # Fit a two-plateau step or a short visible interfacial stroke; never use
    # a desired height/order to choose the subpixel location.
    lo=max(0,int(proposal)-10);p=profile[lo:min(len(profile),int(proposal)+11)];n=len(p)
    if n<15:return float(proposal)
    def loss(a,b):return float(np.sum((p[a:b]-p[a:b].mean())**2))
    two=min((loss(0,k)+loss(k,n),k) for k in range(4,n-3))
    three=min((loss(0,a)+loss(a,b)+loss(b,n),a,b) for a in range(4,n-4) for b in range(a+1,min(n-3,a+7)))
    return float(lo+(three[1]+three[2]-1)/2) if three[0]<.55*max(two[0],1.) else float(lo+two[1]-.5)

def reservoir_surface(frame,geometry):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float);x0,y0,x1,y1=map(int,geometry['reservoir_surface_search_band']);cols=np.arange(x0,x1)
    permitted=np.ones(len(cols),bool)
    for a,b,c,d in geometry['tube_boxes']:
        margin=max(5.,.25*(c-a));permitted&=~((cols>=a-margin)&(cols<=c+margin))
    for a,b,c,d in geometry['exclude_boxes']:
        if b<y1 and d>y0:permitted&=~((cols>=a)&(cols<=c))
    cols=cols[permitted]
    if len(cols)<24:return None,{'reason':'too few reservoir columns outside reviewed bores/occluders'}
    chunks=[ch for ch in np.array_split(cols,16) if len(ch)>=4];candidates=[]
    for index,ch in enumerate(chunks):
        profile=_profile(gray,ch);response=_response(profile)
        for y in _peaks(response,y0+3,y1-3,count=3):candidates.append((float(np.mean(ch)),float(y),float(response[y]),index))
    if len(candidates)<8:return None,{'reason':'no broad visible free-surface edge within reviewed reservoir band'}
    c=np.asarray(candidates);best=None
    # Fit consensus across independently observed column groups. The strongest
    # connected line wins; narrow/wide rise and expected score are never inputs.
    for i in range(len(c)):
        for j in range(i+1,len(c)):
            if abs(c[j,0]-c[i,0])<.4*(x1-x0):continue
            slope=(c[j,1]-c[i,1])/(c[j,0]-c[i,0])
            if abs(slope)>.4:continue
            intercept=c[i,1]-slope*c[i,0];residual=np.abs(c[:,1]-slope*c[:,0]-intercept);chosen=[]
            for group in range(len(chunks)):
                ids=np.flatnonzero((c[:,3]==group)&(residual<=2.5))
                if len(ids):chosen.append(int(ids[np.argmax(c[ids,2])]))
            if len(chosen)<max(8,int(np.ceil(.7*len(chunks)))):continue
            s=c[chosen]
            if s[:,0].max()-s[:,0].min()<.7*(x1-x0):continue
            support=(len(chosen),float(np.minimum(s[:,2],60).sum()))
            if best is None or support>best[0]:best=(support,s)
    if best is None:return None,{'reason':'free-surface candidates do not form one sufficiently supported shared line','edge_candidate_count':len(candidates)}
    s=best[1];xx=[];yy=[]
    for x,y,amplitude,group in s:
        ch=chunks[int(group)];xx.append(x);yy.append(_interface(_profile(gray,ch),y))
    slope,intercept=np.polyfit(xx,yy,1);rms=float(np.sqrt(np.mean((np.asarray(yy)-slope*np.asarray(xx)-intercept)**2)))
    if rms>3:return None,{'reason':'shared free-surface edge is too inconsistent across columns','fit_rms_px':rms}
    out={'slope':float(slope),'intercept':float(intercept),'rms_px':rms,'support_columns':sum(len(chunks[int(i)]) for i in s[:,3]),'supported_column_groups':len(s),'column_groups':len(chunks),'reviewed_shared_reservoir_geometry':True,'surface_points_xy':list(map(list,zip(map(float,xx),map(float,yy))))}
    return out,{'surface':out}

def meniscus(frame,tube,geometry,surface):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float);a,b=tube['x_left'],tube['x_right'];width=b-a;pad=max(1,int(.08*width));cols=np.arange(int(np.ceil(a))+pad,int(np.floor(b))-pad+1)
    if len(cols)<4:return None,{'reason':'annotated tube interior is below grayscale interface resolution'}
    # A tube opening/rounded end is itself a strong horizontal optical edge.
    # Leave one bore width plus the derivative radius around those endpoints;
    # an interface merged into an opening is not independently resolved.
    middle=(a+b)/2;base=surface['slope']*middle+surface['intercept'];end_margin=max(12,int(np.ceil(width))+4)
    lo=max(2,int(tube['y_start'])+end_margin);hi=min(gray.shape[0]-2,int(tube['y_end'])-end_margin,int(base)+5)
    profile=_profile(gray,cols);response=_response(profile)
    outside=np.r_[np.arange(max(0,int(a-1.5*width)),max(0,int(a)-4)),np.arange(min(gray.shape[1],int(b)+5),min(gray.shape[1],int(b+1.5*width)))]
    for x0,y0,x1,y1 in geometry['tube_boxes']:outside=outside[(outside<x0-2)|(outside>x1+2)]
    external=_response(_profile(gray,outside)) if len(outside)>=4 else np.zeros(gray.shape[0])
    accepted=[];diagnostics=[]
    gradient=np.zeros_like(gray[:,cols]);gradient[2:-2]=np.abs(gray[4:,cols]-gray[:-4,cols])
    for y in _peaks(response,lo,hi,count=20,separation=7):
        # The interface fit reads +/-10 rows and the derivative +/-2. Exclude
        # candidates whose observation window touches a reviewed occluder.
        if any(x0-12<=middle<=x1+12 and y0-12<=y<=y1+12 for x0,y0,x1,y1 in geometry['exclude_boxes']):continue
        strength=float(response[y]);external_strength=float(np.max(external[max(0,y-3):y+4]));support=float(np.mean(np.max(gradient[max(0,y-3):y+4],axis=0)>=max(4.,.3*strength)))
        contrast_excess=strength-external_strength
        valid=support>=.7 and strength>=6 and contrast_excess>=4 and strength>=1.6*external_strength
        diagnostics.append({'proposal_y':y,'inner_contrast':strength,'outside_contrast':external_strength,'column_support':support,'accepted':bool(valid)})
        if valid:accepted.append((contrast_excess,y))
    if not accepted:return None,{'reason':'no meniscus edge distinguished from exterior/background edges','candidates':diagnostics}
    accepted.sort(reverse=True)
    if len(accepted)>1 and accepted[1][0]>.85*accepted[0][0] and abs(accepted[1][1]-accepted[0][1])>12:
        return None,{'reason':'multiple similarly supported separated meniscus candidates; no temporal/expected-order guessing','candidates':diagnostics}
    y=_interface(profile,accepted[0][1]);return y,{'y':y,'candidates':diagnostics,'selection':'greatest tube-local contrast above its exterior, independent of the other tube'}

def observe(frame,geometry,narrow,wide):
    surface,diagnostic=reservoir_surface(frame,geometry)
    if surface is None:return None,None,diagnostic
    levels={}
    for key,tube in (('narrow',narrow),('wide',wide)):
        present,identity=wall_identity(frame,tube,geometry);diagnostic[key+'_wall_identity']=identity
        if not present:
            diagnostic[key]={'reason':'tube-wall evidence no longer supports the reviewed first-frame position'}
            return surface,None,diagnostic
        level,detail=meniscus(frame,tube,geometry,surface);diagnostic[key]=detail
        if level is None:return surface,None,diagnostic
        levels[key]=level
    return surface,levels,diagnostic
