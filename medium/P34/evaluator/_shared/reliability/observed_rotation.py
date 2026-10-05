"""Observed in-plane texture rotation; no displacement/radius law is used."""
import cv2,numpy as np

def local_circle_observations(frames,proposals,radius):
    """Refine coarse locations using visible circular edges in each frame."""
    centers=np.full((len(frames),2),np.nan);radii=np.full(len(frames),np.nan)
    for i,(frame,p) in enumerate(zip(frames,proposals)):
        if not np.isfinite(p).all():continue
        x0,y0=np.maximum(0,np.floor(p-2*radius)).astype(int);x1,y1=np.minimum(frame.shape[1::-1],np.ceil(p+2*radius)).astype(int)
        crop=frame[y0:y1,x0:x1]
        if min(crop.shape[:2])<2*radius:continue
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);gray=cv2.GaussianBlur(gray,(3,3),.6)
        circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,max(8,radius*.6),param1=70,param2=12,minRadius=max(6,int(.8*radius)),maxRadius=int(1.3*radius))
        if circles is None:continue
        candidates=circles[0].astype(float);candidates[:,:2]+=[x0,y0]
        chosen=min(candidates,key=lambda x:np.linalg.norm(x[:2]-p)+.25*abs(x[2]-radius))
        if np.linalg.norm(chosen[:2]-p)>radius:continue
        centers[i]=chosen[:2];radii[i]=chosen[2]
    return centers,radii

def polar_texture_rotation(frames,centers,radii,min_correlation=.65):
    """Return independent adjacent-frame polar-texture registration evidence.

    Applicable only to stable visible in-plane texture. Unknown/changing texture
    stays missing. Angular direction and speed are not chosen from translation.
    """
    n=len(frames);radii=np.broadcast_to(np.asarray(radii,float),(n,))
    phase=np.full(n,np.nan);confidence=np.zeros(n);correlations=[]
    angles=np.arange(360)*np.pi/180;radial=np.linspace(.24,.87,30)
    previous=None;previous_index=None;previous_phase=None
    for i,(frame,center,radius) in enumerate(zip(frames,centers,radii)):
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius<8:continue
        x=center[0]+radius*radial[:,None]*np.cos(angles)[None,:]
        y=center[1]+radius*radial[:,None]*np.sin(angles)[None,:]
        if x.min()<0 or y.min()<0 or x.max()>=frame.shape[1]-1 or y.max()>=frame.shape[0]-1:continue
        patch=cv2.remap(frame.astype('float32'),x.astype('float32'),y.astype('float32'),cv2.INTER_LINEAR)
        patch-=patch.mean(axis=1,keepdims=True);energy=float(np.sum(patch*patch))
        if np.sqrt(energy/patch.size)<2.:continue
        if previous is None:
            phase[i]=0.;confidence[i]=1.;previous=patch;previous_index=i;previous_phase=0.;continue
        if i-previous_index>2:continue
        cross=np.fft.ifft(np.fft.fft(patch,axis=1)*np.conj(np.fft.fft(previous,axis=1)),axis=1).real.sum(axis=(0,2))
        norm=max(np.sqrt(energy*np.sum(previous*previous)),1e-12)
        signed=(np.arange(360)+180)%360-180;near=np.flatnonzero(np.abs(signed)<45)
        j=int(near[np.argmax(cross[near])]);corr=float(cross[j]/norm)
        # Repeated markings have equivalent peaks a full texture period
        # apart. Use the principal increment under the declared dense-time
        # sampling assumption, never choose a phase from translational speed.
        if float((cross.max()-cross[j])/norm)>.05:continue
        a,b,c=cross[(j-1)%360],cross[j],cross[(j+1)%360];denom=a-2*b+c
        sub=float(np.clip(.5*(a-c)/denom,-.5,.5)) if abs(denom)>1e-8 else 0.
        degrees=(j+sub+180)%360-180
        # Large frame-to-frame shifts cannot disambiguate repeated markings.
        if corr<min_correlation or abs(degrees)>90:continue
        phase[i]=previous_phase+np.radians(degrees);confidence[i]=corr
        correlations.append(corr);previous=patch;previous_index=i;previous_phase=phase[i]
    valid=np.isfinite(phase)
    return phase,confidence,{'method':'independent polar texture registration','valid_fraction':float(valid.mean()),
        'min_correlation':min_correlation,'median_correlation':float(np.median(correlations)) if correlations else None,
        'angular_span_rad':float(np.ptp(phase[valid])) if valid.any() else None,
        'assumptions':'stable in-plane texture; adjacent rotation below 45 degrees and below half the visible texture repetition period; no rolling-law prediction used'}


def planar_texture_rotation(frames,centers,radii,min_correlation=.65,feature_hue=None):
    """Measure rigid in-plane texture rotation with independently fitted translation.

    Patch registration can absorb subpixel centre jitter that otherwise aliases
    repeated markings. No displacement/radius relation enters the angle fit.
    Missing/textureless observations remain missing; gaps are not extrapolated.
    """
    n=len(frames);radii=np.broadcast_to(np.asarray(radii,float),(n,))
    phase=np.full(n,np.nan);confidence=np.zeros(n);previous=None;last=None;values=[]
    radius=float(np.nanmedian(radii));half=int(np.ceil(1.12*radius));size=2*half+1
    yy,xx=np.mgrid[:size,:size];mask=(((xx-half)**2+(yy-half)**2)<(.88*radius)**2).astype('uint8')*255
    angles=np.arange(180)*np.pi/90;radial=np.linspace(.25,.85,20)
    for i,(f,c,r) in enumerate(zip(frames,centers,radii)):
        if not np.isfinite(c).all() or not np.isfinite(r) or r<8:continue
        if min(c)<half or c[0]+half>=f.shape[1] or c[1]+half>=f.shape[0]:continue
        if feature_hue is None:
            gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype('float32')
        else:
            hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV).astype(float)
            distance=abs(hsv[:,:,0]-feature_hue);distance=np.minimum(distance,180-distance)
            gray=(255*np.exp(-.5*(distance/6.)**2)*np.clip((hsv[:,:,1]-50)/100,0,1)*np.clip((hsv[:,:,2]-40)/80,0,1)).astype('float32')
        x=c[0]+r*radial[:,None]*np.cos(angles);y=c[1]+r*radial[:,None]*np.sin(angles)
        polar=cv2.remap(gray,x.astype('float32'),y.astype('float32'),cv2.INTER_LINEAR)
        if float(np.sqrt(np.mean((polar-polar.mean(1,keepdims=True))**2)))<2:continue
        patch=cv2.getRectSubPix(gray,(size,size),tuple(map(float,c)))/255
        if previous is None:previous=patch;last=i;phase[i]=0.;confidence[i]=1.;continue
        if i-last>2:continue
        try:
            corr,w=cv2.findTransformECC(previous,patch,np.eye(2,3,dtype='float32'),cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,75,1e-5),mask,3)
        except cv2.error:continue
        angle=float(np.arctan2(w[1,0],w[0,0]));center_shift=w[:,:2]@np.array([half,half])+w[:,2]-[half,half]
        if corr<.90 or abs(angle)>np.pi/4 or np.linalg.norm(center_shift)>.25*r:continue
        phase[i]=phase[last]+angle;confidence[i]=corr;values.append(corr);previous=patch;last=i
    valid=np.isfinite(phase)
    diagnostic={'method':'independent Euclidean image registration with centre-jitter correction',
        'valid_fraction':float(valid.mean()),'min_correlation':.90,
        'median_correlation':float(np.median(values)) if values else None,
        'angular_span_rad':float(np.ptp(phase[valid])) if valid.any() else None,
        'assumptions':'visible stable in-plane texture; adjacent rotation below 45 degrees and half of any repeated texture period; no rolling-law prediction used'}
    if feature_hue is not None:
        diagnostic.update(method='independent registration of observed chromatic fiducials',feature_hue=float(feature_hue))
        return phase,confidence,diagnostic
    if valid.mean()>=.90:return phase,confidence,diagnostic
    fallback=polar_texture_rotation(frames,centers,radii,min_correlation)
    fallback[2]['rigid_registration_diagnostic']=diagnostic
    return fallback


def chromatic_fiducial_rotation(frames,centers,radii):
    """Separate saturated attached markings from achromatic specular highlights.

    Select a compact minority hue inside the initial object, using appearance
    only. No motion law, target angular speed or score enters feature selection.
    A stable planar-registration test is still required; otherwise use the
    original texture method and expose its diagnostic.
    """
    radii=np.broadcast_to(np.asarray(radii,float),(len(frames),))
    valid=np.flatnonzero(np.isfinite(centers).all(axis=1)&np.isfinite(radii))
    if len(valid):
        i=int(valid[0]);f=frames[i];c=centers[i];r=radii[i]
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV);yy,xx=np.mgrid[:f.shape[0],:f.shape[1]]
        roi=((xx-c[0])**2+(yy-c[1])**2)<(.87*r)**2
        saturated=roi&(hsv[:,:,1]>70)&(hsv[:,:,2]>60)
        if saturated.sum()>30:
            hist=np.bincount(hsv[:,:,0][saturated],minlength=180).astype(float)
            smoothed=sum(np.roll(hist,k) for k in range(-3,4));body=int(np.argmax(smoothed))
            d=abs(np.arange(180)-body);d=np.minimum(d,180-d);smoothed[d<20]=0
            hue=int(np.argmax(smoothed));distance=abs(hsv[:,:,0].astype(float)-hue);distance=np.minimum(distance,180-distance)
            mask=(saturated&(distance<9)).astype('uint8')
            count,labels,stats,cents=cv2.connectedComponentsWithStats(mask,8)
            components=[j for j in range(1,count) if 3<=stats[j,cv2.CC_STAT_AREA]<=.08*np.pi*r*r and min(stats[j,cv2.CC_STAT_WIDTH],stats[j,cv2.CC_STAT_HEIGHT])/max(stats[j,cv2.CC_STAT_WIDTH],stats[j,cv2.CC_STAT_HEIGHT])>.45]
            area=sum(stats[j,cv2.CC_STAT_AREA] for j in components)
            if len(components)>=2 and .005*np.pi*r*r<=area<=.2*np.pi*r*r:
                phase,confidence,diagnostic=planar_texture_rotation(frames,centers,radii,feature_hue=hue)
                diagnostic['initial_fiducial_components']=len(components)
                if diagnostic['valid_fraction']>=.95:return phase,confidence,diagnostic
    return planar_texture_rotation(frames,centers,radii)
