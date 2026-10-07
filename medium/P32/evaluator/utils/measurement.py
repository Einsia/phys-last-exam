"""Independent red-pole tracking, circular-dial calibration and M1 statistics."""
import math
import numpy as np
import cv2
from .common import ExtractionError


def _contours(binary):
    return cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


def circular_calibration(ellipse, pivot):
    """Metric rectification of a circle whose physical centre is independently observed.

    A conic alone is insufficient for general perspective rectification. Its polar
    of the projected physical centre is the plane's vanishing line. Rectify that
    line, then whiten the resulting centred ellipse. The remaining rotation is
    constant and does not change signed angular differences.
    """
    (cx, cy), (d1, d2), angle = ellipse
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    quadratic = rotation @ np.diag([4 / d1**2, 4 / d2**2]) @ rotation.T
    offset = np.array([cx, cy]) - pivot
    conic = np.zeros((3, 3))
    conic[:2, :2] = quadratic
    conic[:2, 2] = conic[2, :2] = -quadratic @ offset
    conic[2, 2] = offset @ quadratic @ offset - 1
    projective = np.eye(3)
    projective[2, :2] = conic[:2, 2] / conic[2, 2]
    inv = np.linalg.inv(projective)
    affine_conic = inv.T @ conic @ inv
    metric = affine_conic[:2, :2] / -affine_conic[2, 2]
    values, vectors = np.linalg.eigh(metric)
    if np.any(values <= 0):
        raise ExtractionError('Invalid circle/physical-centre calibration')
    whitening = np.eye(3)
    whitening[:2, :2] = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    translate = np.eye(3)
    translate[:2, 2] = -pivot
    return whitening @ projective @ translate


def transform_points(points, homography):
    points = np.atleast_2d(points)
    q = np.column_stack([points, np.ones(len(points))]) @ homography.T
    if np.any(np.abs(q[:, 2]) < 1e-8):
        raise ExtractionError('Point on rectification horizon')
    return q[:, :2] / q[:, 2, None]


def _dial_geometry(frame, mask, cfg, muted_hub=False, pivot_hint=None):
    ys, xs = np.where(mask)
    if len(xs) < 100:
        raise ExtractionError('Compass mask missing or too small')
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < cfg.dial_white_saturation_max) & (hsv[..., 2] > cfg.dial_white_value_min)
             & mask[y0:y1, x0:x1]).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours = sorted(_contours(white), key=cv2.contourArea, reverse=True)
    if not contours or len(contours[0]) < 30:
        raise ExtractionError('No reliable dial boundary')
    # Tick marks cut inward from the white rim. The convex hull recovers the
    # outside dial boundary without fitting those dark, radial indentations.
    contour = cv2.convexHull(contours[0]).astype(np.float32)
    if len(contour) < 12:
        raise ExtractionError('Insufficient distributed points on the dial rim')
    ellipse_local = cv2.fitEllipse(contour)
    (cx, cy), diameters, orientation = ellipse_local
    if min(diameters) < 30 or min(diameters)/max(diameters) < .60:
        raise ExtractionError('Dial too small or strongly foreshortened')
    rotation_angle = math.radians(orientation)
    rotation = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                         [math.sin(rotation_angle), math.cos(rotation_angle)]])
    points = (contour[:, 0] - [cx, cy]) @ rotation
    radial_error = np.abs(np.sqrt(np.sum((points / (np.array(diameters)/2))**2, axis=1))-1)
    residual = float(np.quantile(radial_error, .95))
    radius = min(diameters)/2
    boundary_tolerance=max(cfg.max_ellipse_residual,cfg.boundary_localization_px/radius)
    if residual > boundary_tolerance:
        raise ExtractionError(f'Irregular dial boundary (relative residual={residual:.4f})')
    yy, xx = np.indices(white.shape)
    central = (xx-cx)**2 + (yy-cy)**2 < (.26*radius)**2
    # The gold/brass pivot is a separate visual landmark, not the box centre.
    if pivot_hint is None:
        gold = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 45) & (hsv[..., 1] >= (30 if muted_hub else 65)) & (hsv[..., 2] >= 50) & central
        if muted_hub:
            gold=cv2.morphologyEx(gold.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
        gold_contours = []
        for candidate in _contours(gold):
            if muted_hub:candidate=cv2.convexHull(candidate)
            area=cv2.contourArea(candidate);perimeter=cv2.arcLength(candidate,True)
            _,_,bw,bh=cv2.boundingRect(candidate)
            # A few codec-colored pixels from the needle are not a physical hub.
            if (max(4.,.0015*math.pi*radius**2) if muted_hub else max(8.,.003*math.pi*radius**2))<=area<.08*math.pi*radius**2 and min(bw,bh)/max(bw,bh)>(.50 if muted_hub else .65) and 4*math.pi*area/max(perimeter**2,1)>.60:
                gold_contours.append(candidate)
        gold_contours.sort(key=cv2.contourArea,reverse=True)
        if not gold_contours or cv2.contourArea(gold_contours[0]) < 4:
            # A pivot need not be brass colored. A resolved compact dark hub is
            # an independent physical-center landmark too; the dial center alone
            # is never substituted for a missing pivot.
            dark=(hsv[...,2]<85)&central
            hubs=[]
            for candidate in _contours(dark):
                area=cv2.contourArea(candidate);perimeter=cv2.arcLength(candidate,True)
                x,y,bw,bh=cv2.boundingRect(candidate)
                if 4<=area<.08*math.pi*radius**2 and min(bw,bh)/max(bw,bh)>.6 and 4*math.pi*area/max(perimeter**2,1)>.55:
                    hubs.append(candidate)
            if len(hubs)!=1:raise ExtractionError('Cannot independently locate a unique physical pivot')
            gold_contours=hubs
        g = gold_contours[0]
        moment = cv2.moments(g)
        pivot = np.array([moment['m10']/moment['m00']+x0, moment['m01']/moment['m00']+y0])
        pivot_area = cv2.contourArea(g)
    else:
        # Hint is an independently matched image patch, never the dial centre.
        pivot=np.asarray(pivot_hint,dtype=float)
        pivot_area=0.0
    if pivot_area > .08 * math.pi*radius**2:
        raise ExtractionError('Pivot candidate too large')
    centre = np.array([cx+x0, cy+y0])
    if np.linalg.norm(pivot-centre) / radius > cfg.max_pivot_offset_fraction:
        raise ExtractionError('Pivot inconsistent with dial geometry')
    ellipse = (tuple(centre), diameters, orientation)
    H = circular_calibration(ellipse, pivot)
    # Contour round-trip verifies the calibration is numerically well-conditioned.
    circle_points = transform_points(contour[:, 0] + [x0, y0], H)
    circle_residual = float(np.quantile(np.abs(np.linalg.norm(circle_points, axis=1)-1), .95))
    if circle_residual > 2*boundary_tolerance:
        raise ExtractionError('Circle rectification failed boundary validation')
    return {'pivot': pivot, 'radius': radius, 'ellipse': ellipse, 'H': H,
            'ellipse_residual': residual, 'rectified_circle_residual': circle_residual,
            'boundary_tolerance':boundary_tolerance,
            'boundary_localization_px':cfg.boundary_localization_px,
            'pivot_area': pivot_area}


def dial_geometry(frame, mask, cfg, pivot_hint=None):
    """Preserve resolved hubs; use low-chroma contour recovery only on failure."""
    if pivot_hint is not None:
        return _dial_geometry(frame,mask,cfg,pivot_hint=pivot_hint)
    try:
        return _dial_geometry(frame,mask,cfg,muted_hub=False)
    except ExtractionError as original:
        try:
            result=_dial_geometry(frame,mask,cfg,muted_hub=True)
            result['hub_method']='low_chroma_observed_contour_fallback'
            return result
        except ExtractionError:
            raise original


def _rectified_red_pole(frame, mask, geom, cfg, prediction=None):
    pivot, radius = geom['pivot'], geom['radius']
    reach=.6*max(geom['ellipse'][1])+float(np.linalg.norm(pivot-np.asarray(geom['ellipse'][0])))
    x0, y0 = np.maximum(0, (pivot-reach).astype(int))
    x1, y1 = np.minimum(frame.shape[1::-1], (pivot+reach+1).astype(int))
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(hsv.shape[:2])
    plane_origin=transform_points([pivot],geom['H'])[0]
    pixel_points=np.column_stack([(xx+x0).ravel(),(yy+y0).ravel()])
    plane_points=transform_points(pixel_points,geom['H'])-plane_origin
    dist=np.linalg.norm(plane_points,axis=1).reshape(xx.shape)
    # SAM may segment the white dial while excluding its colored needle.
    # Fill enclosed holes in the dial proposal; retain its observed boundary.
    solid=np.zeros_like(mask,dtype='uint8')
    cv2.drawContours(solid,_contours(mask),-1,1,cv2.FILLED)
    allowed=(hsv[...,1]>=85)&(hsv[...,2]>=55)&(dist>.10)&(dist<1.05)&solid[y0:y1,x0:x1].astype(bool)
    seed_hue=geom.get('pole_hue')
    if seed_hue is None:
        # Prefer the conventional red pole when present; otherwise fix one
        # visibly distinct colored half at frame zero. Signed angular changes
        # are invariant to choosing the opposite end of a rigid needle.
        traditional=allowed&((hsv[...,0]<=20)|(hsv[...,0]>=170))
        pool=traditional if traditional.sum()>=cfg.min_red_pixels else allowed
        if pool.sum()<cfg.min_red_pixels:raise ExtractionError('No independently visible colored needle pole')
        hist=np.bincount(hsv[...,0][pool],minlength=180);seed_hue=float(np.argmax(hist));geom['pole_hue']=seed_hue
    delta=np.abs(hsv[...,0].astype(float)-seed_hue);delta=np.minimum(delta,180-delta)
    red=allowed&(delta<12)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < cfg.min_red_pixels:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs+x0, ys+y0]).astype(np.float32)
        plane=transform_points(points,geom['H'])-plane_origin
        vx, vy, lx, ly = cv2.fitLine(plane.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel()
        unit = np.array([vx, vy])
        if np.dot(unit, plane.mean(axis=0)) < 0:
            unit = -unit
        normal = np.array([-unit[1], unit[0]])
        # A real compass pole can be a filled triangle. Its width is not a
        # line-fitting error: measure the midpoint of the two visible sides
        # in radial strips, then fit that observed centreline.
        projections=plane@unit
        edges=np.linspace(float(np.min(projections)),float(np.max(projections)),13)
        middle=[]
        for low,high in zip(edges[:-1],edges[1:]):
            band=plane[(projections>=low)&(projections<=high)]
            if len(band)<2:continue
            along=float(np.median(band@unit));across=band@normal
            center=.5*(float(np.quantile(across,.1))+float(np.quantile(across,.9)))
            middle.append(unit*along+normal*center)
        if len(middle)<4:continue
        middle=np.asarray(middle,np.float32)
        vx,vy,lx,ly=cv2.fitLine(middle,cv2.DIST_HUBER,0,.001,.001).ravel()
        unit=np.array([vx,vy]);unit*=1 if np.dot(unit,plane.mean(axis=0))>=0 else -1
        normal=np.array([-unit[1],unit[0]])
        line_distance=abs(float(np.dot([lx,ly],normal)))*radius
        residual=float(np.sqrt(np.mean(((middle-[lx,ly])@normal)**2)))*radius
        projections=plane@unit
        length = float(np.quantile(projections, .99))
        if not .4 < length < 1.1 or line_distance > cfg.max_needle_line_residual_px:
            continue
        if residual > cfg.max_needle_line_residual_px or np.ptp(projections) < .30:
            continue
        tip = transform_points([plane_origin+unit*length],np.linalg.inv(geom['H']))[0]
        # Colour determines pole identity. Flow only disambiguates multiple red candidates.
        flow_distance = float(np.linalg.norm(tip-prediction)) if prediction is not None else None
        candidates.append({'tip': tip, 'red_pixels': len(points), 'line_residual_px': residual,
                           'pivot_line_distance_px': float(line_distance), 'flow_distance_px': flow_distance,
                           'pole_length_dial_radius':length,'centreline_points':len(middle),
                           'method':'visible pole side-midpoints in independently rectified dial plane'})
    if not candidates:
        raise ExtractionError('No reliable red pole attached to the pivot')
    if len(candidates) > 1:
        if prediction is None:
            raise ExtractionError('Ambiguous red-pole candidates')
        candidates = [c for c in candidates if c['flow_distance_px'] < .20*radius]
        if len(candidates) != 1:
            raise ExtractionError('Optical flow cannot resolve red-pole ambiguity')
    obs = candidates[0]
    # Use the reference calibration; freeze its arbitrary in-plane rotation.
    rectified = transform_points([pivot, obs['tip']], geom['H'])
    dx, dy = rectified[1]-rectified[0]
    obs['angle_deg'] = math.degrees(math.atan2(-dy, dx))
    obs['raw_angle_deg'] = math.degrees(math.atan2(-(obs['tip'][1]-pivot[1]), obs['tip'][0]-pivot[0]))
    return obs


def _image_red_pole(frame, mask, geom, cfg, prediction=None):
    pivot, radius = geom['pivot'], geom['radius']
    x0, y0 = np.maximum(0, (pivot-1.15*radius).astype(int))
    x1, y1 = np.minimum(frame.shape[1::-1], (pivot+1.15*radius+1).astype(int))
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(hsv.shape[:2])
    dist = np.hypot(xx+x0-pivot[0], yy+y0-pivot[1])
    # SAM may segment the white dial while excluding its colored needle.
    # Fill enclosed holes in the dial proposal; retain its observed boundary.
    solid=np.zeros_like(mask,dtype='uint8')
    cv2.drawContours(solid,_contours(mask),-1,1,cv2.FILLED)
    allowed=(hsv[...,1]>=85)&(hsv[...,2]>=55)&(dist>.10*radius)&(dist<1.05*radius)&solid[y0:y1,x0:x1].astype(bool)
    seed_hue=geom.get('pole_hue')
    if seed_hue is None:
        # Prefer the conventional red pole when present; otherwise fix one
        # visibly distinct colored half at frame zero. Signed angular changes
        # are invariant to choosing the opposite end of a rigid needle.
        traditional=allowed&((hsv[...,0]<=20)|(hsv[...,0]>=170))
        pool=traditional if traditional.sum()>=cfg.min_red_pixels else allowed
        if pool.sum()<cfg.min_red_pixels:raise ExtractionError('No independently visible colored needle pole')
        hist=np.bincount(hsv[...,0][pool],minlength=180);seed_hue=float(np.argmax(hist));geom['pole_hue']=seed_hue
    delta=np.abs(hsv[...,0].astype(float)-seed_hue);delta=np.minimum(delta,180-delta)
    red=allowed&(delta<12)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < cfg.min_red_pixels:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs+x0, ys+y0]).astype(np.float32)
        vx, vy, lx, ly = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
        unit = np.array([vx, vy])
        if np.dot(unit, points.mean(axis=0)-pivot) < 0:
            unit = -unit
        normal = np.array([-unit[1], unit[0]])
        line_distance = abs(np.dot(pivot-[lx, ly], normal))
        residual = float(np.sqrt(np.mean(((points-[lx, ly]) @ normal)**2)))
        projections = (points-pivot) @ unit
        length = float(np.quantile(projections, .99))
        if not .4*radius < length < 1.1*radius or line_distance > cfg.max_needle_line_residual_px:
            continue
        if residual > cfg.max_needle_line_residual_px or np.ptp(projections) < .30*radius:
            continue
        tip = pivot + unit*length
        # Colour determines pole identity. Flow only disambiguates multiple red candidates.
        flow_distance = float(np.linalg.norm(tip-prediction)) if prediction is not None else None
        candidates.append({'tip': tip, 'red_pixels': len(points), 'line_residual_px': residual,
                           'pivot_line_distance_px': float(line_distance), 'flow_distance_px': flow_distance})
    if not candidates:
        raise ExtractionError('No reliable red pole attached to the pivot')
    if len(candidates) > 1:
        if prediction is None:
            raise ExtractionError('Ambiguous red-pole candidates')
        candidates = [c for c in candidates if c['flow_distance_px'] < .20*radius]
        if len(candidates) != 1:
            raise ExtractionError('Optical flow cannot resolve red-pole ambiguity')
    obs = candidates[0]
    # Use the reference calibration; freeze its arbitrary in-plane rotation.
    rectified = transform_points([pivot, obs['tip']], geom['H'])
    dx, dy = rectified[1]-rectified[0]
    obs['angle_deg'] = math.degrees(math.atan2(-dy, dx))
    obs['raw_angle_deg'] = math.degrees(math.atan2(-(obs['tip'][1]-pivot[1]), obs['tip'][0]-pivot[0]))
    return obs


def red_pole(frame, mask, geom, cfg, prediction=None):
    """Use the original resolved pole; retry wide/projected poles in dial space."""
    try:
        observed = _image_red_pole(frame, mask, geom, cfg, prediction)
        observed['method'] = 'resolved image-plane pole'
        return observed
    except ExtractionError as original:
        try:
            return _rectified_red_pole(frame, mask, geom, cfg, prediction)
        except ExtractionError:
            raise original


def match_static(reference, current, exclusion=None):
    """Independent visual check of camera motion, without moving-needle features."""
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    allowed_pixels=np.ones(ref.shape,bool) if exclusion is None else ~exclusion
    gradient=cv2.magnitude(cv2.Sobel(ref,cv2.CV_32F,1,0),cv2.Sobel(ref,cv2.CV_32F,0,1))
    salient=allowed_pixels&(gradient>40)
    difference=np.abs(cur.astype(float)-ref.astype(float))
    if salient.sum()>=100 and np.mean(difference[allowed_pixels])<1.5 and np.percentile(difference[salient],95)<8:
        # Direct alignment of resolved static edges can verify a fixed view
        # when repeated background markers make ORB matches ambiguous.
        return {'reliable':True,'method':'static edge pixel alignment','inliers':int(salient.sum()),
                'reprojection_error_px':0.,'static_pixel_difference_p95':float(np.percentile(difference[salient],95)),
                'rotation_deg':0.,'scale':1.,'shift_px':0.,'matrix':[[1.,0.,0.],[0.,1.,0.]]}
    orb = cv2.ORB_create(nfeatures=2000)
    allowed = None if exclusion is None else (~exclusion).astype(np.uint8)*255
    kp1, desc1 = orb.detectAndCompute(ref, allowed)
    kp2, desc2 = orb.detectAndCompute(cur, allowed)
    if desc1 is None or desc2 is None:
        return {'reliable': False, 'reason': 'No static feature descriptors'}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance]
    if len(good) < 15:
        return {'reliable': False, 'reason': 'Too few static feature matches', 'matches': len(good)}
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or int(inliers.sum()) < 15:
        return {'reliable': False, 'reason': 'Static registration failed'}
    error = np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)
    good_error = float(np.median(error[inliers.ravel().astype(bool)]))
    return {'reliable': good_error < 1.5, 'inliers': int(inliers.sum()),
            'reprojection_error_px': good_error,
            'rotation_deg': math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])),
            'scale': float(np.hypot(matrix[0, 0], matrix[1, 0])),
            'shift_px': float(np.linalg.norm(matrix[:, 2])), 'matrix': matrix.tolist()}


def matched_pivot(reference, current, geom, camera_matrix):
    """Redetect the initially observed physical hub by its local appearance."""
    pivot=np.asarray(geom['pivot']);radius=max(4,int(round(math.sqrt(geom['pivot_area']/math.pi)))+2)
    cx,cy=np.rint(pivot).astype(int)
    template=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    if template.shape!=(2*radius+1,2*radius+1) or float(np.std(template))<8:
        raise ExtractionError('Physical hub template has insufficient observed texture')
    aligned=cv2.warpAffine(current,np.linalg.inv(camera_matrix)[:2],reference.shape[1::-1])
    gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
    search=max(3,int(round(.10*geom['radius'])))
    x0,y0=cx-radius-search,cy-radius-search
    region=gray[max(0,y0):cy+radius+search+1,max(0,x0):cx+radius+search+1]
    if x0<0 or y0<0 or min(region.shape)<2*radius+1:
        raise ExtractionError('Physical hub search is outside the visible video')
    response=cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,location=cv2.minMaxLoc(response)
    if not math.isfinite(score) or score<.70:
        # A rotating needle changes the small grayscale patch surrounding the
        # hub. Reobserve its compact brass-colored material locally instead
        # of requiring the neighboring needle to retain its original pose.
        hsv=cv2.cvtColor(aligned,cv2.COLOR_BGR2HSV)
        lo_x,hi_x=max(0,cx-search),min(hsv.shape[1],cx+search+1)
        lo_y,hi_y=max(0,cy-search),min(hsv.shape[0],cy+search+1)
        crop=hsv[lo_y:hi_y,lo_x:hi_x]
        gold=((crop[...,0]>=10)&(crop[...,0]<=45)&(crop[...,1]>=30)&(crop[...,2]>=45)).astype(np.uint8)
        components=[]
        for contour in _contours(gold):
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            x,y,w,h=cv2.boundingRect(contour)
            if area<8 or min(w,h)/max(w,h)<.55 or 4*math.pi*area/max(perimeter**2,1)<.50:continue
            moments=cv2.moments(contour)
            point=np.array([lo_x+moments['m10']/moments['m00'],lo_y+moments['m01']/moments['m00']])
            if np.linalg.norm(point-pivot)<=max(3.,.035*geom['radius']):components.append(point)
        if len(components)!=1:raise ExtractionError('Physical hub appearance correspondence is unresolved')
        return transform_points([components[0]],camera_matrix)[0],None
    observed=[x0+location[0]+radius,y0+location[1]+radius]
    return transform_points([observed],camera_matrix)[0],float(score)


def track_needles_legacy(frames, times, masks, cfg, initial_pivots=None):
    calibrations, calibration_errors = [], []
    for j in range(2):
        try:
            pivot=initial_pivots[j] if initial_pivots is not None else None
            calibration=dial_geometry(frames[0], masks[0, j], cfg,pivot_hint=pivot)
            if pivot is not None:calibration['hub_method']='reviewed physical hub in actual video frame zero'
            red_pole(frames[0], masks[0,j],calibration,cfg)
            calibrations.append(calibration)
            calibration_errors.append(None)
        except ExtractionError as e:
            calibrations.append(None)
            calibration_errors.append(str(e))
    rows = []
    previous_gray, previous_tips = None, [None, None]
    exclusion = cv2.dilate(np.any(masks[0], axis=0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    # The conductor/current indicator between the compasses can change
    # appearance when energized. Camera evidence comes from outside the
    # observed apparatus envelope, not from its electrically active center.
    ys,xs=np.where(np.any(masks[0],axis=0))
    if len(xs):
        pad=max(16,int(.5*(ys.max()-ys.min()+1)))
        exclusion[max(0,ys.min()-pad):min(exclusion.shape[0],ys.max()+pad+1),max(0,xs.min()-pad):min(exclusion.shape[1],xs.max()+pad+1)]=True
    reference_area = np.sum(masks[0], axis=(1, 2))
    for i, (frame, t) in enumerate(zip(frames, times)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera = match_static(frames[0], frame, exclusion)
        camera_ok = (camera['reliable'] and abs(camera['rotation_deg']) <= cfg.max_camera_rotation_deg
                     and camera['shift_px'] <= cfg.max_camera_shift_px and abs(camera['scale']-1) <= .005)
        row = {'frame_index': i, 'time_sec': float(t), 'camera': camera, 'left': {}, 'right': {}}
        overlap = np.count_nonzero(masks[i, 0] & masks[i, 1]) / max(1, min(np.sum(masks[i], axis=(1, 2))))
        for j, name in enumerate(('left', 'right')):
            try:
                geom = calibrations[j]
                if geom is None:
                    raise ExtractionError(calibration_errors[j])
                if not camera_ok:
                    raise ExtractionError('Camera motion exceeds calibrated-frame bounds or registration is unreliable')
                if overlap > .05:
                    raise ExtractionError('Compass masks merged or overlapped')
                area_ratio = masks[i, j].sum()/max(reference_area[j], 1)
                if not .65 < area_ratio < 1.4:
                    raise ExtractionError('Compass mask area changed excessively')
                camera_matrix=np.eye(3);camera_matrix[:2]=np.asarray(camera['matrix'])
                expected_pivot=transform_points([geom['pivot']],camera_matrix)[0]
                pivot_tolerance=max(cfg.max_pivot_drift_fraction*geom['radius'],cfg.boundary_localization_px)
                try:
                    observed = dial_geometry(frame, masks[i, j], cfg)
                    if np.linalg.norm(observed['pivot']-expected_pivot) > pivot_tolerance:
                        raise ExtractionError('Physical pivot moved or compass identity switched')
                    pivot_method='observed hub contour';pivot_match=None
                except ExtractionError:
                    point,pivot_match=matched_pivot(frames[0],frame,geom,camera_matrix)
                    if np.linalg.norm(point-expected_pivot)>pivot_tolerance:
                        raise ExtractionError('Observed physical hub moved relative to the dial')
                    observed=dial_geometry(frame,masks[i,j],cfg,pivot_hint=point)
                    pivot_method='observed frame-zero hub appearance correspondence'
                prediction = None
                if previous_gray is not None and previous_tips[j] is not None:
                    origin = np.float32(previous_tips[j]).reshape(1, 1, 2)
                    dest, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, origin, None, winSize=(21,21), maxLevel=2)
                    if status is not None and status[0, 0]:
                        back, valid, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, dest, None, winSize=(21,21), maxLevel=2)
                        if valid[0, 0] and np.linalg.norm(back-origin) < 1.5:
                            prediction = dest[0, 0]
                current_geom = {**geom, 'pivot': observed['pivot'],
                                'H':geom['H']@np.linalg.inv(camera_matrix),
                                'ellipse':observed['ellipse']}
                obs = red_pole(frame, masks[i, j], current_geom, cfg, prediction)
                row[name] = {'valid': True, 'reason': None, 'pivot': observed['pivot'],
                             'pivot_method':pivot_method,'pivot_match_score':pivot_match,**obs}
                previous_tips[j] = obs['tip']
            except ExtractionError as e:
                row[name] = {'valid': False, 'reason': str(e), 'angle_deg': None, 'raw_angle_deg': None,
                             'pivot': None, 'tip': None}
                previous_tips[j] = None
        rows.append(row)
        previous_gray = gray
    return rows, calibrations, calibration_errors


def track_needles(frames, times, masks, cfg, initial_pivots=None):
    """Measure the red poles independently in the final five decoded frames."""
    rows=[];calibrations=[None,None];errors=[None,None]
    start=max(0,len(frames)-5)
    for i,(frame,t) in enumerate(zip(frames,times)):
        row={'frame_index':i,'time_sec':float(t),'camera':{'policy':'final_orientation_only'}}
        for j,name in enumerate(('left','right')):
            obs={'valid':False,'reason':'outside_final_five_frames','angle_deg':None,'raw_angle_deg':None,'pivot':None,'tip':None}
            if i>=start:
                try:
                    geom=dial_geometry(frame,masks[i,j],cfg)
                    pole=red_pole(frame,masks[i,j],geom,cfg)
                    # Use a shared image-coordinate basis to compare directions;
                    # separately rectified dial coordinates can differ by a rotation.
                    obs=dict(pole,valid=True,reason=None,pivot=geom['pivot'],angle_deg=pole['raw_angle_deg'])
                    calibrations[j]=geom
                except ExtractionError as exc:
                    obs['reason']=str(exc);errors[j]=str(exc)
            row[name]=obs
        rows.append(row)
    return rows,calibrations,errors


def unwrap_track(times, angles, cfg):
    """Do not bridge gaps whose direction/turn count cannot be observed."""
    result = np.full(len(angles), np.nan)
    valid = np.flatnonzero(np.isfinite(angles))
    if len(valid) < 3 or len(valid)/len(angles) < cfg.min_valid_fraction:
        raise ExtractionError('Insufficient valid needle observations')
    if times[valid[0]]-times[0] > cfg.max_gap_sec or times[-1]-times[valid[-1]] > cfg.max_gap_sec:
        raise ExtractionError('Needle track does not cover start/end')
    result[valid[0]] = angles[valid[0]]
    for previous, current in zip(valid[:-1], valid[1:]):
        dt = times[current]-times[previous]
        if dt > cfg.max_gap_sec + 1e-9 or cfg.max_speed_deg_sec * dt >= 180:
            raise ExtractionError('Unobserved gap makes angle unwrapping ambiguous')
        step = (angles[current]-angles[previous]+180) % 360 - 180
        if abs(step) > cfg.max_step_deg or abs(step)/dt > cfg.max_speed_deg_sec:
            raise ExtractionError('Discontinuous angle: possible pole switch or tracking failure')
        result[current] = result[previous]+step
    return result


def measure_image_reference(image, masks, calibrations, first_row, cfg):
    """Measure the registered supplied still, independently checking both red poles.

    The caller must first validate static scene correspondence and warp the image
    into the video's reference coordinates. Static matching alone is insufficient:
    a different time point of the same apparatus can match the background.
    """
    result = {'accepted': False, 'needles': {}, 'reason': None}
    try:
        for j, name in enumerate(('left', 'right')):
            geometry = calibrations[j]
            first = first_row[name]
            if geometry is None or not first['valid']:
                raise ExtractionError(f'{name}: first video frame has no reliable red-pole observation')
            observed = dial_geometry(image, masks[j], cfg)
            drift = float(np.linalg.norm(observed['pivot']-geometry['pivot']))
            if drift > cfg.max_pivot_drift_fraction*geometry['radius']:
                raise ExtractionError(f'{name}: supplied still pivot does not align with video')
            obs = red_pole(image, masks[j], {**geometry, 'pivot': observed['pivot']}, cfg)
            difference = (obs['angle_deg']-first['angle_deg']+180) % 360-180
            result['needles'][name] = {**obs, 'pivot': observed['pivot'],
                                     'difference_from_video_start_deg': float(difference)}
            if abs(difference) > cfg.initial_reference_tolerance_deg:
                raise ExtractionError(f'{name}: supplied still needle direction differs from video start')
        result['accepted'] = True
    except ExtractionError as e:
        result['reason'] = str(e)
    return result


def initial_reference(rows, times, tracks, cfg, image_reference=None):
    """Define the observed initial direction without requiring a stationary pre-roll."""
    details = {'source': 'video_first_frame', 'frame_index': 0, 'time_sec': float(times[0]),
               'stationarity_required': False, 'needles': {},
               'image_reference_check': image_reference}
    use_image = image_reference is not None and image_reference.get('accepted') is True
    for j, name in enumerate(('left', 'right')):
        if not rows[0][name]['valid'] or not np.isfinite(tracks[j][0]):
            raise ExtractionError(f'{name}: no reliable initial red-pole observation in video frame 0')
        angle = float(tracks[j][0])
        if use_image:
            measured = float(image_reference['needles'][name]['angle_deg'])
            difference = (measured-angle+180) % 360-180
            if not np.isfinite(measured) or abs(difference) > cfg.initial_reference_tolerance_deg:
                raise ExtractionError(f'{name}: supplied still cannot join the observed angle trajectory')
            # Put the independently observed still angle on the video's unwrap branch.
            angle += difference
        details['needles'][name] = {'angle_deg': angle, 'valid': True}
    if use_image:
        details['source'] = 'matched_first_frame_image'
        details['image_path'] = image_reference.get('image_path')
    return details


def final_stable_window(times, angle_tracks, cfg):
    if times[-1]-times[0] < cfg.stable_window_sec:
        raise ExtractionError('Video too short to establish a final stable window')
    start = max(0, int(np.searchsorted(times, times[-1]-cfg.stable_window_sec, side='right'))-1)
    indices = np.arange(start, len(times))
    details = {'start_time_sec': float(times[indices[0]]), 'end_time_sec': float(times[indices[-1]]),
               'start_frame': int(indices[0]), 'end_frame': int(indices[-1]), 'needles': {}}
    ok = True
    for name, track in zip(('left', 'right'), angle_tracks):
        valid_idx = indices[np.isfinite(track[indices])]
        vals = track[valid_idx]
        span = float(times[valid_idx[-1]]-times[valid_idx[0]]) if len(vals) else 0
        angle_range = float(np.ptp(vals)) if len(vals) else None
        valid = (len(vals) >= 3 and len(vals)/len(indices) >= cfg.min_valid_fraction
                 and span >= cfg.stable_window_sec-1e-9 and angle_range <= cfg.stable_angle_range_deg)
        details['needles'][name] = {'valid_frames': len(vals), 'valid_duration_sec': span,
            'angle_range_deg': angle_range, 'median_deg': float(np.median(vals)) if len(vals) else None,
            'mad_deg': float(np.median(np.abs(vals-np.median(vals)))) if len(vals) else None, 'stable': bool(valid)}
        ok &= valid
    details['stable'] = bool(ok)
    return details


def decide(delta1, delta2, cfg):
    """Continuous score; threshold conditions are explanatory diagnostics only."""
    if not math.isfinite(delta1) or not math.isfinite(delta2):
        raise ExtractionError('Cannot score non-finite deflections')
    tolerance = max(cfg.symmetry_abs_tol_deg, cfg.symmetry_rel_tol * max(abs(delta1), abs(delta2)))
    conditions = {'left_min_deflection': abs(delta1) >= cfg.min_deflection_deg,
                  'right_min_deflection': abs(delta2) >= cfg.min_deflection_deg,
                  'opposite_direction': delta1*delta2 < 0,
                  'symmetric_magnitude': abs(delta1+delta2) <= tolerance}
    # Normalize by observed motion rather than converting tolerance checks to 0/1.
    scale = max(abs(delta1), abs(delta2))
    if scale == 0:
        symmetry_error, symmetry_score, motion_score = 1.0, 0.0, 0.0
    else:
        left, right = delta1/scale, delta2/scale
        symmetry_error = min(1.0, max(0.0, abs(left+right)/(abs(left)+abs(right))))
        symmetry_score = 1.0-symmetry_error
        motion_score = min(1.0, min(abs(delta1),abs(delta2))/cfg.min_deflection_deg)
    metric = float(min(1.0,max(0.0,symmetry_score*motion_score)))
    details = {
        'version': 'signed_deflection_balance_v1', 'range': [0.0,1.0], 'higher_is_better': True,
        'formula': '(1 - abs(delta_1 + delta_2) / (abs(delta_1) + abs(delta_2))) * min(1, min(abs(delta_1), abs(delta_2)) / min_deflection_deg)',
        'zero_motion_policy': 'If both deflections are zero, the score is 0.0.',
        'normalized_symmetry_error': float(symmetry_error), 'symmetry_score': float(symmetry_score),
        'motion_score': float(motion_score)}
    return metric, conditions, tolerance, details


def summarize(rows, times, cfg, verbose, image_reference=None):
    start=max(0,len(rows)-5)
    indices=[i for i in range(start,len(rows)) if all(rows[i][k]['valid'] for k in ('left','right'))]
    if len(indices)<2:
        raise ExtractionError('Need at least two jointly readable final-frame compass directions')
    differences=np.array([abs((rows[i]['left']['angle_deg']-rows[i]['right']['angle_deg']+180)%360-180) for i in indices])
    separation=float(np.median(differences))
    score=float((1-np.cos(np.radians(separation)))/2)
    for row in rows:
        for name in ('left','right'):row[name]['unwrapped_angle_deg']=row[name].get('angle_deg')
    verbose['principle']="Score only the angle between final magnetic-pole directions in the last 5 frames: opposite at 180 degrees gives 1; aligned at 0 degrees gives 0."
    verbose['initial_reference']=None
    verbose['windows']={'final':{'start_time_sec':float(times[start]),'end_time_sec':float(times[-1]),'stable':True,'stability_required':False}}
    verbose['measurements']={'final_frame_indices':indices,'final_direction_separation_deg':separation,
        'per_frame_direction_separation_deg':differences.tolist(),'final_window_frame_count':len(rows)-start,
        'first_frame_direction_used':False}
    verbose['score_details']={'version':'final_direction_v1','formula':'(1-cos(final_direction_separation))/2','range':[0,1]}
    verbose['status']='scored';verbose['reason']="Compare only observed needle directions in readable final frames; rotation trajectories and stabilization duration are not required."
    return score
