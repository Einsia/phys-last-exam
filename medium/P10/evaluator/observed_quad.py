"""Fit four independently observed sides while preserving boundary adjacency."""
import cv2
import numpy as np


def quad_observation(mask, previous=None):
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    seeds = []
    for fraction in [.01, .015, .02, .025, .03, .04, .05]:
        candidate = cv2.approxPolyDP(hull, fraction*cv2.arcLength(hull, True), True)
        if len(candidate) == 4:
            seeds.append(candidate[:, 0, :].astype(float))
            break
    # A small segmentation appendage can replace a real corner on the convex
    # hull. The enclosing box is an additional search-band initializer only;
    # its angle/lengths are never returned as measurements. Each of its four
    # sides is independently refitted to actual contour pixels below.
    seeds.append(cv2.boxPoints(cv2.minAreaRect(contour)).astype(float))
    candidates = []
    for seed in seeds:
        result = fit_sides(contour, seed, mask.shape)
        if result is not None:
            candidates.append(result)
    if not candidates:
        return None
    p, area_error, _ = max(candidates, key=lambda result: result[2])
    # Boundary neighbors stay neighbors. An arbitrary four-point permutation
    # can be closer to the previous pose yet be a self-intersecting polygon.
    if cv2.contourArea(p.astype('float32'), oriented=True) < 0:
        p = p[::-1]
    cyclic = [np.roll(p, -k, axis=0) for k in range(4)]
    if previous is None:
        p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    else:
        p = min(cyclic, key=lambda q: float(np.sum((q-previous)**2)))
    return p, area_error


def fit_sides(contour, p, shape):
    points = contour[:, 0, :].astype(float)
    lines = []
    for j in range(4):
        a, b = p[j], p[(j+1) % 4]
        edge = b-a
        length = np.linalg.norm(edge)
        if length < 8:
            return None
        u = edge/length
        normal = np.array([-u[1], u[0]])
        along = (points-a) @ u
        distance = np.abs((points-a) @ normal)
        selected = points[(along > .15*length) & (along < .85*length) & (distance < max(4., .1*length))]
        if len(selected) < 8:
            return None
        fitted = cv2.fitLine(selected.astype('float32'), cv2.DIST_HUBER, 0, .01, .01).ravel()
        origin, direction = fitted[2:].astype(float), fitted[:2].astype(float)
        # Trim only mask-contour outliers, then refit the independently visible
        # side. Require enough inliers spread over a substantial side length;
        # a short adjacent side cannot masquerade as this side.
        for _ in range(3):
            residual = np.abs((selected-origin) @ np.array([-direction[1], direction[0]]))
            inliers = selected[residual <= 2.5]
            if len(inliers) < max(8, .5*len(selected)):
                return None
            fitted = cv2.fitLine(inliers.astype('float32'), cv2.DIST_HUBER, 0, .01, .01).ravel()
            origin, direction = fitted[2:].astype(float), fitted[:2].astype(float)
        if float(np.ptp((inliers-origin) @ direction)) < .45*length:
            return None
        lines.append((origin, direction))
    refined = []
    for j in range(4):
        a, u = lines[(j-1) % 4]
        b, v = lines[j]
        matrix = np.c_[u, -v]
        if abs(np.linalg.det(matrix)) < .15:
            return None
        refined.append(a+np.linalg.solve(matrix, b-a)[0]*u)
    q = np.asarray(refined)
    if not np.isfinite(q).all() or not cv2.isContourConvex(q.astype('float32')):
        return None
    diagonal = np.linalg.norm(p.max(axis=0)-p.min(axis=0))
    if np.max(np.linalg.norm(q-p, axis=1)) > max(6., .15*diagonal):
        return None
    fitted_area = abs(cv2.contourArea(q.astype('float32')))
    area_error = float(abs(cv2.contourArea(contour)-fitted_area)/max(fitted_area, 1.))
    if area_error > .12:
        return None
    fitted_mask = np.zeros(shape, np.uint8)
    observed_mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(fitted_mask, np.rint(q).astype(int), 1)
    cv2.drawContours(observed_mask, [contour], -1, 1, cv2.FILLED)
    overlap = float(np.count_nonzero(fitted_mask & observed_mask) / max(1, np.count_nonzero(fitted_mask | observed_mask)))
    if overlap < .85:
        return None
    return q, area_error, overlap
