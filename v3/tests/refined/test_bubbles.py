import unittest
import cv2
import numpy as np
from refined_evaluators.bubbles import circle_fit, inspect_frame, partition_geometry, signed_partition_fit, summarize
from refined_evaluators.definitions import DEFAULTS


def fixture(direction=1, radius=300, double=False, equal=False):
    frame=np.zeros((600,900,3),np.uint8)+15
    small={'center':[290,280],'radius':110}
    large={'center':[495,280],'radius':165 if not equal else 112}
    mid,u,n,half=partition_geometry(small,large)
    for item,is_small in ((small,True),(large,False)):
        theta=np.linspace(0,2*np.pi,1600)
        points=np.asarray(item['center'])+item['radius']*np.c_[np.cos(theta),np.sin(theta)]
        for p in points:
            if ((p-mid)@u<0)==is_small:cv2.circle(frame,tuple(np.rint(p).astype(int)),1,(235,235,235),-1)
    z=np.linspace(-half,half,500)
    if radius is None:offset=z*0
    else:offset=np.sqrt(radius**2-z**2)-np.sqrt(radius**2-half**2)
    for sign in ([1,-1] if double else [direction]):
        pts=mid+z[:,None]*n+sign*offset[:,None]*u
        cv2.polylines(frame,[np.rint(pts).astype('int32')],False,(235,235,235),1)
    return frame,[[*small['center'],small['radius']],[*large['center'],large['radius']]]


class BubbleGeometryTests(unittest.TestCase):
    def test_independent_circle_fit_and_signed_curvature(self):
        z=np.linspace(-80,80,150);radius=330
        for sign in (-1,1):
            points=np.c_[sign*(np.sqrt(radius**2-z*z)-np.sqrt(radius**2-80**2)),z]
            fit=signed_partition_fit(points,np.zeros(2),np.array([1.,0]),np.array([0.,1]),80)
            self.assertAlmostEqual(fit['signed_radius'],sign*radius,places=4)
    def test_visible_ideal_and_wrong_direction_are_both_measurable(self):
        errors=[]
        for sign in (1,-1):
            frame,initial=fixture(direction=sign,radius=330)
            row,_=inspect_frame(frame,initial,DEFAULTS['P47'])
            self.assertTrue(row['valid'],row['reason']);errors.append(row['raw_m1_error'])
        self.assertLess(errors[0],.15);self.assertGreater(errors[1],1.8)
    def test_resolved_two_sided_partition_is_not_score_selected(self):
        frame,initial=fixture(double=True,radius=220)
        row,_=inspect_frame(frame,initial,DEFAULTS['P47'])
        self.assertFalse(row['valid']);self.assertIn('Two resolved partition boundaries',row['reason'])
    def test_nearly_equal_bubbles_are_unidentifiable(self):
        frame,initial=fixture(equal=True)
        row,_=inspect_frame(frame,initial,DEFAULTS['P47'])
        self.assertFalse(row['valid']);self.assertIn('sizes',row['reason'])
    def test_precisely_measured_straight_partition_is_a_limit_not_missing(self):
        frame,initial=fixture(radius=None)
        row,_=inspect_frame(frame,initial,DEFAULTS['P47'])
        self.assertTrue(row['valid'],row['reason'])
        self.assertEqual(row.get('error_kind'),'unbounded_straight_partition')
        m,q=summarize([row]*20,np.arange(20)*.05,DEFAULTS['P47'])
        self.assertEqual(q,0);self.assertIsNone(m['raw_m1_error'])
    def test_pts_weighting_and_missing_frames_not_dropped(self):
        row={'valid':True,'formed':True,'raw_m1_error':.2,'outer':{'small':{'radius':100},'large':{'radius':200}},'partition':{'signed_radius':250}}
        bad={'valid':False,'formed':True,'reason':'occluded'}
        _,q=summarize([row,bad,bad,row],np.array([0,.2,.4,1.]),DEFAULTS['P47'])
        self.assertIsNone(q)
