import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.common import ExtractionError, Thresholds, clean_json, new_result
from utils.measurement import (circular_calibration, transform_points, decide, summarize,
                               unwrap_track, dial_geometry, red_pole, track_needles, measure_image_reference)
from utils.media import VideoWriter, decode
from batch import associate, output_location


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Thresholds()
        self.times = np.arange(101)*.05

    def assess(self,d1,d2,start=70):
        rows=[]
        for t in self.times:
            fraction=np.clip((t-.75)/3.25,0,1)
            rows.append({name:{'valid':True,'angle_deg':float((start+d*fraction+180)%360-180)}
                         for name,d in (('left',d1),('right',d2))})
        verbose=new_result('synthetic')['verbose']['M1']
        value=summarize(rows,self.times,self.cfg,verbose)
        return value,verbose

    def test_opposite_equal(self):
        self.assertEqual(self.assess(30,-30)[0],1)

    def test_same_direction(self):
        self.assertEqual(self.assess(30,30)[0],0)

    def test_zero_motion_and_one_small(self):
        self.assertEqual(self.assess(0,0)[0],0)
        self.assertAlmostEqual(self.assess(4,-6)[0],0.64)

    def test_magnitude_mismatch(self):
        self.assertAlmostEqual(self.assess(40,-15)[0],1-25/55)

    def test_crossing_180(self):
        value,verbose=self.assess(30,-30,start=170)
        self.assertEqual(value,1)
        self.assertAlmostEqual(verbose['measurements']['delta_1_deg'],30)

    def test_pole_switch_rejected(self):
        angles=np.zeros(101); angles[30:]=180
        with self.assertRaisesRegex(ExtractionError,'pole switch'):
            unwrap_track(self.times,angles,self.cfg)

    def test_long_gap_rejected(self):
        angles=np.zeros(101);angles[40:43]=np.nan
        with self.assertRaisesRegex(ExtractionError,'gap'):
            unwrap_track(self.times,angles,self.cfg)

    def moving_rows(self,opposite=True,start=70):
        return [{name:{'valid':True,'angle_deg':float((start+sign*min(t,3)*20+180)%360-180)}
                 for name,sign in (('left',1),('right',-1 if opposite else 1))} for t in self.times]

    def test_immediate_motion_accepted_without_still_image(self):
        for opposite,expected in ((True,1),(False,0)):
            verbose=new_result('synthetic')['verbose']['M1']
            self.assertEqual(summarize(self.moving_rows(opposite),self.times,self.cfg,verbose),expected)
            self.assertEqual(verbose['initial_reference']['source'],'video_first_frame')
            self.assertFalse(verbose['initial_reference']['stationarity_required'])
            self.assertEqual(set(verbose['windows']),{'post'})
            self.assertAlmostEqual(verbose['measurements']['delta_1_deg'],60)

    def test_supplied_still_angle_is_actually_used(self):
        ref={'accepted':True,'image_path':'matched.png',
             'needles':{name:{'angle_deg':68.0} for name in ('left','right')}}
        verbose=new_result('synthetic')['verbose']['M1']
        self.assertAlmostEqual(summarize(self.moving_rows(),self.times,self.cfg,verbose,ref),1-4/120)
        self.assertEqual(verbose['initial_reference']['source'],'matched_first_frame_image')
        self.assertAlmostEqual(verbose['measurements']['delta_1_deg'],62)
        self.assertAlmostEqual(verbose['measurements']['delta_2_deg'],-58)

    def test_still_angle_join_across_180(self):
        ref={'accepted':True,'needles':{name:{'angle_deg':-179.0} for name in ('left','right')}}
        verbose=new_result('synthetic')['verbose']['M1']
        self.assertAlmostEqual(summarize(self.moving_rows(start=179),self.times,self.cfg,verbose,ref),1-4/120)
        self.assertAlmostEqual(verbose['measurements']['theta_1_pre_deg'],181)
        self.assertAlmostEqual(verbose['measurements']['delta_1_deg'],58)

    def test_rejected_still_falls_back_to_video_start(self):
        verbose=new_result('synthetic')['verbose']['M1']
        summarize(self.moving_rows(),self.times,self.cfg,verbose,{'accepted':False,'reason':'mismatch'})
        self.assertEqual(verbose['initial_reference']['source'],'video_first_frame')

    def test_missing_first_observation_still_rejected(self):
        rows=self.moving_rows()
        rows[0]['left']={'valid':False,'angle_deg':None}
        with self.assertRaisesRegex(ExtractionError,'initial red-pole'):
            summarize(rows,self.times,self.cfg,new_result('synthetic')['verbose']['M1'])

    def test_no_final_stability(self):
        rows=[{name:{'valid':True,'angle_deg':t*20} for name in ('left','right')} for t in self.times]
        with self.assertRaisesRegex(ExtractionError,'Final'):
            summarize(rows,self.times,self.cfg,new_result('synthetic')['verbose']['M1'])

    def test_short_video_with_sufficient_final_window(self):
        times=np.arange(16)*.05
        rows=[{name:{'valid':True,'angle_deg':sign*min(t,.2)*100}
               for name,sign in (('left',1),('right',-1))} for t in times]
        self.assertEqual(summarize(rows,times,self.cfg,new_result('synthetic')['verbose']['M1']),1)

    def test_score_continuous_across_old_tolerance_boundary(self):
        a=decide(25,-20,self.cfg)[0]
        b=decide(25,-19.9,self.cfg)[0]
        self.assertAlmostEqual(a,1-5/45)
        self.assertAlmostEqual(b,1-5.1/44.9)
        self.assertLess(abs(a-b),.01)

    def test_normalized_score_examples(self):
        for d1,d2,expected in ((30,-30,1.0),(30,-20,.8),(30,-10,.5),
                               (30,20,0.0),(0,0,0.0),(0,30,0.0),(2,-2,.4)):
            score,_,_,details=decide(d1,d2,self.cfg)
            self.assertIsInstance(score,float)
            self.assertAlmostEqual(score,expected)
            self.assertAlmostEqual(score,details['symmetry_score']*details['motion_score'])

    def test_score_range_and_symmetry(self):
        for d1 in (-180,-30,-2,0,2,30,180):
            for d2 in (-180,-30,-2,0,2,30,180):
                score=decide(d1,d2,self.cfg)[0]
                self.assertGreaterEqual(score,0.0)
                self.assertLessEqual(score,1.0)
                self.assertAlmostEqual(score,decide(d2,d1,self.cfg)[0])
                self.assertAlmostEqual(score,decide(-d1,-d2,self.cfg)[0])

    def test_nonfinite_deflections_rejected(self):
        for value in (math.nan,math.inf,-math.inf):
            with self.assertRaises(ExtractionError):
                decide(value,-30,self.cfg)


class GeometryTests(unittest.TestCase):
    def test_projected_circle_uses_true_pivot(self):
        angle=np.linspace(0,2*np.pi,500,endpoint=False)
        points=np.column_stack([np.cos(angle),np.sin(angle)])*70
        camera=np.array([[1.0,.12,200],[.08,.8,150],[.001,-.0007,1.]])
        projected=transform_points(points,camera)
        ellipse=cv2.fitEllipse(projected.astype(np.float32))
        pivot=transform_points([[0,0]],camera)[0]
        rectifier=circular_calibration(ellipse,pivot)
        corrected=transform_points(projected,rectifier)
        self.assertLess(np.max(np.abs(np.linalg.norm(corrected,axis=1)-1)),1e-5)
        a=np.unwrap(np.arctan2(corrected[:,1],corrected[:,0]))
        self.assertLess(np.max(np.abs((a-a[0])-angle)),1e-5)

    def fixture(self,angle):
        image=np.zeros((240,240,3),np.uint8)
        centre=(120,120)
        cv2.circle(image,centre,90,(30,145,200),-1)
        cv2.circle(image,centre,82,(225,225,225),-1)
        u=np.array([math.cos(math.radians(angle)),-math.sin(math.radians(angle))])
        n=np.array([-u[1],u[0]])
        for sign,color in ((1,(20,20,180)),(-1,(30,30,30))):
            pts=np.array([np.array(centre)+sign*u*72,np.array(centre)+n*5,np.array(centre)-n*5],np.int32)
            cv2.fillConvexPoly(image,pts,color)
        cv2.circle(image,centre,7,(30,145,200),-1)
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.circle(mask,centre,90,1,-1)
        return image,mask.astype(bool)

    def test_red_identity_and_pivot(self):
        cfg=Thresholds()
        for angle in (20,110,-175):
            image,mask=self.fixture(angle)
            geom=dial_geometry(image,mask,cfg)
            obs=red_pole(image,mask,geom,cfg)
            error=(obs['angle_deg']-angle+180)%360-180
            self.assertLess(abs(error),1.5)
            self.assertLess(np.linalg.norm(geom['pivot']-[120,120]),1)

    def test_missing_pole_rejected(self):
        image,mask=self.fixture(30)
        geom=dial_geometry(image,mask,Thresholds())
        image[:]=(225,225,225)
        with self.assertRaises(ExtractionError):
            red_pole(image,mask,geom,Thresholds())

    def test_supplied_still_requires_matching_red_poles(self):
        image,mask=self.fixture(30)
        geometry=dial_geometry(image,mask,Thresholds())
        obs={'valid':True,**red_pole(image,mask,geometry,Thresholds())}
        first_row={name:obs for name in ('left','right')}
        args=(np.stack([mask,mask]),[geometry,geometry],first_row,Thresholds())
        accepted=measure_image_reference(image,*args)
        self.assertTrue(accepted['accepted'])
        wrong,_=self.fixture(70)
        rejected=measure_image_reference(wrong,*args)
        self.assertFalse(rejected['accepted'])
        self.assertIn('needle direction differs',rejected['reason'])

    def test_compass_identity_swap_rejected(self):
        left,mask=self.fixture(30)
        right,_=self.fixture(60)
        image=np.concatenate([left,right],axis=1)
        masks=np.stack([np.concatenate([mask,np.zeros_like(mask)],axis=1),
                        np.concatenate([np.zeros_like(mask),mask],axis=1)])
        # Static texture provides camera-registration features outside both objects.
        noise=np.random.default_rng(7).integers(0,180,image.shape,dtype=np.uint8)
        image[~masks.any(axis=0)]=noise[~masks.any(axis=0)]
        sequence=np.stack([masks,masks[::-1],masks])
        rows,_,errors=track_needles([image]*3,np.array([0,.05,.1]),sequence,Thresholds())
        self.assertEqual(errors,[None,None])
        self.assertTrue(rows[0]['left']['valid'])
        self.assertFalse(rows[1]['left']['valid'])
        self.assertIn('identity switched',rows[1]['left']['reason'])


class IOTests(unittest.TestCase):
    def test_model_sample_output_naming(self):
        model,sample_id,result,debug=output_location(Path('/results'),Path('/input/continuation.mp4'),
            {'model':'minimax_h3','sample_id':'sample_00'})
        self.assertEqual((model,sample_id),('minimax_h3','sample_00'))
        self.assertEqual(str(result),'/results/minimax_h3/result_sample_00.json')
        self.assertEqual(str(debug),'/results/minimax_h3/debug_sample_00')
        with self.assertRaises(ValueError):
            output_location(Path('/results'),Path('/input/a.mp4'),{'model':'../escape','sample_id':'sample_00'})

    def test_explicit_sample_id_does_not_depend_on_video_stem(self):
        directory=Path('/input')
        entries=[{'sample_id':'sample_02','video_path':'b.mp4','model':'minimax_h3'},
                 {'sample_id':'sample_00','video_path':'continuation.mp4','model':'minimax_h3'}]
        record=associate(directory/'continuation.mp4',entries,directory,directory)
        self.assertEqual(record['sample_id'],'sample_00')
        self.assertEqual(output_location(Path('/results'),directory/'continuation.mp4',record)[2].name,
                         'result_sample_00.json')

    def test_nonfinite_json(self):
        data=clean_json({'a':np.float32(np.nan),'b':[math.inf,-math.inf], 'c':np.array([1,2])})
        self.assertEqual(data,{'a':None,'b':[None,None],'c':[1,2]})
        json.dumps(data,allow_nan=False)

    def test_vfr_intermediate_preserves_pts(self):
        with tempfile.TemporaryDirectory() as tmp:
            times=np.array([10,10.04,10.10,10.13,10.21])
            path=Path(tmp)/'out.mp4'
            w=VideoWriter(path,65,49,times)
            try:
                for i,t in enumerate(times):
                    w.write(np.full((49,65,3),40*i,np.uint8),t)
            finally: w.close()
            frames,decoded,_=decode(path)
            self.assertEqual(len(frames),len(times))
            np.testing.assert_allclose(decoded,times-times[0],atol=1e-6)

    def test_metadata_order_conflict_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            entries=[{'sample_id':'b','model':'B'},{'sample_id':'a','model':'A'}]
            self.assertEqual(associate(directory/'a.mp4',entries,directory,directory)['model'],'A')
            self.assertEqual(associate(directory/'unknown.mp4',entries,directory,directory),{})
            with self.assertRaises(ValueError):
                associate(directory/'a.mp4',[{'sample_id':'a','video_path':'b.mp4'}],directory,directory)
            with self.assertRaises(ValueError):
                associate(directory/'a.mp4',[{'sample_id':'a'},{'sample_id':'a'}],directory,directory)

    def test_batch_continues_after_invalid_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'a.mp4').write_bytes(b'invalid video')
            (directory/'b.mp4').write_bytes(b'invalid video')
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'batch.py'),
                     '--input_dir',str(directory),'--device','cpu',
                     '--sam2_checkpoint',str(directory/'missing.pt')]
            p=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            data=json.loads((directory/'eval_results/unknown_model/batch_summary.json').read_text())
            self.assertEqual(len(data['samples']),2)
            for sample in data['samples']:
                self.assertFalse(sample['M1']['extract_success'])
                self.assertIsNone(sample['M1']['metric'])
                self.assertEqual(sample['status'],'environment_error')


if __name__=='__main__':
    unittest.main()
