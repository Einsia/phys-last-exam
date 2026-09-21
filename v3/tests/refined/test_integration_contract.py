from pathlib import Path
from unittest import TestCase
from tempfile import TemporaryDirectory
import json,subprocess,sys
import numpy as np
from refined_evaluators.batch import associate,output_location,load_metadata
from refined_evaluators.common import ExtractionError
from refined_evaluators.tasks import circle_observation,quad_observation
from refined_evaluators.vision import load_annotation
from types import SimpleNamespace
import cv2
ROOT=Path(__file__).resolve().parents[2]

class BatchContract(TestCase):
    def test_metadata_paths_not_enumeration(self):
        with TemporaryDirectory() as td:
            root=Path(td);video=root/'second.mp4';video.touch();image=root/'second.png';image.touch()
            entries=[{'sample_id':'sample_42','video_path':'second.mp4','image_path':'second.png','model':'known'}]
            a=associate(video,entries,root,root);m,s,p,d=output_location(root/'results',video,a)
            self.assertEqual(s,'sample_42');self.assertEqual(p.name,'result_sample_42.json');self.assertEqual(a['image_path'],str(image))
    def test_unknown_model_stays_null(self):
        m,s,p,d=output_location('/tmp/results','foo.mp4',{});self.assertIsNone(m);self.assertEqual(s,'foo');self.assertEqual(p.parent.name,'unknown_model')
    def test_conflicting_metadata_rejected(self):
        with TemporaryDirectory() as td:
            root=Path(td);v=root/'a.mp4';v.touch()
            with self.assertRaises(ValueError):associate(v,[{'video_path':'a.mp4'},{'video_path':'a.mp4'}],root,root)
    def test_failed_sample_does_not_stop_batch_or_reuse_stale_result(self):
        with TemporaryDirectory() as td:
            root=Path(td);(root/'a.mp4').write_bytes(b'not a video');(root/'b.mp4').write_bytes(b'not a video')
            proc=subprocess.run([sys.executable,str(ROOT/'g8/P37/evaluator/batch.py'),'--input_dir',str(root)],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            summary=json.loads((root/'eval_results/unknown_model/batch_summary.json').read_text());self.assertEqual(len(summary['samples']),2)
            self.assertTrue(all(s['M1']['extract_success'] is False and s['M1']['metric'] is None
                                and s['proxy']['score'] is None and s['status']=='consistency_error'
                                and not s['proxy']['physics_attempted'] for s in summary['samples']))
    def test_shell_works_outside_workspace(self):
        p=subprocess.run(['bash',str(ROOT/'g8/P37/scripts/run_eval.sh'),'--help'],cwd='/tmp',capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)

class GeometryContract(TestCase):
    def test_circle_fit_ignores_hole_and_highlight(self):
        m=np.zeros((120,160),np.uint8);cv2.circle(m,(85,61),27,1,-1);cv2.circle(m,(90,53),6,0,-1);o=circle_observation(m)
        self.assertTrue(o['valid']);self.assertLess(abs(o['radius']-27),1);self.assertLess(np.linalg.norm(np.array(o['center'])-[85,61]),.2)
    def test_quad_material_identity_across_rotation(self):
        previous=None
        for theta in [0,10,20,30,45,60]:
            rect=((120.,120.),(36.,130.),-theta);points=cv2.boxPoints(rect);mask=np.zeros((250,250),np.uint8);cv2.fillPoly(mask,[points.astype(int)],1);obs=quad_observation(mask,previous)
            self.assertIsNotNone(obs)
            if previous is not None:self.assertLess(np.max(np.linalg.norm(obs[0]-previous,axis=1)),35)
            previous=obs[0]
    def test_annotation_cannot_be_reused_for_another_video(self):
        with TemporaryDirectory() as td:
            root=Path(td);v=root/'new.mp4';v.write_bytes(b'changed');a=root/'annotation.json';a.write_text(json.dumps({'source_video_sha256':'wrong'}))
            with self.assertRaises(ExtractionError):load_annotation(SimpleNamespace(annotation=str(a),video_path=str(v),image_path=None),[np.zeros((100,100,3),np.uint8)],root)
