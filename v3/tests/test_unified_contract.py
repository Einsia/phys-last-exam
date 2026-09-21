"""Contract edge cases and replay of frozen G3/G7 measurements (no video inference)."""
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unified_evaluators.contract import aggregate, already_scored_result, classical_result, finalize, gm, metric, physeval_result, q
from unified_evaluators.physics import g7_result, score_g3
from scripts.finalize_all_eval import rich_metrics


class ContractTests(unittest.TestCase):
    def test_public_result_has_strict_metric_shape(self):
        metadata={'sample_id':'sample_00','model_folder':'minimax_h3',
                  'video_path':'output_videos/minimax_h3/sample_00.mp4',
                  'image_path':'first_frames/gpt/gpt_01.png',
                  'video_prompt':'prompt','model':'minimax_h3','seed':42}
        metrics={'M1':metric(0.2,q(0.2,0.2),True),'M2':metric(defined=False)}
        blocks={'M1':{'defined':True,'principle':'measure','measurement_steps':['track'], 'evidence':[]},
                'M2':{'defined':False,'principle':'undefined','measurement_steps':[], 'evidence':[]}}
        result=finalize('P1',metadata,metrics,blocks,[],{'run_started_at_utc':'x'})
        self.assertEqual(set(result),{'task_id','video_path','image_path','video_prompt','model','seed','metrics','verbose'})
        self.assertEqual(set(result['metrics']['M1']),{'extract_success','metric'})
        self.assertEqual(result['metrics']['M1']['metric'],0.575)
        self.assertEqual(result['verbose']['M1']['raw_metric'],0.2)
        self.assertEqual(result['verbose']['M1']['_scoring_summary']['metric_semantics'],'normalized_final_score')
        self.assertEqual(result['metrics']['M2'],{'extract_success':None,'metric':None})
        self.assertEqual(result['video_path'],'minimax_h3/videos/sample_00.mp4')
        self.assertEqual(result['model'],'minimax-h3')
        self.assertEqual(result['verbose']['M1']['_scoring_summary']['score'],
                         result['verbose']['M1']['scoring']['proxy_score'])

    def test_public_score_round_trip_preserves_raw_values_and_null_states(self):
        metadata={'sample_id':'sample_00','video_path':'sample_00.mp4'}
        for value,physics in ((False,0.0),(12.0,0.5),({'error':4.0},0.2)):
            for defined in (True,False):
                with self.subTest(value=value,m2_defined=defined):
                    metrics={'M1':metric(value,physics,True),'M2':metric(defined=defined)}
                    blocks={'M1':{'defined':True},'M2':{'defined':defined}}
                    first=finalize('P1',metadata,metrics,blocks,[],{})
                    self.assertEqual(first['metrics']['M1']['metric'],0.15+0.85*physics)
                    self.assertEqual(first['verbose']['M1']['raw_metric'],value)
                    self.assertIsNone(first['metrics']['M2']['metric'])
                    self.assertEqual(first['metrics']['M2']['extract_success'],False if defined else None)
                    self.assertEqual(rich_metrics(first),metrics)
                    second=finalize('P1',metadata,rich_metrics(first),first['verbose'],[],{})
                    self.assertEqual(second,first)

    def test_half_error_and_recognition(self):
        self.assertEqual(q(0.2,0.2),0.5)
        self.assertEqual(q(-0.2,0.2),0.5)
        self.assertEqual(metric(0.2,q(0.2,0.2),True)['proxy_score'],0.575)

    def test_undefined_is_not_failed(self):
        one = metric(0.0,1.0,True)
        absent = aggregate({'M1':one,'M2':metric(defined=False)})
        failed = aggregate({'M1':one,'M2':metric()})
        self.assertEqual(absent['score'],1.0)
        self.assertEqual(absent['score_status'],'complete')
        self.assertTrue(absent['overall_proxy_valid'])
        self.assertEqual(failed['score'],0.5)
        self.assertEqual(failed['score_status'],'partial')
        self.assertFalse(failed['overall_proxy_valid'])

    def test_invalid_measurements_get_no_recognition(self):
        for value in (None,float('nan'),float('inf'),{'one':None}):
            with self.subTest(value=value):
                result = metric(value,0.5,True)
                self.assertFalse(result['extract_success'])
                self.assertIsNone(result['metric'])
                self.assertIsNone(result['physics_score'])
                self.assertEqual(result['proxy_score'],0)

    def test_zero_physics_is_measurable(self):
        result = metric(False,0.0,True)
        self.assertTrue(result['extract_success'])
        self.assertEqual(result['proxy_score'],0.15)

    def test_undefined_verbose_null_placeholder(self):
        raw={'metrics':{'M1':metric(0.0,1.0,True),'M2':metric(defined=False)},'verbose':{'M1':{},'M2':None}}
        metrics,blocks=already_scored_result(raw)
        self.assertEqual(aggregate(metrics)['score'],1)
        self.assertFalse(blocks['M2']['defined'])

    def test_internal_missing_is_not_dropped(self):
        self.assertIsNone(gm([1.0,None]))
        self.assertEqual(gm([1.0,0.0]),0)

    def test_multislot_m2_groups_before_recognition(self):
        raw = {'metrics':{},'verbose':{}}
        for key,error in [('M1',0.0),('M2',1.0),('M3',3.0)]:
            raw['metrics'][key] = {'extract_success':True,'metric':error}
            raw['verbose'][key] = {'error_scale_a':1.0,'principle':'test','non_residual':False}
        result,_ = physeval_result(raw)
        self.assertEqual(set(result),{'M1','M2'})
        self.assertEqual(result['M2']['metric'],{'M2':1.0,'M3':3.0})
        self.assertAlmostEqual(result['M2']['proxy_score'],0.15+0.85*math.sqrt(.5*.25))
        raw['metrics']['M3']['extract_success'] = False
        raw['metrics']['M3']['metric'] = None
        result,_ = physeval_result(raw)
        self.assertFalse(result['M2']['extract_success'])
        self.assertEqual(aggregate(result)['score'],0.5)

    def test_classical_metric_is_physical_error(self):
        raw={'task_id':'P19','metrics':{k:{'extract_success':True,'metric':0.9} for k in ('M1','M2')},
             'verbose':{'measurements':{'level_difference_signed_norm':-0.4,'terminal_surface_speed':0.2,
                'score_normalization':{'M1':{'raw_measurement':'level_difference_signed_norm','physics_score':q(-0.4,1)},
                                       'M2':{'raw_measurement':'terminal_surface_speed','physics_score':q(0.2,1)}}}}}
        result,_ = classical_result(raw)
        self.assertEqual(result['M1']['metric'],-0.4)
        self.assertAlmostEqual(result['M1']['proxy_score'],0.15+0.85/1.4)

    def test_p12_partial_ray_coverage(self):
        raw={'task_id':'P12','extract_success':True,'metrics':{'M1':0.0},'measurements':{'per_ray_snell_residual':[0.0]}}
        result,_=g7_result(raw)
        self.assertTrue(result['M2']['extract_success'])
        self.assertAlmostEqual(result['M2']['physics_score'],1/3)
        self.assertAlmostEqual(result['M2']['proxy_score'],0.15+0.85/3)

    def test_p27_undefined_event_does_not_invalidate_area(self):
        raw={'task_id':'P27','extract_success':True,'metrics':{'M1':None,'M2_initial_projected_area_log_error':0.0},'measurements':{}}
        result,_=g7_result(raw)
        self.assertFalse(result['M1']['extract_success'])
        self.assertTrue(result['M2']['extract_success'])
        self.assertEqual(aggregate(result)['score'],0.5)

    def test_replay_all_g3_frozen_values_and_gates(self):
        count=0
        fixture_root = ROOT/'work/g1_g9_v2_20260911/before/g3'
        if not fixture_root.exists():
            self.skipTest('historical replay fixtures are not part of the source-only repository')
        for path in sorted(fixture_root.glob('P*/eval_results/minimax_h3/result*.json')):
            old=json.loads(path.read_text())
            source=old['source_metadata']
            metrics,_=score_g3(old['task_id'],source['raw_metrics'],source['metric_validity'])
            with self.subTest(path=path.name):
                for key in ('M1','M2'):
                    self.assertEqual(metrics[key]['extract_success'],old['metrics'][key]['extract_success'])
                    self.assertAlmostEqual(metrics[key]['proxy_score'],old['metrics'][key]['proxy_score'],delta=1e-12)
            count+=1
        self.assertEqual(count,168)

    def test_replay_g7_frozen_values_and_gates(self):
        count=0
        fixture_root = ROOT/'work/g1_g9_v2_20260911/before/g7'
        if not fixture_root.exists():
            self.skipTest('historical replay fixtures are not part of the source-only repository')
        for path in sorted(fixture_root.glob('P*/eval_results/minimax_h3/result*.json')):
            old=json.loads(path.read_text())
            if 'proxy' not in old: # Five overwritten sample_00 outputs have no historical physical measurement.
                continue
            verbose=old['verbose']
            raw={'task_id':old['task_id'],'metrics':verbose['raw_metrics'],
                 'measurements':verbose['measurements'],'extract_success':verbose['status']['extract_success']}
            metrics,_=g7_result(raw)
            with self.subTest(path=str(path)):
                for key in ('M1','M2'):
                    self.assertEqual(metrics[key]['extract_success'],old['metrics'][key]['extract_success'])
                    self.assertAlmostEqual(metrics[key]['proxy_score'],old['metrics'][key]['proxy_score'],delta=1e-12)
            count+=1
        self.assertEqual(count,115)


if __name__=='__main__':
    unittest.main()
