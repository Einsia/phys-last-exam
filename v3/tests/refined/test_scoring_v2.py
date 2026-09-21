import copy
import json
import math
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from refined_evaluators.scoring import residual_score, metric_score, aggregate, score_result, LEGACY_VERSION, VERSION
from refined_evaluators.output import write_scoring_calculation

ROOT = Path(__file__).resolve().parents[2]


class RecognitionScoringTests(unittest.TestCase):
    def test_document_anchors_signed_error_and_unit_invariance(self):
        for ratio, expected in ((0,1),(.5,.7166666666666667),(1,.575),(2,.43333333333333335),(4,.32),(8,.24444444444444444)):
            for sign in (-1,1):
                self.assertAlmostEqual(metric_score(residual_score(sign*ratio*.25,.25),True),expected)
                self.assertAlmostEqual(residual_score(sign*ratio*.25,.25),residual_score(sign*ratio*250,250))
    def test_nonfinite_values_and_invalid_scale_rejected(self):
        for e in (None,True,float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):residual_score(e,1)
        for a in (None,0,-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):residual_score(1,a)
    def test_finite_large_errors_do_not_have_a_hard_cutoff(self):
        scores=[residual_score(x,1) for x in (0,1,2,10,1e100,1e308)]
        self.assertTrue(all(a>b>0 for a,b in zip(scores,scores[1:])))
    def test_reliably_wrong_gets_recognition_but_failure_does_not(self):
        self.assertEqual(metric_score(0,True),.15)
        self.assertEqual(metric_score(None,False),0)
    def test_defined_but_failed_metrics_keep_their_weight(self):
        for a,b,expected,status in ((.8,.4,.6,'complete'),(.8,None,.4,'partial'),(None,.8,.4,'partial'),(None,None,0,'unavailable')):
            metrics={k:{'extract_success':v is not None,'proxy_score':v or 0} for k,v in (('M1',a),('M2',b))}
            out=aggregate(metrics)
            self.assertAlmostEqual(out['score'],expected)
            self.assertEqual(out['score_status'],status)
            self.assertEqual(out['overall_proxy_valid'],status=='complete')
    def test_undefined_m2_is_excluded_and_m1_can_reach_full_score(self):
        for score in (0,.15,.575,1):
            out=aggregate({'M1':{'extract_success':True,'proxy_score':score},
                           'M2':{'extract_success':None,'proxy_score':0}})
            self.assertEqual(out['score'],score)
            self.assertEqual(out['weights'],{'M1':1.,'M2':0.})
            self.assertEqual(out['score_status'],'complete')
            self.assertTrue(out['overall_proxy_valid'])
            self.assertEqual(out['defined_metrics'],['M1'])
            self.assertEqual(out['not_applicable_metrics'],['M2'])
    def test_failed_single_metric_and_no_defined_metrics_are_unavailable(self):
        for success,weights in ((False,{'M1':1.,'M2':0.}),(None,{'M1':0.,'M2':0.})):
            out=aggregate({'M1':{'extract_success':success,'proxy_score':0},
                           'M2':{'extract_success':None,'proxy_score':0}})
            self.assertEqual(out['score'],0)
            self.assertEqual(out['weights'],weights)
            self.assertEqual(out['score_status'],'unavailable')
            self.assertFalse(out['overall_proxy_valid'])
    def test_undefined_m1_does_not_reduce_defined_m2(self):
        out=aggregate({'M1':{'extract_success':None,'proxy_score':0},
                       'M2':{'extract_success':True,'proxy_score':.8}})
        self.assertEqual(out['score'],.8)
        self.assertEqual(out['weights'],{'M1':0.,'M2':1.})
        self.assertTrue(out['overall_proxy_valid'])
    def fixture(self, task):
        group='g8' if task in ('P34','P37','P38','P39','P41') else 'g9'
        path=ROOT/'work/g8_g9_v2_20260910/before'/group/task/'eval_results/minimax_h3/result_sample_00.json'
        return json.loads(path.read_text())
    def test_all_historical_measurements_retained_and_m2_not_invented(self):
        for task in ('P34','P37','P38','P39','P41','P43','P49'):
            d=self.fixture(task);original=copy.deepcopy(d);out=score_result(d)
            self.assertEqual(out['verbose']['M1']['raw_measurement'],original['verbose']['M1']['measurements'])
            self.assertEqual(out['normalization_source']['previous_metrics'],original['metrics'])
            self.assertIsNone(out['metrics']['M2']['extract_success'])
            self.assertIsNone(out['metrics']['M2']['metric'])
            self.assertIsNone(out['verbose']['M2'])
            self.assertAlmostEqual(out['proxy']['score'],out['metrics']['M1']['proxy_score'])
            self.assertEqual(score_result(copy.deepcopy(out)),out)
            self.assertEqual(json.loads(json.dumps(out,allow_nan=False)),out)
    def test_previous_v2_migration_preserves_metrics_and_recognition_once(self):
        for task in ('P34','P37','P38','P39','P41','P43','P49'):
            d=score_result(self.fixture(task))
            d['normalization_source']['version']=LEGACY_VERSION
            d['proxy'].update(version=LEGACY_VERSION,score=.5*d['metrics']['M1']['proxy_score'],
                              weights={'M1':.5,'M2':.5},score_status='partial',overall_proxy_valid=False)
            original=copy.deepcopy(d)
            out=score_result(d)
            self.assertEqual(out['metrics'],original['metrics'])
            self.assertEqual(out['verbose'],original['verbose'])
            self.assertEqual(out['normalization_source']['previous_metrics'],original['normalization_source']['previous_metrics'])
            self.assertEqual(out['normalization_source']['version'],VERSION)
            self.assertEqual(out['proxy']['score'],original['metrics']['M1']['proxy_score'])
            self.assertEqual(out['aggregation_history'][0]['previous_proxy'],original['proxy'])
            self.assertEqual(score_result(copy.deepcopy(out)),out)
    def test_scoring_evidence_uses_actual_weights(self):
        result=score_result(self.fixture('P38'))
        with TemporaryDirectory() as tmp:
            path=write_scoring_calculation(result,tmp)
            self.assertIn('综合分 = 1 × 0.575 = 0.575',path.read_text())
            self.assertIn('不计入分母',path.read_text())
    def test_residual_metric_is_original_error_not_final_score(self):
        d=self.fixture('P49');out=score_result(d);metric=out['metrics']['M1']
        self.assertAlmostEqual(metric['metric'],.6192128265565755)
        self.assertAlmostEqual(metric['physics_score'],1/(1+.6192128265565755))
    def test_nonresidual_mappings_preserved(self):
        for task in ('P34','P37','P38','P39'):
            d=self.fixture(task);q=d['metrics']['M1']['metric']
            self.assertEqual(score_result(d)['metrics']['M1']['physics_score'],q)
    def test_invalid_raw_measurements_are_traceable_not_scored(self):
        d=self.fixture('P49');d['verbose']['M1']['measurements']['raw_m1_error']=None
        out=score_result(d);m=out['metrics']['M1']
        self.assertIs(m['extract_success'],False);self.assertIsNone(m['metric'])
        self.assertIsNone(m['physics_score']);self.assertEqual(m['recognition_score'],0)
        self.assertEqual(out['proxy']['score'],0)
    def test_legacy_failed_values_never_receive_recognition(self):
        d=self.fixture('P49');d['metrics']['M1']['extract_success']=False
        out=score_result(d)
        self.assertIsNone(out['metrics']['M1']['metric'])
        self.assertEqual(out['metrics']['M1']['proxy_score'],0)
        self.assertIsNotNone(out['normalization_source']['previous_metrics']['M1']['metric'])
