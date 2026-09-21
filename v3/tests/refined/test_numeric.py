import unittest
import numpy as np
from refined_evaluators.numeric import saturated_ratio,oscillation_count_score,error_score,soft_error_score,fit_discharge_exponent,count_cycles,match_events,height_peak,require_track,terminal_windows
from refined_evaluators.common import ExtractionError
from refined_evaluators.definitions import DEFAULTS

class ScoringTests(unittest.TestCase):
    def test_ring_ratios_and_scale(self):
        for ratio,score in [(0,1),(.8,1),(.9,.5),(.95,.25),(1,0),(1.2,0)]:
            self.assertAlmostEqual(saturated_ratio(ratio,1,.2)[1],score)
            self.assertAlmostEqual(saturated_ratio(13*ratio,13,.2)[1],score)
    def test_cycles_score_anchors_and_scale(self):
        for solid,slotted,expected in [(0,5,1),(1,3,.75),(1,2,2/3),(1,1,.5),(2,1,1/3),(3,1,.25)]:
            ratio,score=oscillation_count_score(solid,slotted)
            self.assertAlmostEqual(ratio,solid/slotted);self.assertAlmostEqual(score,expected)
            self.assertEqual(oscillation_count_score(13*solid,13*slotted),(ratio,score))
    def test_cycles_score_monotonic_without_cliff_or_plateau(self):
        scores=[oscillation_count_score(n,100)[1] for n in [0,40,80,90,95,98,99,100,101,102,105,110,120,200]]
        self.assertTrue(all(a>b for a,b in zip(scores,scores[1:])))
        self.assertLess(abs(oscillation_count_score(99,100)[1]-.5),.01)
        self.assertLess(abs(oscillation_count_score(101,100)[1]-.5),.01)
    def test_cycles_zero_reference_is_not_neutral(self):
        for solid in [0,1,5]:self.assertEqual(oscillation_count_score(solid,0),(None,0.))
    def test_cycles_invalid_counts(self):
        for value in [-1,.5,float('inf'),float('nan')]:
            for counts in [(value,2),(2,value)]:
                with self.assertRaises(ValueError):oscillation_count_score(*counts)
    def test_zero_denominator(self):self.assertEqual(saturated_ratio(0,0,.2),(None,0))
    def test_invalid_margin(self):
        for v in [0,-1,1.01,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):saturated_ratio(1,2,v)
    def test_stokes_scores(self):
        for vr,score in [(4,1),(3,.75),(2,.5),(1,.25),(8,0)]:self.assertAlmostEqual(error_score(abs(vr/4-1)),score)

class SoftTimingTests(unittest.TestCase):
    def test_anchors_and_former_cutoff(self):
        for dt,expected in [(0,1),(.25,2/3),(.5,.5),(1,.3333333333333333),(1.875,.21052631578947367)]:
            self.assertAlmostEqual(soft_error_score(dt,.5),expected)
    def test_monotonic_positive_and_time_units(self):
        dt=np.array([0,.1,.499,.5,.501,1,2,5,100.])
        scores=[soft_error_score(x,.5) for x in dt]
        self.assertTrue(all(a>b>0 for a,b in zip(scores,scores[1:])))
        for x in dt:self.assertAlmostEqual(soft_error_score(x,.5),soft_error_score(x*1000,500))
    def test_invalid_error_and_scale(self):
        for x in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):soft_error_score(x,.5)
        for x in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):soft_error_score(1,x)

class DischargeFitTests(unittest.TestCase):
    def test_narrow_height_at_initial_fill_can_fit_free_exponent(self):
        t=np.arange(0,4,.25);h=np.linspace(100,99,len(t))
        for beta in [.5,0,-4,2]:
            q=3*h**beta;result=fit_discharge_exponent(h,q,q*.01,t,DEFAULTS['P41'])
            self.assertEqual(result['failures'],[])
            self.assertAlmostEqual(result['fit']['velocity'],beta,places=7)
            self.assertEqual(len(result['independent_fit_frames']),len(t))
    def test_volume_units_do_not_change_beta(self):
        t=np.arange(0,4,.25);h=np.linspace(100,90,len(t));q=2*np.sqrt(h)
        a=fit_discharge_exponent(h,q,q*.01,t,DEFAULTS['P41'])
        b=fit_discharge_exponent(h*10,q*1000,q*10,t,DEFAULTS['P41'])
        self.assertAlmostEqual(a['fit']['velocity'],b['fit']['velocity'])
    def test_zero_discharge_and_constant_height_remain_undefined(self):
        t=np.arange(0,4,.25)
        for h,q in [(100-t,0*t),(t*0+100,t*0+3)]:
            result=fit_discharge_exponent(h,q,0*t,t,DEFAULTS['P41'])
            self.assertIsNone(result['fit']);self.assertTrue(result['failures'])
    def test_correlated_windows_and_rejected_flow_are_reported(self):
        t=np.arange(0,4,.05);h=100-t;q=2*np.sqrt(h);q[0]=0;q[1]=-1
        result=fit_discharge_exponent(h,q,np.ones(len(t))*.1,t,DEFAULTS['P41'])
        ids=result['independent_fit_frames']
        self.assertTrue(np.all(np.diff(t[ids])>=.25-1e-9))
        self.assertEqual(result['frame_selection_reasons'][:2],['nonpositive_discharge']*2)
        self.assertIn('correlated_with_selected_window',result['frame_selection_reasons'])
        self.assertIsNotNone(result['fit'])

class PeriodTests(unittest.TestCase):
    def setUp(self):self.t=np.linspace(0,4.2,421)
    def test_nonoverlapping_counts(self):
        p,c,tail=count_cycles(10*np.cos(2*np.pi*self.t),self.t,1,.5,.1,True)
        self.assertEqual(len(c),4);self.assertEqual(c[0]['end_frame'],c[1]['start_frame'])
    def test_release_is_not_fabricated(self):
        p,c,_=count_cycles(10*np.cos(2*np.pi*self.t),self.t,1,.5,.1,False);self.assertEqual(len(c),3)
    def test_strong_damping_without_return(self):
        p,c,_=count_cycles(10*np.exp(-8*self.t),self.t,1,.5,.1,True);self.assertEqual(len(c),0)
    def test_noise_only(self):
        p,c,_=count_cycles(.05*np.sin(80*self.t),self.t,1,.5,.1,False);self.assertEqual(len(c),0)
    def test_half_period_tail(self):
        t=np.linspace(0,2.5,251);p,c,tail=count_cycles(10*np.cos(2*np.pi*t),t,1,.5,.1,True);self.assertEqual(len(c),2)
    def test_shared_cutoff(self):
        _,c,_=count_cycles(.9*np.cos(2*np.pi*self.t),self.t,1,.5,.1,True);self.assertEqual(len(c),0)
    def test_long_occlusion(self):
        valid=np.ones(len(self.t),bool);valid[100:150]=False
        with self.assertRaises(ExtractionError):require_track(valid,self.t,.1,.5)

class PeakTests(unittest.TestCase):
    def test_rising_tail_is_censored(self):
        t=np.linspace(0,1,25);p=height_peak(t,t,.01,.12);self.assertTrue(p['right_censored'])
    def test_plateau_is_measured(self):
        t=np.linspace(0,2,49);p=height_peak(np.minimum(t,1),t,.01,.12);self.assertFalse(p['right_censored']);self.assertEqual(p['height'],1)
    def test_downward_leg_proves_peak(self):
        t=np.linspace(0,2,49);p=height_peak(1-(t-1)**2,t,.01,.12);self.assertFalse(p['right_censored'])
    def test_static_is_zero(self):
        t=np.linspace(0,2,49);p=height_peak(np.zeros(len(t)),t,.01,.12);self.assertFalse(p['right_censored']);self.assertEqual(p['height'],0)

class EventTests(unittest.TestCase):
    def test_global_matching_beats_greedy(self):
        pairs,tp,fp,fn,score=match_events([1,1.6],[.6,1.1],.5);self.assertEqual((tp,fp,fn,score),(2,0,0,1))
    def test_one_flash_cannot_match_twice(self):
        pairs,tp,fp,fn,score=match_events([1,1.5],[1.2],.5);self.assertEqual(tp,1);self.assertAlmostEqual(score,2/3)
    def test_tolerance_boundaries(self):
        self.assertEqual(match_events([1],[1.5],.5)[-1],1)
        self.assertEqual(match_events([1],[1.5001],.5)[-1],0)
        self.assertEqual(match_events([1],[.5],.5)[-1],1)
    def test_extra_flash(self):self.assertEqual(match_events([1,2],[1,2,4])[-1],.8)
    def test_dark_and_no_events(self):
        self.assertEqual(match_events([1,2],[])[-1],0);self.assertEqual(match_events([],[])[-1],0)

class TerminalTests(unittest.TestCase):
    def test_longest_independent_terminal_window(self):
        t=np.linspace(0,3,73);r=np.full(len(t),10.);y=30*t;w=terminal_windows(y,r,t,np.ones(len(t),bool),DEFAULTS['P49']);self.assertEqual(w[0]['start_frame'],0);self.assertEqual(w[0]['end_frame'],72);self.assertAlmostEqual(w[0]['velocity'],30)
    def test_acceleration_is_not_terminal(self):
        t=np.linspace(0,1,25);w=terminal_windows(200*t*t,np.ones(len(t))*10,t,np.ones(len(t),bool),DEFAULTS['P49']);self.assertEqual(w,[])
    def test_stopped_or_upward_not_settling(self):
        t=np.linspace(0,3,73)
        for y in [0*t,-30*t]:self.assertEqual(terminal_windows(y,np.full(len(t),10.),t,np.ones(len(t),bool),DEFAULTS['P49']),[])
    def test_acceleration_then_deceleration_not_hidden_by_long_fit(self):
        t=np.linspace(0,3,73);y=30*t+4*np.sin(2*np.pi*t/3)
        w=terminal_windows(y,np.full(len(t),5.),t,np.ones(len(t),bool),DEFAULTS['P49'])
        self.assertFalse(any(x['start_frame']==0 and x['end_frame']==72 for x in w))
    def test_boundary_and_occlusion_exclusion(self):
        t=np.linspace(0,3,73);v=np.ones(len(t),bool);v[45:]=False;w=terminal_windows(30*t,np.full(len(t),10.),t,v,DEFAULTS['P49']);self.assertEqual(w[0]['end_frame'],44)

if __name__=='__main__':unittest.main()
