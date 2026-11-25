"""Checks for exact interval replay and propagation-envelope construction."""
from fractions import Fraction as F
import numpy as np
from verified_sphere import I, trig, verify
from diffraction_model import DiffractionEnvelope
from comparison_models import comparisons


def test_verified_sphere_decisions_from_rational_endpoints():
    report=verify()
    for name,row in report['budgets'].items():
        lower=lambda key:F(row[key]['lower_rational'])
        upper=lambda key:F(row[key]['upper_rational'])
        assert lower('endpoint_minus')-upper('reference_plus')>F('.59')
        assert upper('one_box_UB')<=lower('reference_minus')+F('.3')
        assert lower('gain')<=lower('endpoint_minus')-upper('reference_plus')


def test_exact_interval_roots_and_signed_products():
    for x in [F(0),F(1,3),F(2),F(3),F(1001,997)]:
        y=I(x).sqrt()
        assert y.lo*y.lo<=x<=y.hi*y.hi
    assert (I(-2,3)*I(-5,-1))==I(-15,10)
    assert I(-2,3).square()==I(0,9)
    assert trig(0)==I(0)
    assert trig(0,True)==I(1)


def test_diffraction_extrema_caps_and_lipschitz():
    d=np.array([2.2,1.868,2.2,1.8]);torso=np.array([0,0,.027,0])
    env=DiffractionEnvelope(d,torso)
    assert env.upper_cap[2]<1
    x=np.linspace(-.1,.5,401)
    values=np.array([env.plus(np.full(4,c)) for c in x])
    assert np.all(np.diff(values,axis=0)>=-1e-14)
    assert np.all(np.diff(values,axis=0)<=np.diff(x)[:,None]*env.upper_slope+1e-14)
    for c in [-.05,0,.1,.3]:
        lo,hi=env.minus(np.full(4,c)),env.plus(np.full(4,c))
        for freq in np.linspace(6,8,7):
            for a in np.linspace(.15,.60,11):
                scale=np.sqrt(2*d/(.299792458/freq*a*(d-a)))
                for absorption in [50,75,100]:
                    truth=10**(-(env.loss(-c*scale)+absorption*torso)/10)
                    assert np.all(lo<=truth+1e-14) and np.all(truth<=hi+1e-14)


def test_paired_gain_and_absolute_optimum_are_distinct():
    a=comparisons()
    assert F(a['product']['absolute_optimum_width'])==F(2,15)
    assert a['paired']['gain_interval']==['0','0']
    assert a['paired']['exclusion_margin_0'] and not a['product']['exclusion_margin_0']


def test_rigid_frame_bound_with_moving_tag_and_fixed_anchors():
    from scipy.spatial.transform import Rotation
    from dynamic_standing import rigid_frame_motion_bound
    rng = np.random.default_rng(113)
    for _ in range(40):
        R0, R = Rotation.random(2, random_state=rng).as_matrix()
        t0, t = rng.normal(size=(2, 3))
        offset = rng.normal(size=3)
        anchors = rng.normal(size=(4, 3))
        bound = rigid_frame_motion_bound(t, R, t0, R0, anchors)
        for s in [0, .13, .71, 1]:
            nominal = (1-s)*(t0+R0@offset)+s*anchors
            actual = (1-s)*(t+R@offset)+s*anchors
            delta = (actual-t)@R-(nominal-t0)@R0
            assert np.max(np.linalg.norm(delta, axis=1)) <= s*bound+1e-12


def test_static_sensitivity_uses_held_offsets_and_preserves_budget_accounting():
    import json
    from g1_geometry import ROOT
    d = json.loads((ROOT/'poc/results/reviewer/dynamics_static/result.json').read_text())
    r = json.loads((ROOT/'poc/results/reviewer/static_sensitivity/summary.json').read_text())
    held = [t['phases'][p]['maxima'] for t in d['trials']
            for p in ['reference_hold', 'candidate_hold']]
    assert r['substituted_base_motion']['relative_geometry_m'] == max(x['nominal_segment_frame_bound_m'] for x in held)
    assert r['substituted_base_motion']['range_vector_m'] == max(x['nominal_tag_offset_m'] for x in held)
    assert r['substituted_base_motion']['range_vector_m'] > r['held_incremental_motion']['range_vector_m']
    assert r['held_nominal_obstruction']['S_minus_upper'] < r['held_nominal_obstruction']['improvement_threshold_at_002']
    assert r['held_nominal_obstruction']['box_upper_floor_from_saturated_qb'] > r['held_nominal_obstruction']['exclusion_threshold_at_036']
    for run in r['runs'].values():
        for row in run['point_widths'].values():
            assert abs(sum(row['decomposition'].values())-row['width']) < 1e-12
