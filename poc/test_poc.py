"""Consistency tests and empirical checks; none is a numerical certificate."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from bounds import Bounds, Envelope, decision, search
from g1_geometry import G1Geometry, hull_sdf, cylinder_sdf
from paper_example import SphereExample, paper_validation


def test_article_regression():
    result = paper_validation()
    assert result['A']['width_rhs'] > result['A']['refinements'][-1]['UB']-result['A']['refinements'][-1]['LB']


def test_hull_signed_distance_cube_faces_edges_corners():
    cube = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    pts = np.array([[0, 0, 0], [.9, .1, 0], [1, .3, .2], [2, 0, 0], [2, 2, 0], [2, 2, 2]])
    np.testing.assert_allclose(hull_sdf(pts, ConvexHull(cube)), [-1, -.1, 0, 1, np.sqrt(2), np.sqrt(3)], atol=1e-14)


def test_cylinder_signed_distance():
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 3], [2, 0, 3], [1, 0, 1]])
    np.testing.assert_allclose(cylinder_sdf(pts, 1, 2), [-1, 1, 1, np.sqrt(2), 0])


def test_decision_strict_and_nonstrict_thresholds():
    ref = dict(f_minus=.2, f_plus=.5)
    assert decision(.6, .8, ref, .1) == 'inconclusive'
    assert decision(.60001, .8, ref, .1) == 'guaranteed improvement'
    assert decision(.2, .3, ref, .1) == 'excluded improvement'


def test_invalid_envelopes_and_Q():
    with pytest.raises(ValueError):
        Envelope(lower_offset=-1)
    b = Bounds(SphereExample(), dict(N=10, e_phi=1e-4, e_lambda=1e-6,
             relative_geometry_m=.01, range_vector_m=.012, direction_offset=.002))
    with pytest.raises(ValueError):
        search(b, [0], [.1], [.2])
    with pytest.raises(ValueError):
        search(b, [0], [.1], [0], candidates=[[.2]])


def test_branch_bound_cover_and_monotonicity():
    b = Bounds(SphereExample(), dict(N=100, e_phi=1e-4, e_lambda=1e-6,
             relative_geometry_m=.01, range_vector_m=.012, direction_offset=.002))
    r = search(b, [0], [.05], [0], max_splits=15, tolerance=0)
    assert len(r['history']) == 16
    assert np.all(np.diff([v['LB'] for v in r['history']]) >= -1e-12)
    assert np.all(np.diff([v['UB'] for v in r['history']]) <= 1e-12)
    for q in np.linspace(0, .05, 101):
        assert b.at([q])['f_plus'] <= r['UB']+1e-12


def test_realization_interval_empirical():
    rng = np.random.default_rng(8)
    g = SphereExample()
    b = Bounds(g, dict(N=1000, e_phi=1e-4, e_lambda=1e-6,
              relative_geometry_m=.01, range_vector_m=.012, direction_offset=.002))
    for q in np.linspace(0, np.pi/6, 12):
        a = b.at([q])
        c = g.analytic(q)
        for _ in range(20):
            w = rng.uniform(b.env.minus(c), b.env.plus(c))
            perturbed = b.h+rng.normal(size=(4, 3))*1e-4
            J = np.einsum('i,ij,ik->jk', w, perturbed, perturbed)
            f = np.linalg.eigvalsh(J)[0]
            assert a['f_minus'] <= f <= a['f_plus']


@pytest.mark.parametrize('three_joints', [False, True])
def test_G1_fixed_tag_Q_and_geometric_Lq_empirical(three_joints):
    c = json.loads(Path(__file__).with_name('config.json').read_text())
    if three_joints:
        c['joints'].append('left_shoulder_roll_joint')
        c['Q_lower'].append(-.1)
        c['Q_upper'].append(.4)
        c['q0'].append(0.)
    g = G1Geometry(c)
    q = np.array(c['q0'])
    before = g.clearance(q, 32)
    tag, torso = g.robot.tag_position().copy(), g.robot.body_position('torso_link')
    dq = np.full(len(q), .001)
    after = g.clearance(q+dq, 32)
    np.testing.assert_allclose(g.robot.tag_position(), tag, atol=1e-15)
    np.testing.assert_allclose(g.robot.body_position('torso_link'), torso, atol=1e-15)
    assert np.all(abs(after-before) <= g.Lq*np.linalg.norm(dq))


def test_ceiling_precedes_candidate_generation():
    b = Bounds(SphereExample(), dict(N=100, e_phi=1e-4, e_lambda=1e-6,
        relative_geometry_m=.01, range_vector_m=.012, direction_offset=.002))
    def forbidden():
        raise AssertionError('ceiling must precede candidate generation')
    r = search(b, [0], [.05], [0], candidates=forbidden, margin=10)
    assert r['LB'] is None and r['evaluation_count'] == 1 and r['split_count'] == 0


def test_G1_pruning_witnesses_and_hard_tradeoff():
    c = json.loads(Path(__file__).with_name('config_hard.json').read_text())
    g = G1Geometry(c)
    b = Bounds(g, c['budget'], Envelope(**c['envelope']))
    assert np.linalg.matrix_rank(b.h[1:]) == 3
    a, z = b.at(c['q0']), b.at(c['Q_upper'])
    assert a['f_minus'] > 0 and z['f_minus'] > a['f_plus']+.02
    assert z['w_minus'][0] > a['w_minus'][0] and z['w_minus'][2] < a['w_minus'][2]
    for q in [c['q0'], c['Q_upper']]:
        pruned, hits = g.clearance(q, 256, witnesses=True)
        full = g.clearance(q, 256, prune=False)
        np.testing.assert_allclose(pruned, full, atol=1e-14)
        np.testing.assert_allclose([h['signed_distance_m'] for h in hits], full, atol=1e-14)
