"""Checker for article Section 6.3 "Radio and torso-motion sensitivity".

Every number and claim of that subsection (and of the appendix paragraph
"Derivation of the G1 local profile") is asserted against the value printed
in the article. One PASS/FAIL line is printed per claim; the exit status is
non-zero if any claim fails.

Three sources are used:
  stored  - the authors' original files in poc/results/reviewer (recorded
            with Python 3.12.10, NumPy 1.26.4, SciPy 1.14.1, MuJoCo 3.2.6);
  replay  - poc/results/reviewer/replay_sensitivity, written here by
            poc/replay_sensitivity.py (pinned Menagerie model, plain MuJoCo,
            no humanoid-uwb-sim);
  live    - quantities recomputed inside this checker (model formulas,
            profile, chain radii, convexity, omission rule).
A printed value is accepted only if BOTH the stored and the replayed value
round to it (nearest last printed digit) and the two agree within
REPLAY_TOL. Nothing replays bit-for-bit here: the standing simulations
differ by about 4e-14 m, and the convex hulls of the collision meshes
(SciPy/Qhull version) shift clearances by about 3e-9 m, which the envelope
slopes magnify to at most 4e-8 in the weights; hence the explicit tolerance.

Usage (from the repository root):
    G1_XML=g1_model/g1.xml PYTHONPATH=poc/menagerie_shim \
        python poc/check_article_sensitivity.py
    ... python -m pytest poc/check_article_sensitivity.py -q
"""
import functools
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_geometry import G1Geometry, ROOT  # noqa: E402

STORED = ROOT/'poc/results/reviewer'
REPLAY = STORED/'replay_sensitivity'
REPLAY_TOL = 1e-7          # absolute, replay against stored (dimensionless or SI units)
TAU = 0.0081               # rad, article: tau_iota
SEC = 'Sec. 6.3'
P_DIFF, P_STAND, P_LOCAL1, P_LOCAL2, P_REPRO = (
    SEC+' par. "Diffraction model"', SEC+' par. "Standing robot"',
    SEC+' par. "Local bounds along the segment"', SEC+' par. "With local bounds at margin 0.02"',
    SEC+' final par. (repository)')
APP = 'App. A "Derivation of the G1 local profile"'
CLAIMS = []


def claim(location, text):
    def register(function):
        CLAIMS.append((location, text, function))
        return function
    return register


@functools.lru_cache(None)
def load(root, name):
    return json.loads((Path(root)/name).read_text())


def both(name, getter, replay_name=None):
    """Value from the stored original and from the replay."""
    return getter(load(STORED, name)), getter(load(REPLAY, replay_name or name))


def rounds_to(value, printed, scale=1.):
    """True if scale*value rounds to the printed decimal string (nearest last digit)."""
    digits = len(printed.split('.')[1]) if '.' in printed else 0
    return abs(scale*value-float(printed)) <= .5*10.**-digits*(1+1e-9)


def printed(pair, text, scale=1.):
    """Stored and replayed value both round to the printed value and agree."""
    stored, replay = pair
    assert rounds_to(stored, text, scale), f'stored {stored!r} does not round to {text}'
    assert rounds_to(replay, text, scale), f'replay {replay!r} does not round to {text}'
    assert abs(stored-replay) <= REPLAY_TOL, f'replay differs from stored by {abs(stored-replay):.3e}'
    return f'{text}: stored {stored:.10g}, replay diff {replay-stored:+.1e}'


def same(pair, expected):
    stored, replay = pair
    assert stored == expected, f'stored {stored!r} != {expected!r}'
    assert replay == expected, f'replay {replay!r} != {expected!r}'
    return f'{expected!r} (stored and replay)'


@functools.lru_cache(None)
def config():
    return json.loads((ROOT/'poc/config_hard.json').read_text())


@functools.lru_cache(None)
def geometry():
    return G1Geometry(config())


@functools.lru_cache(None)
def diffraction():
    from diffraction_model import DiffractionEnvelope, torso_intersections
    g = geometry()
    torso = torso_intersections(g)
    return DiffractionEnvelope(np.linalg.norm(g.anchors-g.p, axis=1), torso), torso


@functools.lru_cache(None)
def local_bounds(prune=True):
    from local_bounds import LocalBounds
    budget = load(STORED, 'local/summary.json')['budget']
    g = G1Geometry(dict(config(), budget=budget))
    return LocalBounds(g, budget, diffraction()[0], tracking_rad=TAU, prune=prune)


def held(result, phase, key):
    return max(t['phases'][phase]['maxima'][key] for t in result['trials'])


# --------------------------------------------------------------- diffraction
def article_loss(v):
    return np.maximum(0, 6.9+(20/np.log(10))*np.arcsinh(v-.1))


def chi(d, F, a):
    return np.sqrt(2*d/((.299792458/F)*a*(d-a)))


@claim(P_DIFF, 'F in [6,8] GHz, a in [0.15,0.60] m, lambda_F=0.299792458/F, chi_i(F,a) extremes')
def diffraction_parameters():
    env, _ = diffraction()
    assert tuple(env.frequency_GHz) == (6., 8.) and tuple(env.edge_distance_m) == (.15, .60)
    F, a = np.meshgrid(np.linspace(6, 8, 81), np.linspace(.15, .60, 181))
    worst = 0.
    for i, d in enumerate(env.lengths):
        values = chi(d, F, a)
        worst = max(worst, abs(values.min()-env.scales[i, 0]), abs(values.max()-env.scales[i, 1]))
    assert worst < 1e-12
    assert np.allclose(env.lengths, [2.2, np.linalg.norm([1.8, -.3, -.4]), 2.2, 1.8], atol=1e-12)
    return f'grid extremes of chi_i equal the two scales of the program (max diff {worst:.1e})'


@claim(P_DIFF, 'D(nu)=max{0, 6.9+(20/ln10) arsinh(nu-0.1)} on the whole real line, no rounded cut-off')
def diffraction_loss():
    env, _ = diffraction()
    v = np.linspace(-50, 50, 200001)
    assert np.array_equal(env.loss(v), article_loss(v))
    zero = .1-np.sinh(6.9*np.log(10)/20)          # exact root, not the rounded ITU -0.78
    assert env.loss(np.array([zero-1e-9]))[0] == 0 and 0 < env.loss(np.array([zero+1e-6]))[0] < 1e-4
    assert env.loss(np.array([-.78]))[0] > 0      # the rounded cut-off would give a jump here
    assert np.max(np.abs(np.diff(env.loss(np.linspace(zero-.01, zero+.01, 20001))))) < 1e-4
    return f'identical on 200001 points; continuous zero at nu={zero:.6f}'


@claim(P_DIFF, 'alpha in [50,100] dB/m; envelopes = min/max of 10^{-[D(-c chi)+alpha t]/10} over F, a, alpha '
               '(equivalently: extreme chi, factors 10^{-100t/10} and 10^{-50t/10})')
def diffraction_envelopes():
    env, torso = diffraction()
    assert tuple(env.torso_loss_dB_per_m) == (50., 100.)
    F, a, alpha = np.meshgrid(np.linspace(6, 8, 21), np.linspace(.15, .60, 46), [50., 75., 100.], indexing='ij')
    worst = 0.
    for c in np.linspace(-.1, .4, 101):
        family = np.stack([10**(-(article_loss(-c*chi(d, F, a))+alpha*t)/10)
                           for d, t in zip(env.lengths, torso)])
        low, high = family.reshape(4, -1).min(axis=1), family.reshape(4, -1).max(axis=1)
        value = np.full(4, c)
        worst = max(worst, np.max(np.abs(env.minus(value)-low)), np.max(np.abs(env.plus(value)-high)))
        equivalent = np.stack([10**(-article_loss(-c*s)/10) for s in env.scales.T])
        assert np.allclose(env.minus(value), equivalent.min(axis=0)*10**(-100*torso/10), rtol=0, atol=1e-15)
        assert np.allclose(env.plus(value), equivalent.max(axis=0)*10**(-50*torso/10), rtol=0, atol=1e-15)
    assert worst < 1e-12
    return f'family min/max over a 21x46x3 grid equals the envelopes at 101 clearances (max diff {worst:.1e})'


@claim(P_DIFF, 't_3 ~ 0.02695714 m and the other t_i are zero')
def torso_path():
    _, torso = diffraction()
    get = lambda r: r['torso_intersection_lengths_m']
    stored, replay = both('physical_sensitivity.json', get, 'physical_sensitivity_inputs.json')
    assert stored[0] == stored[1] == stored[3] == 0 and replay[0] == replay[1] == replay[3] == 0
    assert torso[0] == torso[1] == torso[3] == 0 and abs(torso[2]-stored[2]) <= REPLAY_TOL
    return printed((stored[2], replay[2]), '0.02695714')


@claim(P_DIFF, 'upper weight cap 0.7332 for link 3')
def upper_cap():
    env, torso = diffraction()
    assert abs(env.upper_cap[2]-10**(-50*torso[2]/10)) < 1e-15 and np.all(np.delete(env.upper_cap, 2) == 1)
    return printed(both('physical_sensitivity.json', lambda r: r['link_caps_upper'][2],
                        'physical_sensitivity_inputs.json'), '0.7332')


@claim(P_DIFF, 'torso factor held fixed during the motion-sensitivity study')
def torso_fixed():
    cap = load(STORED, 'physical_sensitivity.json')['link_caps_upper'][2]
    count = 0
    for root in (STORED, REPLAY):
        for name in ['static_sensitivity/diffraction_gain.json', 'local/diffraction_local.json',
                     'local/diffraction_tracked.json', 'diffraction_assumed_2mm.json']:
            for a in load(root, name)['evaluations']:
                count += 1
                assert a['w_plus'][2] <= cap+REPLAY_TOL
                assert a['w_minus'][2] <= load(STORED, 'physical_sensitivity.json')['link_caps_lower'][2]+REPLAY_TOL
    return f'one constant torso factor: link-3 weights never exceed the nominal caps in {count} evaluations'


@claim(P_DIFF, 'with the 2 mm budget: LB ~ 0.3641')
def diffraction_lb():
    name = 'diffraction_assumed_2mm.json'
    same(both(name, lambda r: (r['budget']['relative_geometry_m'], r['budget']['range_vector_m'])), (.002, .004))
    same(both(name, lambda r: r['selected']['q']), [-.3, 1.45])
    return printed(both(name, lambda r: r['LB']), '0.3641')


@claim(P_DIFF, 'f+(q0)+0.02 ~ 0.2045')
def diffraction_threshold():
    name = 'diffraction_assumed_2mm.json'
    same(both(name, lambda r: r['reference']['q']), [-.6, 1.])
    same(both(name, lambda r: r['improvement_threshold'] == r['reference']['f_plus']+.02), True)
    return printed(both(name, lambda r: r['improvement_threshold']), '0.2045')


@claim(P_DIFF, 'LB > f+(q0)+0.02: conditional guaranteed improvement at margin 0.02')
def diffraction_decision():
    name = 'diffraction_assumed_2mm.json'
    same(both(name, lambda r: r['LB'] > r['improvement_threshold']), True)
    same(both(name, lambda r: r['epsilon_g']), .02)
    return same(both(name, lambda r: r['theorem_outcome']), 'guaranteed improvement')


# ------------------------------------------------------------ standing robot
DYN = 'dynamics_static/result.json'


@claim(P_STAND, 'five standing simulations differing only in small random initial base velocities')
def standing_trials():
    same(both(DYN, lambda r: len(r['trials'])), 5)
    same(both(DYN, lambda r: (r['seed'], r['timestep_s'], r['settling_s'], r['observation_s'])),
         (20261011, .002, 5, 6))
    source = (ROOT/'poc/dynamic_standing.py').read_text()
    assert 'data.qvel[:6] = 0 if trial == 0 else rng.uniform(-.002, .002, 6)' in source
    assert source.count('rng.') == 1, 'the random generator must only draw the base velocities'
    return '5 trials; the only random draw is the initial base velocity (trial 0: zero, others within +-0.002)'


@claim(P_STAND, 'released base, floor contact, stiff position actuators (gain 500 N m/rad), no balance feedback')
def standing_model_claim():
    import hashlib
    import mujoco
    from humanoid_uwb_sim.standing import standing_model
    model, path = standing_model(.002)
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.neq == 0
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'experiment_floor')
    assert floor >= 0 and model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.nu == 29 and np.all(model.actuator_gainprm[:, 0] == 500)
    assert np.all(model.actuator_biasprm[:, 1] == -500)       # position servo: kp*(ctrl-q)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    same(both(DYN, lambda r: r['model_sha256']), sha)
    source = (ROOT/'poc/dynamic_standing.py').read_text()
    import re
    assert len(re.findall(r'data\.ctrl\[[^\]]*\] = ', source)) == 3, 'controls are joint targets only'
    assert Path(sys.modules['humanoid_uwb_sim'].__file__).resolve().is_relative_to(ROOT/'poc/menagerie_shim')
    return 'free joint, added plane, 29 position actuators with kp=500, pinned g1.xml (sha256 as stored), plain MuJoCo'


@claim(P_STAND, 'the arm moves from q0 to qb of the full-rank configuration')
def standing_arm():
    assert config()['q0'] == [-.6, 1.] and config()['Q_upper'] == [-.3, 1.45]
    source = (ROOT/'poc/dynamic_standing.py').read_text()
    assert "arm0, arm1 = np.array(config['q0']), np.array(config['Q_upper'])" in source
    same(both('local/summary.json', lambda r: (r['reference'], r['candidate'])), ([-.6, 1.], [-.3, 1.45]))
    return same(both(DYN, lambda r: (r['arm_ramp_s'], r['static_windows_s'])),
                ([1, 3], dict(reference_hold=[0, 1], candidate_hold=[5, 6])))


@claim(P_STAND, 'held q0: tag up to 22.67 mm from its kinematic nominal position')
def standing_tag_q0():
    return printed(both(DYN, lambda r: held(r, 'reference_hold', 'nominal_tag_offset_m')), '22.67', 1e3)


@claim(P_STAND, 'held qb: tag up to 16.24 mm')
def standing_tag_qb():
    return printed(both(DYN, lambda r: held(r, 'candidate_hold', 'nominal_tag_offset_m')), '16.24', 1e3)


@claim(P_STAND, 'rigid-motion bound of the anchors in the torso frame reaches 137.1 mm (q0)')
def standing_frame_q0():
    return printed(both(DYN, lambda r: held(r, 'reference_hold', 'nominal_segment_frame_bound_m')), '137.1', 1e3)


@claim(P_STAND, 'rigid-motion bound reaches 95.85 mm (qb)')
def standing_frame_qb():
    return printed(both(DYN, lambda r: held(r, 'candidate_hold', 'nominal_segment_frame_bound_m')), '95.85', 1e3)


def below(key, limit, scale=1e3):
    get = lambda r: max(held(r, p, key) for p in ['reference_hold', 'candidate_hold'])
    stored, replay = both(DYN, get)
    assert stored*scale < limit and replay*scale < limit and abs(stored-replay) <= REPLAY_TOL
    return f'max over both holds and five runs: stored {stored*scale:.6g} < {limit}, replay diff {replay-stored:+.1e}'


@claim(P_STAND, 'drift within each hold below 0.12 mm for the tag')
def standing_drift_tag():
    return below('within_phase_tag_drift_m', .12)


@claim(P_STAND, 'drift within each hold below 0.51 mm for the rigid-motion bound')
def standing_drift_frame():
    return below('within_phase_frame_drift_bound_m', .51)


@claim(P_STAND, 'joint tracking errors below 8.1 mrad')
def standing_tracking():
    get = lambda r: max(t['phases'][p]['maxima']['controlled_joint_target_error_rad']
                        for t in r['trials'] for p in t['phases'])
    stored, replay = both(DYN, get)
    assert stored < TAU and replay < TAU and abs(stored-replay) <= REPLAY_TOL
    below('controlled_joint_target_error_rad', 8.1)
    return f'max over all phases and runs: stored {stored*1e3:.6g} mrad < 8.1, replay diff {replay-stored:+.1e}'


SUMMARY = 'static_sensitivity/summary.json'


@claim(P_STAND, 'uniform budgets e_i ~ 0.02267 m and e_fr ~ 0.1371 m (maxima of the held postures)')
def uniform_budgets():
    for root in (STORED, REPLAY):
        d, s = load(root, DYN), load(root, SUMMARY)['substituted_base_motion']
        assert s['range_vector_m'] == d['static_nominal_tag_budget_m'] == max(
            held(d, p, 'nominal_tag_offset_m') for p in ['reference_hold', 'candidate_hold'])
        assert s['relative_geometry_m'] == d['static_nominal_frame_budget_m'] == max(
            held(d, p, 'nominal_segment_frame_bound_m') for p in ['reference_hold', 'candidate_hold'])
        for name in ['restricted_gain', 'restricted_exclusion', 'diffraction_gain']:
            b = load(root, f'static_sensitivity/{name}.json')['budget']
            assert (b['range_vector_m'], b['relative_geometry_m']) == (s['range_vector_m'], s['relative_geometry_m'])
    return (printed(both(SUMMARY, lambda r: r['substituted_base_motion']['range_vector_m']), '0.02267')+'; '
            + printed(both(SUMMARY, lambda r: r['substituted_base_motion']['relative_geometry_m']), '0.1371'))


@claim(P_STAND, 'both full-rank comparisons and the diffraction comparison inconclusive after 32 subdivisions')
def uniform_inconclusive():
    expected = {'restricted_gain': ('config_hard.json', .02), 'restricted_exclusion': ('config_bnb.json', .36),
                'diffraction_gain': ('config_hard.json', .02)}
    for name, (file, margin) in expected.items():
        same(both(f'static_sensitivity/{name}.json', lambda r: (r['config_file'], r['epsilon_g'])), (file, margin))
        same(both(SUMMARY, lambda r: (r['runs'][name]['theorem_outcome'], r['runs'][name]['split_count'],
                                      r['runs'][name]['stop_reason'])), ('inconclusive', 32, 'split limit'))
    return 'margins 0.02 and 0.36 (clip envelopes) and 0.02 (diffraction): inconclusive, split limit 32'


@claim(P_STAND, 'clip envelopes: [LB,UB] ~ [0.04921, 0.5988]')
def uniform_interval():
    for name in ['restricted_gain', 'restricted_exclusion']:
        text = (printed(both(SUMMARY, lambda r: r['runs'][name]['LB']), '0.04921')+'; '
                + printed(both(SUMMARY, lambda r: r['runs'][name]['UB']), '0.5988'))
    return text


@claim(P_STAND, 'in (12), the part of sum_i(Lambda_i^+ + Lambda_i^-) eps_c,i due to L^eta eta grows from 0.0720 ...')
def width_term_assumed():
    terms = lambda r: r['width_terms']['restricted_gain']['assumed_budget']['relative_geometry']
    live = 4*(5.+4.)*config()['budget']['relative_geometry_m']       # sum_i (Lambda+ + Lambda-) * 0.002
    assert abs(live-load(STORED, SUMMARY)['width_terms']['restricted_gain']['assumed_budget']['relative_geometry']) < 1e-15
    return printed(both(SUMMARY, terms), '0.0720')


@claim(P_STAND, '... to 4.936')
def width_term_held():
    terms = lambda r: r['width_terms']['restricted_gain']['held_nominal_budget']['relative_geometry']
    for root in (STORED, REPLAY):
        s = load(root, SUMMARY)
        assert abs(4*9.*s['substituted_base_motion']['relative_geometry_m']-terms(s)) < 1e-12
    return printed(both(SUMMARY, terms), '4.936')


@claim(P_STAND, 'Gamma_inf <= 0.3017 is unchanged; the geometric contribution alone substantially exceeds it')
def gamma_bound():
    gamma = lambda r: r['width_terms']['restricted_gain']['Gamma_interval'][1]
    text = printed(both(SUMMARY, gamma), '0.3017')
    for root in (STORED, REPLAY):
        w = load(root, SUMMARY)['width_terms']['restricted_gain']
        assert w['held_nominal_budget']['relative_geometry'] > 10*w['Gamma_interval'][1]
        assert w['Gamma_interval'][0] == 0
    return text+' (budget-independent: zero-budget ceiling minus best nominal lower index); 4.936 > 10 x 0.3017'


# --------------------------------------------------------------- local bounds
@claim(P_LOCAL1, 'torso-frame segment point (1-s)p_tor+s R_tor^T(a_i-t_tor): a rigid torso motion moves it by at most s e_fr')
def rigid_motion_bound():
    from scipy.spatial.transform import Rotation
    from dynamic_standing import rigid_frame_motion_bound
    rng = np.random.default_rng(63)
    worst = -np.inf
    for _ in range(200):
        R0, R = Rotation.random(2, random_state=rng).as_matrix()
        t0, t = rng.normal(size=(2, 3))
        p_tor, anchors = rng.normal(size=3), rng.normal(size=(4, 3))
        e_fr = rigid_frame_motion_bound(t, R, t0, R0, anchors)
        for s in np.linspace(0, 1, 11):
            nominal = (1-s)*p_tor+s*(anchors-t0)@R0
            actual = (1-s)*p_tor+s*(anchors-t)@R
            worst = max(worst, np.max(np.linalg.norm(actual-nominal, axis=1)-s*e_fr))
    assert worst <= 1e-12
    return f'200 random rigid motions, 11 values of s, 4 anchors: max(displacement - s e_fr) = {worst:.1e}'


def independent_profile(b, s, active):
    """zeta_il(s) of the article, written out independently of local_bounds.profile."""
    g = b.geometry
    m, d = g.robot.model, g.robot.data
    torso = g.robot._bid['torso_link']
    R0, t0 = d.xmat[torso].reshape(3, 3), d.xpos[torso]
    reach, origin_error = chain_radii(b)
    zeta = np.zeros((4, len(s)))
    gap = 0.
    for i in range(4):
        r = g.p[None, :]+s[:, None]*(g.anchors[i]-g.p)[None, :]
        zeta[i] = s*b.b['relative_geometry_m']
        for j, tau in zip(b.jids, b.tau):
            if j in active:
                o = d.xanchor[j]
                world = np.linalg.norm(r-o, axis=1)
                torso_frame = np.linalg.norm((r-t0)@R0-(o-t0)@R0, axis=1)
                gap = max(gap, np.max(np.abs(world-torso_frame)))
                zeta[i] += tau*(world+origin_error[j])
    return zeta, gap


def chain_radii(b):
    """o-bar (chain radii of Eq. (g1Lq)) and e^o, recomputed from the model."""
    m = b.geometry.robot.model
    torso = b.geometry.robot._bid['torso_link']
    reach, parents = {}, {}
    for j in b.jids:
        body = int(m.jnt_bodyid[j])
        reach[j] = float(np.linalg.norm(m.jnt_pos[j]))
        chain = []
        while body != torso:
            reach[j] += float(np.linalg.norm(m.body_pos[body]))
            chain.append(body)
            body = int(m.body_parentid[body])
        parents[j] = chain[1:]                       # proper ancestor bodies below the torso
    error = {j: sum(tau*(reach[j]+reach[k]) for k, tau in zip(b.jids, b.tau)
                    if k != j and int(m.jnt_bodyid[k]) in parents[j]) for j in b.jids}
    return reach, error


@claim(P_LOCAL1, 'zeta_il(s) = s e_fr + sum_{iota in J_l} tau_iota(||r_i(s,0)-o_iota(q)|| + e_iota^o), nominal world frame; '
                 'distances unchanged in the nominal torso frame')
def profile_formula():
    b = local_bounds()
    s = np.linspace(0, 1, 257)
    worst = gap = 0.
    for q in [config()['q0'], config()['Q_upper'], [-.45, 1.225]]:
        b.geometry.pose(q)
        for _, _, active in b.geometry.geoms:
            zeta, g = independent_profile(b, s, active)
            worst = max(worst, np.max(np.abs(zeta-b.profile(s, active))))
            gap = max(gap, g)
    assert worst < 1e-14 and gap < 1e-12
    assert b.b['relative_geometry_m'] == load(STORED, DYN)['static_nominal_frame_budget_m']
    return f'program profile equals the formula (max diff {worst:.1e}); world/torso-frame distance gap {gap:.1e}'


@claim(P_LOCAL1, 'e_iota^o = sum_{iota\'} tau_iota\'(o-bar_iota + o-bar_iota\') over preceding controlled joints; chain radii of (g1Lq)')
def origin_displacement_bound():
    b = local_bounds()
    reach, error = chain_radii(b)
    names = b.geometry.joints
    for j, name in zip(b.jids, names):
        assert abs(reach[j]-b.geometry.constants['joint_reach_m'][name]) < 1e-15
        assert abs(error[j]-b.origin_errors[j]) < 1e-15
    shoulder, elbow = b.jids
    assert error[shoulder] == 0 and abs(error[elbow]-TAU*(reach[elbow]+reach[shoulder])) < 1e-15
    # The bound holds throughout the joint-error paths q+t*dq (sampled check).
    g = b.geometry
    worst = 0.
    for q in [config()['q0'], config()['Q_upper']]:
        g.pose(q)
        nominal = g.robot.data.xanchor[elbow].copy()
        for dq in [(-TAU, -TAU), (-TAU, TAU), (TAU, -TAU), (TAU, TAU)]:
            for t in np.linspace(0, 1, 11):
                g.pose(np.asarray(q)+t*np.asarray(dq))
                worst = max(worst, np.linalg.norm(g.robot.data.xanchor[elbow]-nominal)/error[elbow])
    assert worst <= 1
    return (f'o-bar = ({reach[shoulder]:.6f}, {reach[elbow]:.6f}) m, e^o = (0, {error[elbow]:.6e}) m; '
            f'sampled origin displacement uses at most {100*worst:.1f}% of e^o')


@claim(APP, 'joint errors |dq| <= tau move a segment point, seen from body l, by at most the joint part of zeta')
def joint_part_of_profile():
    b = local_bounds()
    g = b.geometry
    d = g.robot.data
    s = np.linspace(0, 1, 257)
    ray = g.p[None, None, :]+s[None, :, None]*(g.anchors-g.p)[:, None, :]
    worst = 0.
    for q in [config()['q0'], config()['Q_upper']]:
        g.pose(q)
        nominal = {gid: (ray-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3, 3) for gid, _, _ in g.geoms}
        bound = {gid: b.profile(s, active)-s*b.b['relative_geometry_m'] for gid, _, active in g.geoms}
        for dq in [(-TAU, -TAU), (-TAU, TAU), (TAU, -TAU), (TAU, TAU)]:
            g.pose(np.asarray(q)+dq)
            for gid, _, _ in g.geoms:
                moved = (ray-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3, 3)
                worst = max(worst, np.max(np.linalg.norm(moved-nominal[gid], axis=2)/bound[gid]))
    assert worst <= 1
    return f'7 bodies x 4 links x 257 nodes x 4 error vertices at q0 and qb: largest displacement/bound ratio {worst:.6f} <= 1'


@claim(P_LOCAL1, 'the function is convex in s, so its maximum on a cell is attained at an endpoint')
def profile_convex():
    b = local_bounds()
    N = int(b.b['N'])
    s = np.linspace(0, 1, 16*N+1)
    worst = 0.
    for q in [config()['q0'], config()['Q_upper']]:
        b.geometry.pose(q)
        for _, _, active in b.geometry.geoms:
            zeta = b.profile(s, active)
            worst = min(worst, np.min(zeta[:, 2:]-2*zeta[:, 1:-1]+zeta[:, :-2]))
            # Voronoi cells of the nodes k/N: endpoints at (k +- 1/2)/N, clipped to [0,1].
            nodes = np.arange(N+1)*16
            left, right = np.maximum(nodes-8, 0), np.minimum(nodes+8, 16*N)
            for k in range(N+1):
                inside = zeta[:, left[k]:right[k]+1]
                assert np.all(inside.max(axis=1) <= np.maximum(zeta[:, left[k]], zeta[:, right[k]])+1e-15)
    assert worst > -1e-13
    return f'second differences >= {worst:.1e} on a 16x finer grid; cell maxima equal endpoint maxima'


@claim(P_LOCAL1, 'tau_iota = 0.0081 rad at the reference and the candidate; realized joint angles within the joint limits')
def tracking_bound():
    import mujoco
    same(both('local/summary.json', lambda r: r['tracking_per_coordinate_rad']), TAU)
    same(both('local/summary.json', lambda r: r['realized_joint_limits_checked']), True)
    for name in ['clip_tracked', 'diffraction_tracked']:
        same(both(f'local/{name}.json', lambda r: r['tracking_rad']), TAU)
    m = geometry().robot.model
    slack = np.inf
    for joint, lo, hi in zip(config()['joints'], config()['Q_lower'], config()['Q_upper']):
        low, high = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        slack = min(slack, lo-TAU-low, high-(hi+TAU))
    assert slack >= 0 and np.all(local_bounds().tau == TAU)
    return f'Q enlarged by 0.0081 rad stays {slack:.4f} rad inside the model joint limits'


def pair(name, f_plus, lb):
    file = f'local/{name}.json'
    same(both(file, lambda r: (r['epsilon_g'], r['reference']['q'], r['selected']['q'], r['config_file'])),
         (.02, [-.6, 1.], [-.3, 1.45], 'config_hard.json'))
    same(both(file, lambda r: r['improvement_threshold'] == r['reference']['f_plus']+.02), True)
    return (printed(both(file, lambda r: r['reference']['f_plus']), f_plus)+'; '
            + printed(both(file, lambda r: r['LB']), lb))


@claim(P_LOCAL2, '(f+(q0),LB) = (0.2262, 0.2028), clip envelopes with tracking errors: inconclusive after 32 subdivisions')
def local_clip():
    text = pair('clip_tracked', '0.2262', '0.2028')
    same(both('local/clip_tracked.json', lambda r: (r['theorem_outcome'], r['split_count'], r['stop_reason'])),
         ('inconclusive', 32, 'split limit'))
    return text


def improvement(name):
    same(both(f'local/{name}.json', lambda r: (r['theorem_outcome'], r['split_count'], r['stop_reason'],
                                               r['LB'] > r['reference']['f_plus']+.02)),
         ('guaranteed improvement', 0, 'decision inequality', True))


@claim(P_LOCAL2, '(0.2347, 0.3221), diffraction model without tracking errors: improvement inequality without subdivision')
def local_diffraction():
    improvement('diffraction_local')
    same(both('local/diffraction_local.json', lambda r: r['tracking_rad']), 0.)
    return pair('diffraction_local', '0.2347', '0.3221')


@claim(P_LOCAL2, '(0.2464, 0.3192), diffraction model with tracking errors: improvement inequality without subdivision')
def local_diffraction_tracked():
    improvement('diffraction_tracked')
    b = local_bounds()
    stored = load(STORED, 'local/diffraction_tracked.json')
    live = b.at(config()['q0'])['f_plus'], b.at(config()['Q_upper'])['f_minus']
    assert abs(live[0]-stored['reference']['f_plus']) <= REPLAY_TOL and abs(live[1]-stored['LB']) <= REPLAY_TOL
    return pair('diffraction_tracked', '0.2464', '0.3192')+'; live recomputation agrees'


@claim(P_LOCAL2, 'sharper enclosure restores the decision with the same nominal torso, tag and anchors')
def same_nominal():
    uniform, local, nominal = (load(STORED, 'static_sensitivity/diffraction_gain.json'),
                               load(STORED, 'local/diffraction_tracked.json'), geometry())
    assert uniform['theorem_outcome'] == 'inconclusive' and local['theorem_outcome'] == 'guaranteed improvement'
    assert uniform['budget'] == local['budget']
    g = local_bounds().geometry
    torso = g.robot._bid['torso_link']
    g.pose(config()['q0']); nominal.pose(config()['q0'])
    assert np.array_equal(g.p, nominal.p) and np.array_equal(g.anchors, nominal.anchors)
    assert np.array_equal(g.robot.data.xpos[torso], nominal.robot.data.xpos[torso])
    assert np.array_equal(g.robot.data.xmat[torso], nominal.robot.data.xmat[torso])
    assert np.allclose(g.p, [.1460, 0, .8870], atol=5e-5)
    return 'same budgets as the inconclusive uniform run; torso pose, tag and anchors identical to config_hard.json'


@claim(P_LOCAL2, 'at qb the omission rule of Lemma 2 retains only body-cell pairs with s_k <= 0.1602 on each link (tracked case)')
def omission_nodes():
    get = lambda r: max(r['selected']['retained_parameter_max'])
    b = local_bounds()
    a = b.at(config()['Q_upper'])
    stored = load(STORED, 'local/diffraction_tracked.json')['selected']
    assert a['retained_parameter_max'] == stored['retained_parameter_max']
    assert a['pruned_cells'] == stored['pruned_cells'] and a['total_cells_per_link'] == 7*257
    assert max(a['retained_parameter_max'])*256 == 41
    same(both('local/clip_tracked.json', get), 41/256)
    return (printed(both('local/diffraction_tracked.json', get), '0.1602')
            + f' (= 41/256); omitted pairs per link {a["pruned_cells"]} of {a["total_cells_per_link"]}')


@claim(P_LOCAL2, 'omitted pairs satisfy the condition of Lemma 2 (their term in the lower bound exceeds the upper bound)')
def omission_rule():
    from g1_geometry import cylinder_sdf, hull_sdf
    b, full = local_bounds(), local_bounds(False)
    g = b.geometry
    m, d = g.robot.model, g.robot.data
    N = int(b.b['N']); s = np.linspace(0, 1, N+1)
    sl, sr = np.maximum(0, s-.5/N), np.minimum(1, s+.5/N)
    count = 0
    for q in [config()['q0'], config()['Q_upper']]:
        a, z = b.at(q), full.at(q)
        for field in ['c_minus', 'c_plus', 'f_minus', 'f_plus']:
            assert np.array_equal(a[field], z[field]), field
        ray = g.p[None, None, :]+s[None, :, None]*(g.anchors-g.p)[:, None, :]
        for gid, hull, active in g.geoms:
            cell = np.maximum(b.profile(sl, active), b.profile(sr, active))
            for i in range(4):
                local = (ray[i]-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3, 3)
                phi = hull_sdf(local, hull) if hull is not None else cylinder_sdf(local, *m.geom_size[gid, :2])
                term = phi-b.b['e_phi']-b.d[i]/(2*N)-cell[i]
                omitted = s > a['retained_parameter_max'][i]
                count += int(omitted.sum())
                assert np.all(term[omitted] > a['c_plus'][i])
    return f'q0 and qb: all {count} pairs beyond the last retained node have terms above the upper bound; bounds identical to full evaluation'


STRESS = 'local/stress.json'


@claim(P_LOCAL2, '32 further stress cases with rigid motions and joint errors on the boundary of the uncertainty set')
def stress_boundary():
    budget = load(STORED, 'local/summary.json')['budget']
    g = local_bounds().geometry
    g.pose(config()['q0'])
    t0 = g.robot.data.xpos[g.robot._bid['torso_link']].copy()
    worst_tag = 0.
    frames = []
    for root in (STORED, REPLAY):
        cases = load(root, STRESS)
        assert len(cases) == 32
        e_i, e_fr = (load(root, 'local/summary.json')['budget'][k] for k in ['range_vector_m', 'relative_geometry_m'])
        for c in cases:
            assert np.all(np.abs(np.abs(c['joint_error_rad'])-TAU) == 0)       # vertices of the joint-error box
            R, shift = np.array(c['torso_rotation']), np.array(c['torso_translation_m'])
            assert np.allclose(R@R.T, np.eye(3), atol=1e-14) and not np.allclose(R, np.eye(3))
            tag = t0+R@(g.p-t0)+shift
            worst_tag = max(worst_tag, abs(np.linalg.norm(tag-g.p)-e_i))        # tag displacement equals e_i
            assert c['rigid_frame_m'] <= e_fr+1e-12
            frames.append(c['rigid_frame_m']/e_fr)
    assert worst_tag < 1e-12 and abs(budget['range_vector_m']-load(REPLAY, 'local/summary.json')['budget']['range_vector_m']) <= REPLAY_TOL
    same(both('local/summary.json', lambda r: r['stress_check']['cases']), 32)
    return (f'|dq_iota| = tau in all cases; tag displacement = e_i (max deviation {worst_tag:.1e} m); '
            f'rotation plus translation use {100*min(frames):.1f}-{100*max(frames):.1f}% of e_fr')


@claim(P_LOCAL2, 'no clearance bound was violated')
def stress_no_violation():
    g = local_bounds().geometry
    g.pose(config()['q0'])
    t0 = g.robot.data.xpos[g.robot._bid['torso_link']].copy()
    margin = np.inf
    for root in (STORED, REPLAY):
        for c in load(root, STRESS):
            tag = t0+np.array(c['torso_rotation'])@(g.p-t0)+np.array(c['torso_translation_m'])
            lengths = np.linalg.norm(g.anchors-tag, axis=1)
            grid, lower, upper = (np.array(c[k]) for k in ['clearance_grid', 'lower', 'upper'])
            # The 512-interval grid minimum brackets the continuous clearance from both sides.
            margin = min(margin, np.min(grid-lengths/(2*512)-lower), np.min(upper-grid))
    assert margin >= 0
    same(both('local/summary.json', lambda r: r['stress_check']['violations']), 0)
    # One case is recomputed here, independently of the stored numbers.
    from local_experiments import actual_clearance
    c = load(STORED, STRESS)[9]
    live, _, _, _ = actual_clearance(g, c['q'], np.array(c['joint_error_rad']), np.array(c['torso_rotation']),
                                     np.array(c['torso_translation_m']))
    assert np.max(np.abs(live-c['clearance_grid'])) <= REPLAY_TOL
    difference = max(abs(x-y) for a, b in zip(load(STORED, STRESS), load(REPLAY, STRESS))
                     for k in ['clearance_grid', 'lower', 'upper'] for x, y in zip(a[k], b[k]))
    return f'smallest slack of the 2x32x4 two-sided checks {margin:.3e} m; replay diff {difference:.1e}'


@claim(P_LOCAL2, 'cell omission gave the same bounds as full evaluation')
def stress_omission():
    same(both('local/summary.json', lambda r: r['stress_check']['pruned_full_agreement']), True)
    replay = load(REPLAY, 'local/summary.json')['stress_check']
    assert replay['pruned_full_agreement_cases'] == 32
    cases = load(REPLAY, STRESS)
    worst = max(c['omission_max_abs_difference'] for c in cases)
    assert worst == 0 and all(min(c['omitted_cells']) > 0 for c in cases)
    return 'replay: identical clearance bounds with and without omission at all 32 stress postures (max diff 0)'


# ------------------------------------------------------------ reproducibility
@claim(P_REPRO, 'source code, configuration files, stored results, exact-arithmetic audit and reproduction instructions')
def repository_contents():
    needed = ['poc/bounds.py', 'poc/g1_geometry.py', 'poc/run.py', 'poc/diffraction_model.py', 'poc/local_bounds.py',
              'poc/local_experiments.py', 'poc/static_sensitivity.py', 'poc/dynamic_standing.py',
              'poc/config.json', 'poc/config_hard.json', 'poc/config_bnb.json',
              'poc/results/result.json', 'poc/results/hard/result.json', 'poc/results/bnb/result.json',
              'poc/results/reviewer/dynamics_static/result.json', 'poc/results/reviewer/static_sensitivity/summary.json',
              'poc/results/reviewer/local/summary.json', 'poc/results/reviewer/diffraction_assumed_2mm.json',
              'poc/verified_sphere.py', 'poc/results/reviewer/verified_sphere.json',
              'README.md', 'REPRODUCTION.md', 'LICENSE', 'reproduce.py', 'requirements.txt', 'g1_model/g1.xml']
    missing = [f for f in needed if not (ROOT/f).is_file()]
    assert not missing, missing
    return f'{len(needed)} required files present'


@claim(P_REPRO, 'the package provides exact sphere replay, a G1 geometry adapter, static searches and finite standing-sensitivity simulations')
def package_parts():
    import importlib
    import humanoid_uwb_sim
    from humanoid_uwb_sim import standing
    for name in ['verified_sphere', 'bounds', 'static_sensitivity', 'dynamic_standing']:
        importlib.import_module(name)
    shim = ROOT/'poc/menagerie_shim'
    assert Path(humanoid_uwb_sim.__file__).resolve().is_relative_to(shim)
    assert Path(standing.__file__).resolve().is_relative_to(shim)
    assert all(hasattr(humanoid_uwb_sim, n) for n in ['G1Model', 'TagMount']) and hasattr(standing, 'standing_model')
    assert not (ROOT/'humanoid-uwb-sim').exists()
    source = (ROOT/'poc/dynamic_standing.py').read_text()
    assert 'humanoid-uwb-sim/' not in source and 'assets_cache' not in source
    assert (shim/'rerun.py').is_file() and (ROOT/'poc/replay_sensitivity.py').is_file()
    assert load(STORED, 'verified_sphere.json')
    return 'verified_sphere.py, menagerie_shim adapter (kinematics + standing model), searches, dynamic_standing.py; no humanoid-uwb-sim'


@claim(P_REPRO, 'stored G1 search results record Python 3.12.10, MuJoCo 3.2.6, NumPy 1.26.4 and SciPy 1.14.1')
def recorded_versions():
    expected = dict(python='3.12.10', mujoco='3.2.6', numpy='1.26.4', scipy='1.14.1')
    files = ['result.json', 'inconclusive/result.json', 'excluded/result.json', 'hard/result.json', 'bnb/result.json']
    for f in files:
        provenance = json.loads((ROOT/'poc/results'/f).read_text())['provenance']
        assert {k: provenance[k] for k in expected} == expected, f
    return f'provenance of the {len(files)} stored G1 searches'


@claim(P_REPRO, 'all G1 computations use ordinary double-precision floating-point arithmetic')
def double_precision():
    b = local_bounds()
    a = b.at(config()['q0'])
    g = geometry()
    arrays = [g.p, g.anchors, g.Lq, g.clearance(config()['q0'], 16), b.d, b.sigma, diffraction()[0].scales,
              b.profile(np.linspace(0, 1, 3), [])]
    assert all(x.dtype == np.float64 for x in arrays) and isinstance(a['f_minus'], float)
    for name in ['diffraction_assumed_2mm.json', 'local/diffraction_tracked.json', 'static_sensitivity/restricted_gain.json']:
        assert 'float64' in load(STORED, name)['certification_status']
    return 'geometry, clearances, profiles and bounds are numpy float64 / Python float'


@claim(P_REPRO, 'replay here against the stored originals (replay_sensitivity/report.json)')
def replay_report():
    import mujoco
    report = load(REPLAY, 'report.json')
    summary = report['summary']
    assert summary['non_float_mismatches'] == 0, 'outcomes, counts or selected postures differ'
    assert summary['max_abs_difference'] <= REPLAY_TOL
    assert report['replay_versions']['mujoco'] == mujoco.__version__ == '3.2.6'
    # Every stored value is replayed (the two partial comparisons are labelled with parentheses).
    assert all(c.get('only_in_stored', 0) == 0 for k, c in report['comparisons'].items() if '(' not in k)
    return (f'{summary["files_compared"]} files, {summary["float_values"]} floats '
            f'({summary["bitwise_identical_floats"]} bit-identical), max abs difference '
            f'{summary["max_abs_difference"]:.2e} <= {REPLAY_TOL:g}, 0 non-float mismatches')


# ------------------------------------------------------------------- drivers
def evaluate(function):
    try:
        return True, function()
    except Exception as error:               # report every failure, including missing files
        return False, f'{type(error).__name__}: {error}'


def main():
    failures = 0
    for location, text, function in CLAIMS:
        ok, detail = evaluate(function)
        failures += not ok
        print(f'{"PASS" if ok else "FAIL"} [{location}] {text} -- {detail}')
    print(f'{len(CLAIMS)-failures}/{len(CLAIMS)} claims passed')
    return 1 if failures else 0


def pytest_generate_tests(metafunc):
    if 'article_claim' in metafunc.fixturenames:
        metafunc.parametrize('article_claim', CLAIMS, ids=[c[2].__name__ for c in CLAIMS])


def test_article_claim(article_claim):
    location, text, function = article_claim
    function()


if __name__ == '__main__':
    sys.exit(main())
