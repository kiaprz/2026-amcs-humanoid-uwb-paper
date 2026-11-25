"""Checker: every number of the article's static G1 study against the software.

Scope: Sections 6.1 "Immobilized Unitree G1 baseline" and 6.2 "Full-rank
reference and competing link changes" (without the "Estimator mismatch"
paragraph), Figs. 3 and 4 and the reproducibility statement on the G1 search
results.  Each claim is tested twice where it is a computed value:

  * on the stored result files of the authors' original runs (poc/results/**),
  * on the replay with the geometry adapter, as recorded by
    poc/replay_g1_static.py in poc/results/replay_report.json.

Printed values are compared by rounding the full float64 value to the nearest
last printed digit (the article's rule for G1 values marked by "approximately").
The checker needs only NumPy; it does not run MuJoCo.

Usage (from the repository root):
    python poc/replay_g1_static.py            # once; writes the replay report (about 8 min)
    python poc/check_article_g1_static.py     # one PASS/FAIL line per claim; exit code 1 on failure
    python -m pytest poc/check_article_g1_static.py -q

If the environment variable AMCS_ARTICLE_TEX names the article's LaTeX source,
the quoted fragments are additionally required to occur in it verbatim.
"""
import csv
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parent
sys.path.insert(0, str(POC))
from replay_g1_static import RUNS, REPORT, RESULTS, TOLERANCE, TRACE_TOLERANCE, headline, run_config, stored_path

S61 = 'Sec. 6.1'
S62 = 'Sec. 6.2'
IMPROVEMENT, INCONCLUSIVE, EXCLUDED = 'guaranteed improvement', 'inconclusive', 'excluded improvement'


def rounds_to(value, printed):
    """True if value, rounded to the nearest last digit of the printed string, equals it."""
    printed = printed.replace('−', '-')
    target = Decimal(printed)
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(target.as_tuple().exponent),
                                                 rounding=ROUND_HALF_EVEN) == target


def all_round(values, printed):
    return len(values) == len(printed) and all(rounds_to(v, p) for v, p in zip(values, printed))


def close(a, b, tol=1e-12):
    return bool(np.allclose(np.array(a, float), np.array(b, float), rtol=0, atol=tol))


def load():
    stored = {name: json.loads(stored_path(name).read_text()) for name in RUNS}
    if not REPORT.exists():
        raise SystemExit('Missing %s: run poc/replay_g1_static.py first.' % REPORT)
    report = json.loads(REPORT.read_text())
    return stored, report


def build_claims():
    """Return the checklist as tuples (location, article fragment, passed, detail)."""
    stored, report = load()
    S = {name: headline(r) for name, r in stored.items()}           # stored results
    R = {name: report['runs'][name]['replayed'] for name in RUNS if name in report['runs']}  # replay
    model = report.get('model', {})
    claims = []

    def claim(loc, quote, ok, detail=''):
        claims.append((loc, quote, bool(ok), str(detail)))

    def value(loc, quote, run, test, show):
        """A computed value: must hold for the stored result and for the replay."""
        missing = run not in R
        ok = test(S[run]) and not missing and test(R[run])
        claim(loc, quote, ok, 'stored %s | replay %s' % (show(S[run]), 'missing' if missing else show(R[run])))

    cfg = {name: run_config(name) for name in RUNS}
    base, hard = cfg['base'], cfg['hard']
    mb, mh = model.get('baseline', {}), model.get('full_rank', {})

    # ---------------------------------------------------------------- Section 6.1: model
    claim(S61, r'physics engine, version 3.2.6',
          all(r['provenance']['mujoco'] == '3.2.6' for r in stored.values())
          and report['environment']['mujoco'] == '3.2.6',
          'stored %s | replay %s' % (sorted({r['provenance']['mujoco'] for r in stored.values()}),
                                     report['environment']['mujoco']))
    claim(S61, r'29 position-actuated joints',
          mb.get('actuator_count') == 29 and mb.get('position_actuator_count') == 29
          and mb.get('actuated_joint_count') == 29 and mb.get('hinge_joint_count') == 29,
          'actuators %s, position type %s, distinct joints %s' % (
              mb.get('actuator_count'), mb.get('position_actuator_count'), mb.get('actuated_joint_count')))
    claim(S61, r'MuJoCo Menagerie, commit \texttt{71f066a}',
          all(r['provenance']['model_manifest']['requested_revision'].startswith('71f066a')
              and r['provenance']['model_verified']['verified'] for r in stored.values())
          and mb.get('g1_xml_sha256_matches_stored') is True and mb.get('manifest_files_mismatched') == []
          and mb.get('manifest_file_count') == 39,
          'revision %s; %s manifest files, mismatched in g1_model: %s' % (
              mb.get('manifest_revision'), mb.get('manifest_file_count'), mb.get('manifest_files_mismatched')))
    claim(S61, r'The floating base and the torso are fixed',
          mb.get('free_joint_count') == 1 and mb.get('tag_displacement_over_Q_m') == 0.0
          and mh.get('tag_displacement_over_Q_m') == 0.0 and mb.get('torso_rotation_is_identity') is True
          and base['base_position_m'] == [0.0, 0.0, 0.793],
          'tag displacement over the corners of Q: %s m (baseline), %s m (full rank)' % (
              mb.get('tag_displacement_over_Q_m'), mh.get('tag_displacement_over_Q_m')))
    claim(S61, r'at the local offset $(0.15,0,0.05)$~m',
          base['tag_offset_m'] == [0.15, 0.0, 0.05] and hard['tag_offset_m'] == [0.15, 0.0, 0.05]
          and mb.get('tag_body') == 'torso_link'
          and close(np.array(S['base']['p'])-mb.get('torso_origin_m', [0, 0, 0]), [0.15, 0, 0.05], 1e-15),
          'p - torso origin = %s' % (np.array(S['base']['p'])-mb.get('torso_origin_m', [0, 0, 0])).tolist())
    value(S61, r'p\approx(0.1460,0,0.8870)', 'base',
          lambda h: all_round(h['p'], ['0.1460', '0', '0.8870']) and h['p'][1] == 0.0, lambda h: h['p'])
    value(S61, r'a_1-p&=(1.8,1.2,-0.4),\ \ a_2-p=(1.8,-1.2,-0.4)', 'base',
          lambda h: close(h['anchor_offsets'], [[1.8, 1.2, -.4], [1.8, -1.2, -.4], [-1.8, 1.2, -.4], [0, 0, 1.8]]),
          lambda h: 'max deviation %.1e m' % np.max(np.abs(np.array(h['anchor_offsets'])-base['anchor_offsets_m'])))
    value(S61, r'd_1=d_2=d_3=2.2$~m and $d_4=1.8$~m', 'base',
          lambda h: close(h['d'], [2.2, 2.2, 2.2, 1.8]), lambda h: h['d'])
    claim(S61, r'\Q=\Cbox=[-1.2,0.8]\times[-0.3,1.6]',
          base['Q_lower'] == [-1.2, -0.3] and base['Q_upper'] == [0.8, 1.6] and base['q0'] == [-0.7, 1.2]
          and base['joints'] == ['left_shoulder_pitch_joint', 'left_elbow_joint']
          and all(stored[n]['config'] == cfg[n] for n in RUNS),
          'Q=[%s, %s], q0=%s, joints %s' % (base['Q_lower'], base['Q_upper'], base['q0'], base['joints']))
    ranges = mb.get('controlled_joint_ranges', {})
    claim(S61, r'candidates are checked against this box and the joint limits of the model',
          len(ranges) == 2 and all(ranges[j][0] <= lo and hi <= ranges[j][1]
                                   for j, lo, hi in zip(base['joints'], base['Q_lower'], base['Q_upper']))
          and all(np.all((np.array(a['q']) >= c['Q_lower']) & (np.array(a['q']) <= c['Q_upper']))
                  for n, c in cfg.items() for a in stored[n]['evaluations']),
          'joint ranges %s; all stored evaluated postures lie in Q' % ranges)
    solids = mb.get('moving_solids', [])
    hand = mb.get('left_rubber_hand_geoms', [])
    claim(S61, r'seven collision geometries of the left arm',
          len(solids) == 7 and all(s['body'].startswith('left_') and s['collision'] for s in solids)
          and all(len(h['moving_bodies']) == 7 for h in S.values())
          and [s['body'] for s in solids] == S['base']['moving_bodies'],
          ', '.join(s['body'] for s in solids))
    claim(S61, r'the rubber hand of the model has no collision geometry and is not included',
          len(hand) == 1 and not hand[0]['collision'] and not hand[0]['in_L'] and hand[0]['group'] != 3,
          hand)
    claim(S61, r'Cylinders are represented by their analytic signed-distance functions, and mesh geometries by the convex hulls',
          {s['type'] for s in solids} == {'cylinder', 'mesh'}
          and all(s['convex_hull'] == (s['type'] == 'mesh') for s in solids),
          '%d cylinders, %d meshes' % (sum(s['type'] == 'cylinder' for s in solids),
                                       sum(s['type'] == 'mesh' for s in solids)))
    budget = dict(N=256, e_phi=1e-5, e_lambda=1e-6, relative_geometry_m=0.002, range_vector_m=0.004,
                  direction_offset=0.001)
    claim(S61, r'$\beta=0$, $e_i=0.004$~m, $\varepsilon_i=0.001$, $L_i^\eta\etab=0.002$~m, $e_\phi=10^{-5}$~m and $e_\lambda=10^{-6}$',
          all(c['budget'] == budget for c in cfg.values()) and all(r['beta'] == 0 for r in stored.values()),
          base['budget'])

    def cpm_ok(r):
        b, d = r['config']['budget'], np.array(r['geometry']['d_m'])
        return all(close(a['c_minus'], np.array(a['c_hat'])-b['e_phi']-d/(2*b['N'])-b['relative_geometry_m'], 1e-15)
                   and close(a['c_plus'], np.array(a['c_hat'])+b['e_phi']+b['relative_geometry_m'], 1e-15)
                   for a in r['evaluations'])
    claim(S61, r'with $N_i=256$ intervals on each segment and the correction $d_i/(2N_i)$',
          all(cpm_ok(r) for r in stored.values()),
          'c^-, c^+ of all %d stored evaluations follow eq. (5)' % sum(len(r['evaluations']) for r in stored.values()))
    claim(S61, r'we use the envelopes of Section~\ref{sec:interval}',
          base['envelope'] == dict(lower_floor=0.0, lower_cap=0.9, lower_slope=18.0, lower_offset=0.01,
                                   upper_floor=0.2, upper_cap=1.0, upper_slope=40.0, upper_offset=0.0),
          base['envelope'])

    def sigma_ok(h):
        return close(h['sigma'], 0.004/np.array(h['d'])+0.001*2.001, 1e-15)
    value(S61, r'range vector by $4$~mm', 'base', sigma_ok, lambda h: 'sigma %s' % h['sigma'])

    # ---------------------------------------------------------------- eq. (15)
    def lq_formula(h, origin):
        """Eq. (15) recomputed from the stored chain radii; all seven solids descend from the first
        controlled joint, and the maximum is attained by those that also descend from the elbow."""
        p, o = np.array(h['p']), np.array(origin)
        reach = list(h['joint_reach'].values())
        out = []
        for off in h['anchor_offsets']:
            dbar = max(np.linalg.norm(p-o), np.linalg.norm(p+off-o))+0.002
            out.append(max(np.linalg.norm([dbar+reach[0]]), np.linalg.norm([dbar+r for r in reach])))
        return out
    for run, key in [('base', 'baseline'), ('hard', 'full_rank')]:
        origin = model.get(key, {}).get('torso_origin_m', [np.nan]*3)
        value(S61 if run == 'base' else S62, r'L_i^q=\max_{\ell\in\Lset}' if run == 'base' else r'L_2^q\approx3.378',
              run, lambda h: close(lq_formula(h, origin), h['Lq'], 1e-12),
              lambda h: 'eq. (15) from chain radii %s' % np.round(lq_formula(h, origin), 6).tolist())
    value(S61, r'L_1^q=L_2^q\approx3.820$, $L_3^q\approx3.472$ and $L_4^q\approx3.169$~m/rad', 'base',
          lambda h: all_round(h['Lq'], ['3.820', '3.820', '3.472', '3.169']) and h['Lq'][0] == h['Lq'][1],
          lambda h: h['Lq'])

    # ---------------------------------------------------------------- Section 6.1: search
    ev = stored['base']['evaluations']
    lo, hi = np.array(base['Q_lower']), np.array(base['Q_upper'])
    value(S61, r'Candidate generation uses $q_0$, two opposite box corners, the centre and 24 postures drawn uniformly at random',
          'base', lambda h: h['phase_counts'] == {'reference': 1, 'candidate': 27, 'box center': 0}
          and h['evaluation_count'] == 28 and h['cache_hits'] == 1 and base['candidate_count'] == 24
          and ev[0]['q'] == base['q0'] and ev[1]['q'] == lo.tolist() and ev[2]['q'] == hi.tolist()
          and ev[3]['q'] == ((lo+hi)/2).tolist(),
          lambda h: '%s, cache hits %d' % (h['phase_counts'], h['cache_hits']))
    value(S61, r'The selected posture is $q_b=(0.8,1.6)$~rad', 'base',
          lambda h: h['qb'] == [0.8, 1.6] and h['q0'] == [-0.7, 1.2], lambda h: h['qb'])
    value(S61, r'where all computed lower weights equal $0.9$', 'base',
          lambda h: h['w_minus_qb'] == [0.9]*4, lambda h: h['w_minus_qb'])
    value(S61, r'the grid clearance of link~1 is $-0.01899$~m', 'base',
          lambda h: rounds_to(h['c_hat_q0'][0], '-0.01899'), lambda h: h['c_hat_q0'][0])
    value(S61, r'and its lower weight is zero', 'base',
          lambda h: h['w_minus_q0'][0] == 0.0, lambda h: h['w_minus_q0'][0])
    value(S61, r'at $q_b$ it is $0.2515$~m', 'base',
          lambda h: rounds_to(h['c_hat_qb'][0], '0.2515'), lambda h: h['c_hat_qb'][0])
    value(S61, r'[f^-(q_0),f^+(q_0)]&\approx[-0.01068,0.2672]', 'base',
          lambda h: rounds_to(h['f0_minus'], '-0.01068') and rounds_to(h['f0_plus'], '0.2672'),
          lambda h: [h['f0_minus'], h['f0_plus']])
    value(S61, r'[f^-(q_b),f^+(q_b)]&\approx[0.6030,0.7014]', 'base',
          lambda h: rounds_to(h['fb_minus'], '0.6030') and rounds_to(h['fb_plus'], '0.7014')
          and h['LB'] == h['fb_minus'], lambda h: [h['fb_minus'], h['fb_plus']])
    value(S61, r'$h_2$, $h_3$, $h_4$ span only a plane', 'base',
          lambda h: h['rank_h234'] == 2 and h['w_minus_q0'][0] == 0.0, lambda h: 'rank %d' % h['rank_h234'])
    value(S61, r'at margin $0.02$ it gives a conditional gain lower bound of approximately $0.3358$ without subdivision',
          'base', lambda h: h['epsilon_g'] == 0.02 and rounds_to(h['gain_lower'], '0.3358')
          and h['split_count'] == 0 and h['outcome'] == IMPROVEMENT and h['LB'] > h['improvement_threshold'],
          lambda h: 'gain %s, %d subdivisions, %s' % (h['gain_lower'], h['split_count'], h['outcome']))
    value(S61, r'\Gamma_\infty=0.1\lambda_{\min}(\sum_i h_i h_i^T)\approx0.06857', 'base',
          lambda h: rounds_to(h['calibration_width'], '0.06857')
          and abs(h['calibration_width']-h['calibration_width_recorded']) < 1e-15
          and close(base['envelope']['upper_cap']-base['envelope']['lower_cap'], 0.1, 1e-15),
          lambda h: h['calibration_width'])
    value(S61, r'increase $\UB-\LB$ to approximately $0.09837$', 'base',
          lambda h: rounds_to(h['width'], '0.09837') and h['UB'] == h['f_cap'], lambda h: h['width'])
    value(S61, r'Refinement therefore cannot reach tolerance $0.01$ with these inputs', 'base',
          lambda h: h['calibration_width'] > 0.01 and base['epsilon_o'] == 0.01,
          lambda h: 'Gamma_inf %.5f > 0.01' % h['calibration_width'])
    readme = (POC.parent/'REPRODUCTION.md').read_text(encoding='utf-8')
    summary = {row['scenario']: row for row in csv.DictReader((RESULTS/'summary.csv').open())}
    outcomes = [('base', 0.02, IMPROVEMENT), ('inconclusive', 0.4, INCONCLUSIVE), ('excluded', 0.8, EXCLUDED)]
    claim(S61, r'lists the improvement, inconclusive and exclusion outcomes of this configuration for the margins $0.02$, $0.4$ and $0.8$',
          all(S[n]['epsilon_g'] == m and S[n]['outcome'] == o and n in R and R[n]['outcome'] == o
              and R[n]['epsilon_g'] == m for n, m, o in outcomes)
          and all(t in readme for t in ['epsilon_g=0.02', 'epsilon_g=0.4', 'epsilon_g=0.8', 'inconclusive',
                                        'excluded improvement'])
          and [summary[k]['outcome'] for k in ['easy', 'inconclusive', 'excluded']] == [o for _, _, o in outcomes],
          '; '.join('margin %s: %s (%s, %d subdivisions)' % (m, S[n]['outcome'], S[n]['stop_reason'], S[n]['split_count'])
                    for n, m, _ in outcomes)+'; listed in REPRODUCTION.md and poc/results/summary.csv')

    # ---------------------------------------------------------------- Section 6.2
    value(S62, r'a_2-p=(1.8,-0.3,-0.4)', 'hard',
          lambda h: close(h['anchor_offsets'], [[1.8, 1.2, -.4], [1.8, -.3, -.4], [-1.8, 1.2, -.4], [0, 0, 1.8]])
          and cfg['bnb']['anchor_offsets_m'] == hard['anchor_offsets_m'], lambda h: h['anchor_offsets'][1])
    value(S62, r'the smallest eigenvalue of $0.9\sum_{i=2}^4h_i h_i^T$ is approximately $0.07017$', 'hard',
          lambda h: h['rank_h234'] == 3 and rounds_to(h['lambda_min_234_lower_cap'], '0.07017')
          and hard['envelope']['lower_cap'] == 0.9,
          lambda h: 'rank %d, eigenvalue %s' % (h['rank_h234'], h['lambda_min_234_lower_cap']))
    same = ['joints', 'base_position_m', 'tag_offset_m', 'budget']
    value(S62, r'The fixed torso, tag, collision solids, discretization and error budgets are unchanged', 'hard',
          lambda h: all(hard[k] == base[k] and cfg['bnb'][k] == base[k] for k in same)
          and h['p'] == S['base']['p'] and h['moving_bodies'] == S['base']['moving_bodies']
          and h['joint_reach'] == S['base']['joint_reach'],
          lambda h: 'identical: %s, p, solids, chain radii' % ', '.join(same))
    value(S62, r'only $d_2\approx1.868$~m, $\sigma_2\approx0.004142$ and $L_2^q\approx3.378$~m/rad change with anchor~2',
          'hard', lambda h: rounds_to(h['d'][1], '1.868') and rounds_to(h['sigma'][1], '0.004142')
          and rounds_to(h['Lq'][1], '3.378') and sigma_ok(h)
          and all(h[k][i] == S['base'][k][i] for k in ['d', 'sigma', 'Lq'] for i in [0, 2, 3])
          and all(h[k][1] != S['base'][k][1] for k in ['d', 'sigma', 'Lq']),
          lambda h: [h['d'][1], h['sigma'][1], h['Lq'][1]])
    claim(S62, r'\Q=[-0.6,-0.3]\times[1.0,1.45],\quad q_0=(-0.6,1.0)',
          all(c['Q_lower'] == [-0.6, 1.0] and c['Q_upper'] == [-0.3, 1.45] and c['q0'] == [-0.6, 1.0]
              for c in [hard, cfg['bnb']]) and {k: v for k, v in hard.items() if k != 'epsilon_g'}
          == {k: v for k, v in cfg['bnb'].items() if k != 'epsilon_g'},
          'Q=[%s, %s], q0=%s; the two full-rank configurations differ only in the margin' % (
              hard['Q_lower'], hard['Q_upper'], hard['q0']))
    e = hard['envelope']
    c = np.linspace(-0.2, 0.4, 6001)
    minus = np.clip(e['lower_floor']+e['lower_slope']*(c-e['lower_offset']), e['lower_floor'], e['lower_cap'])
    plus = np.clip(e['upper_cap']+e['upper_slope']*(c-e['upper_offset']), e['upper_floor'], e['upper_cap'])
    claim(S62, r'\psi^-(c)&=\clip(0.1+4c;\,0.1,\,0.9)',
          close(minus, np.clip(0.1+4*c, 0.1, 0.9), 1e-15),
          'max deviation %.1e on [-0.2, 0.4] m' % np.max(np.abs(minus-np.clip(0.1+4*c, 0.1, 0.9))))
    claim(S62, r'\psi^+(c)&=\clip(0.25+5c;\,0.2,\,1)',
          close(plus, np.clip(0.25+5*c, 0.2, 1.0), 1e-15),
          'max deviation %.1e on [-0.2, 0.4] m' % np.max(np.abs(plus-np.clip(0.25+5*c, 0.2, 1.0))))
    evh = stored['hard']['evaluations']
    value(S62, r'Candidate generation evaluates two opposite corners and the centre', 'hard',
          lambda h: h['evaluation_count'] == 3 and h['cache_hits'] == 2 and hard['candidate_count'] == 0
          and [a['q'] for a in evh] == [hard['Q_lower'], hard['Q_upper'],
                                        ((np.array(hard['Q_lower'])+hard['Q_upper'])/2).tolist()],
          lambda h: '%d evaluations (q0 is the lower corner), cache hits %d' % (h['evaluation_count'], h['cache_hits']))
    for run in ['hard', 'bnb']:
        value(S62, r'The upper corner $q_b=(-0.3,1.45)$ is selected', run,
              lambda h: h['qb'] == [-0.3, 1.45] == hard['Q_upper'] and h['q0'] == [-0.6, 1.0],
              lambda h: '%s (margin %s)' % (h['qb'], h['epsilon_g']))
    value(S62, r'[f^-(q_0),f^+(q_0)]&\approx[0.08369,0.1764]', 'hard',
          lambda h: rounds_to(h['f0_minus'], '0.08369') and rounds_to(h['f0_plus'], '0.1764'),
          lambda h: [h['f0_minus'], h['f0_plus']])
    value(S62, r'[f^-(q_b),f^+(q_b)]&\approx[0.2315,0.4075]', 'hard',
          lambda h: rounds_to(h['fb_minus'], '0.2315') and rounds_to(h['fb_plus'], '0.4075'),
          lambda h: [h['fb_minus'], h['fb_plus']])
    value(S62, r'change from $0.1000$, $0.4823$, $0.3609$ and $0.4532$', 'hard',
          lambda h: all_round(h['w_minus_q0'], ['0.1000', '0.4823', '0.3609', '0.4532']), lambda h: h['w_minus_q0'])
    value(S62, r'to $0.5417$, $0.5462$, $0.2502$ and $0.5399$', 'hard',
          lambda h: all_round(h['w_minus_qb'], ['0.5417', '0.5462', '0.2502', '0.5399']), lambda h: h['w_minus_qb'])
    value(S62, r'links~1, 2 and~4 gain, whereas link~3 loses', 'hard',
          lambda h: [b > a for a, b in zip(h['w_minus_q0'], h['w_minus_qb'])] == [True, True, False, True],
          lambda h: np.round(np.array(h['w_minus_qb'])-h['w_minus_q0'], 4).tolist())
    value(S62, r'For $\varepsilon_g=0.02$ the gain lower bound is $0.05506$, and the decision inequality holds without subdivision',
          'hard', lambda h: h['epsilon_g'] == 0.02 and rounds_to(h['gain_lower'], '0.05506')
          and h['outcome'] == IMPROVEMENT and h['split_count'] == 0 and h['LB'] > h['improvement_threshold'],
          lambda h: 'gain %s, %d subdivisions, %s' % (h['gain_lower'], h['split_count'], h['outcome']))
    value(S62, r'For the larger margin $\varepsilon_g=0.36$ the exclusion threshold is approximately $0.4437$', 'bnb',
          lambda h: h['epsilon_g'] == 0.36 and rounds_to(h['exclusion_threshold'], '0.4437')
          and h['exclusion_threshold'] == h['f0_minus']+0.36, lambda h: h['exclusion_threshold'])
    value(S62, r'The ceiling $\fcap\approx0.5615$ and the initial box bound exceed this threshold', 'bnb',
          lambda h: rounds_to(h['f_cap'], '0.5615') and h['f_cap'] > h['exclusion_threshold']
          and h['root_upper'] > h['exclusion_threshold'],
          lambda h: 'f_cap %s, root bound %s' % (h['f_cap'], h['root_upper']))
    value(S62, r'the candidate fails the improvement test', 'bnb',
          lambda h: not h['LB'] > h['improvement_threshold'],
          lambda h: 'LB %.7f <= %.7f' % (h['LB'], h['improvement_threshold']))
    value(S62, r'establishes the exclusion inequality after 1778 subdivisions and 3559 posture evaluations', 'bnb',
          lambda h: h['outcome'] == EXCLUDED and h['stop_reason'] == 'decision inequality'
          and h['split_count'] == 1778 and h['evaluation_count'] == 3559 and h['UB'] <= h['exclusion_threshold'],
          lambda h: '%s, %d subdivisions, %d evaluations' % (h['outcome'], h['split_count'], h['evaluation_count']))
    value(S62, r'with $\LB\approx0.2315$, $\UB\approx0.4437$', 'bnb',
          lambda h: rounds_to(h['LB'], '0.2315') and rounds_to(h['UB'], '0.4437'), lambda h: [h['LB'], h['UB']])
    value(S62, r'f^-(q_0)+0.36-\UB\approx1.94\times10^{-5}', 'bnb',
          lambda h: rounds_to(h['exclusion_slack'], '1.94e-5') and h['exclusion_slack'] > 0,
          lambda h: h['exclusion_slack'])
    value(S62, r'the upper bound of $w_1$ is at most $0.8986<1$', 'bnb',
          lambda h: h['cover_weight_upper_max'][0] <= 0.8986 and rounds_to(h['cover_weight_upper_max'][0], '0.8986')
          and h['cover_excludes_simultaneous_upper_saturation'] is True,
          lambda h: h['cover_weight_upper_max'][0])
    value(S62, r'unlike in Section~\ref{sec:g1poc}, the four upper weights cannot reach their cap simultaneously', 'base',
          lambda h: h['cover_weight_upper_max'] == [1.0]*4
          and h['cover_excludes_simultaneous_upper_saturation'] is False and h['w_plus_qb'] == [1.0]*4,
          lambda h: 'baseline cover: %s' % h['cover_weight_upper_max'])
    times = [S[n]['search_elapsed_s'] for n in ['base', 'inconclusive', 'excluded']]
    hw = {(r['hardware']['cpu'], r['hardware']['model']) for r in stored.values()}
    claim(S62, r'the searches took $0.06$--$5.93$~s for the baseline configuration',
          rounds_to(min(times), '0.06') and rounds_to(max(times), '5.93') and hw == {('Apple M1', 'MacBookPro17,1')},
          'recorded search_elapsed_s %s on %s (stored only; not reproduced)' % (np.round(times, 3).tolist(), sorted(hw)))
    claim(S62, r'$0.19$~s and $304.6$~s for the two full-rank runs',
          rounds_to(S['hard']['search_elapsed_s'], '0.19') and rounds_to(S['bnb']['search_elapsed_s'], '304.6'),
          'recorded search_elapsed_s %.4f and %.4f (stored only; not reproduced)' % (
              S['hard']['search_elapsed_s'], S['bnb']['search_elapsed_s']))

    # ---------------------------------------------------------------- Fig. 3
    F3 = 'Fig. 3'
    panels = [('a', 'base', 'q0', [-0.7, 1.2], '−0.01899'), ('b', 'base', 'qb', [0.8, 1.6], '0.2515'),
              ('c', 'hard', 'q0', [-0.6, 1.0], '−0.01774'), ('d', 'hard', 'qb', [-0.3, 1.45], '0.1167')]
    quotes = dict(a=r'(a)~$q_0=(-0.7,1.2)$', b=r'(b)~$q_b=(0.8,1.6)$', c=r'(c)~$q_0=(-0.6,1.0)$',
                  d=r'(d)~$q_b=(-0.3,1.45)$')
    for letter, run, key, q, label in panels:
        # poc/plot_postures_gray.py prints '%.4g' of the link-1 grid clearance above each panel.
        value(F3, quotes[letter], run,
              lambda h: h[key] == q and ('%.4g' % h['c_hat_'+key][0]).replace('-', '−') == label
              and rounds_to(h['c_hat_'+key][0], label),
              lambda h: 'panel label c1 = %s m (printed in the figure: %s)' % (
                  ('%.4g' % h['c_hat_'+key][0]).replace('-', '−'), label))
    value(F3, r'in (b) the tag itself', 'base',
          lambda h: h['link1_witness_node_qb'] == 0 and h['link1_witness_tag_distance_m_qb'] == 0.0,
          lambda h: 'node %d' % h['link1_witness_node_qb'])
    value(F3, r'in (d) the adjacent node, $8.6$~mm from the tag', 'hard',
          lambda h: h['link1_witness_node_qb'] == 1 and rounds_to(1e3*h['link1_witness_tag_distance_m_qb'], '8.6')
          and close(h['link1_witness_tag_distance_m_qb'], h['d'][0]/256, 1e-12),
          lambda h: 'node %d, %.4f mm' % (h['link1_witness_node_qb'], 1e3*h['link1_witness_tag_distance_m_qb']))

    # ---------------------------------------------------------------- Fig. 4
    F4 = 'Fig. 4'
    grid = np.load(RESULTS/'hard'/'map.npz')
    z, thr = grid['f_minus'], stored['hard']['improvement_threshold']
    fmap = report.get('fig4_map', {})
    claim(F4, r'Sampled lower bound $f^-$ on a $35\times35$ grid of postures',
          z.shape == (35, 35) and hard['map_grid'] == 35
          and close(grid['shoulder_pitch'], np.linspace(-0.6, -0.3, 35), 0) and close(grid['elbow'], np.linspace(1.0, 1.45, 35), 0)
          and abs(z[0, 0]-S['hard']['f0_minus']) < 1e-12 and abs(z[-1, -1]-S['hard']['fb_minus']) < 1e-12
          and fmap.get('grid') == [35, 35] and fmap.get('axes_exact') is True
          and fmap.get('max_abs_diff_f_minus', 1) < TRACE_TOLERANCE,
          'stored map %s, corners equal f^-(q0), f^-(qb); replay max |diff| %.2e' % (
              z.shape, fmap.get('max_abs_diff_f_minus', float('nan'))))
    claim(F4, r'hatching: samples with $f^-(q)>f^+(q_0)+0.02$',
          thr == S['hard']['f0_plus']+0.02 and 0 < int(np.sum(z > thr)) < z.size and z[-1, -1] > thr and not z[0, 0] > thr
          and fmap.get('hatched_samples_identical') is True,
          '%d of %d samples above %.7f (replay: %s)' % (
              np.sum(z > thr), z.size, thr, fmap.get('samples_above_improvement_threshold', {}).get('replayed')))
    hist = stored['bnb']['history']
    exc, ub = stored['bnb']['exclusion_threshold'], np.array([v['UB'] for v in hist])
    lbs = np.array([v['LB'] for v in hist])
    tex = (POC.parent/'figsrc'/'fig_g1map_n10_EN.tex').read_text(encoding='utf-8')
    claim(F4, r'$\UB$ crosses the exclusion threshold after 1778 subdivisions, whereas the ceiling alone is not sufficient',
          len(hist) == 1779 and [v['splits'] for v in hist] == list(range(1779)) and np.all(ub[:-1] > exc) and ub[-1] <= exc
          and np.all(np.diff(ub) <= 0) and np.all(np.diff(lbs) >= 0) and stored['bnb']['geometry']['f_cap'] > exc
          and ub[0] == S['bnb']['root_upper'] and stored['bnb']['epsilon_g'] == 0.36
          and all(('%.6f' % v) in tex for v in [exc, stored['bnb']['geometry']['f_cap']])
          and ('(%.5f,%.5f)' % (1778, ub[-1])) in tex,
          'UB(1777)=%.7f > %.7f >= UB(1778)=%.7f; f_cap=%.4f; values present in figsrc/fig_g1map_n10_EN.tex' % (
              ub[-2], exc, ub[-1], stored['bnb']['geometry']['f_cap']))

    # ---------------------------------------------------------------- reproducibility statement
    RS = 'Sec. 6.3, last par.'
    versions = {(r['provenance']['python'], r['provenance']['mujoco'], r['provenance']['numpy'],
                 r['provenance']['scipy']) for r in stored.values()}
    claim(RS, r'The stored G1 search results record Python 3.12.10, MuJoCo 3.2.6, NumPy 1.26.4 and SciPy 1.14.1',
          versions == {('3.12.10', '3.2.6', '1.26.4', '1.14.1')}, sorted(versions))
    claim(RS, r'all G1 computations use ordinary double-precision floating-point arithmetic',
          all('float64' in r['certification_status'] for r in stored.values()),
          stored['base']['certification_status'])
    claim(S61, r'the software repository contains a geometry adapter that repeats the stored searches',
          sorted(report.get('runs', {})) == sorted(RUNS) and (POC/'menagerie_shim'/'humanoid_uwb_sim'/'__init__.py').exists(),
          'replayed runs: %s; replay environment %s' % (', '.join(report.get('runs', {})), report.get('environment')))

    # ---------------------------------------------------------------- replay against the stored files
    for name in RUNS:
        run = report.get('runs', {}).get(name)
        loc = 'Replay %s' % name
        if run is None:
            claim(loc, '', False, 'not in the replay report')
            continue
        flags = run['exact']
        claim(loc, '', flags['all'] and flags['theorem_outcome'] and flags['split_count'] and flags['evaluation_count']
              and flags['selected_posture'] and flags['cover_boxes'] and flags['evaluated_posture_set_and_phases'],
              'outcome, counts, postures, evaluated set, cover and witnesses identical to %s (same evaluation order: %s)' % (
                  run['stored_result'], flags['evaluation_sequence']))
        claim(loc, '', run['headline_max_abs_diff'] < TOLERANCE,
              'article quantities: max |replay - stored| = %.2e < %.0e' % (run['headline_max_abs_diff'], TOLERANCE))
        claim(loc, '', run['trace_max_abs_diff'] < TRACE_TOLERANCE,
              'all stored numbers (every posture and box): max |replay - stored| = %.2e < %.0e' % (
                  run['trace_max_abs_diff'], TRACE_TOLERANCE))
    return claims


def article_text():
    path = os.environ.get('AMCS_ARTICLE_TEX')
    return ' '.join(Path(path).read_text(encoding='utf-8').split()) if path else None


def main():
    claims = build_claims()
    tex = article_text()
    failed = 0
    for loc, quote, ok, detail in claims:
        if tex is not None and quote and ' '.join(quote.split()) not in tex:
            ok, detail = False, 'fragment not found in the article source; '+detail
        failed += not ok
        print('%s [%s] %s%s' % ('PASS' if ok else 'FAIL', loc, ('"%s": ' % quote) if quote else '', detail))
    print('%d claims, %d failed%s' % (len(claims), failed,
                                      '' if tex is not None else ' (AMCS_ARTICLE_TEX not set: fragments not matched against the article)'))
    return 1 if failed else 0


def pytest_generate_tests(metafunc):
    if 'article_claim' in metafunc.fixturenames:
        claims = build_claims()
        metafunc.parametrize('article_claim', claims,
                             ids=['%s: %s' % (c[0], c[1][:50] or c[3][:50]) for c in claims])


def test_article_claim(article_claim):
    loc, quote, ok, detail = article_claim
    tex = article_text()
    assert tex is None or not quote or ' '.join(quote.split()) in tex, 'fragment not in the article: '+quote
    assert ok, '%s "%s": %s' % (loc, quote, detail)


if __name__ == '__main__':
    sys.exit(main())
