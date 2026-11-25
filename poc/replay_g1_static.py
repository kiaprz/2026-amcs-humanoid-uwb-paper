"""Replay of all static G1 searches of the article and comparison with the stored results.

The article (Sections 6.1 and 6.2) reports five searches:

    base          poc/config.json, margin 0.02               -> poc/results/
    inconclusive  poc/config.json, margin 0.4, 8 subdivisions -> poc/results/inconclusive/
    excluded      poc/config.json, margin 0.8                -> poc/results/excluded/
    hard          poc/config_hard.json (margin 0.02)         -> poc/results/hard/
    bnb           poc/config_bnb.json (margin 0.36)          -> poc/results/bnb/

The stored result files come from the authors' original runs and are kept.
This script repeats every search with the geometry adapter
(poc/menagerie_shim), repeats the post-processing of poc/run.py (calibration
width, final-cover weight bounds, clearance witnesses) and the sampled map of
Fig. 4, and writes poc/results/replay_report.json with, for every run, the
maximal absolute differences against the stored file and exact-match flags
for the discrete outcomes (decision, counts, postures, covers, witnesses).
Nothing in the stored result folders is overwritten.

Usage (from the repository root; PowerShell syntax):
    $env:G1_XML="$PWD/g1_model/g1.xml"; $env:PYTHONPATH="$PWD/poc/menagerie_shim"
    python poc/replay_g1_static.py                 # all five runs and the map
    python poc/replay_g1_static.py base hard       # a subset; the report is merged
    python poc/replay_g1_static.py bnb --no-map    # the long exclusion run only

Float64 arithmetic; the replay checks reproducibility, not the error budgets.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parent
RESULTS = POC/'results'
REPORT = RESULTS/'replay_report.json'
# Tolerance for the quantities printed in the article (see headline(): reference and selected
# posture, LB, UB, thresholds, gain, geometry constants, widths, final-cover weight bounds).
TOLERANCE = 1e-9
# Sanity bound for every other stored number (all evaluated postures, boxes, witnesses).  The
# computed clearances at arbitrary postures agree with the stored ones only to a few nanometres
# (MuJoCo keeps mesh vertices in float32), and the envelope slopes (up to 40 per metre) amplify
# this in the weights; the report lists the actual maxima per field.
TRACE_TOLERANCE = 1e-7

# name -> (config file, margin override, split-limit override, stored folder)
RUNS = {
    'base': ('config.json', 0.02, None, ''),
    'inconclusive': ('config.json', 0.4, 8, 'inconclusive'),
    'excluded': ('config.json', 0.8, None, 'excluded'),
    'hard': ('config_hard.json', None, None, 'hard'),
    'bnb': ('config_bnb.json', None, None, 'bnb'),
}

# Floating-point leaves below these keys are compared exactly (postures, boxes, grid parameters).
EXACT_KEYS = {'q', 'lo', 'hi', 's', 'config'}
# Run-time measurements and environment records are not compared.
SKIPPED_KEYS = {'elapsed_s', 'search_elapsed_s', 'total_elapsed_s', 'phase_seconds',
                'provenance', 'hardware', 'assumptions'}


def run_config(name):
    """Configuration of a run: the configuration file with the documented overrides."""
    cfgfile, margin, max_splits, _ = RUNS[name]
    config = json.loads((POC/cfgfile).read_text())
    if margin is not None:
        config['epsilon_g'] = margin
    if max_splits is not None:
        config['max_splits'] = max_splits
    return config


def stored_path(name):
    return RESULTS/RUNS[name][3]/'result.json'


def replay(name):
    """Repeat one search and the post-processing of poc/run.py; return (result, geometry, bounds)."""
    from bounds import Bounds, Envelope, search
    from g1_geometry import G1Geometry
    config = run_config(name)
    g = G1Geometry(config)
    b = Bounds(g, config['budget'], Envelope(**config['envelope']))
    rng = np.random.default_rng(config['seed'])
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])

    def candidates():
        return [lo, hi, (lo+hi)/2]+list(rng.uniform(lo, hi, (config['candidate_count'], len(lo))))
    result = search(b, lo, hi, config['q0'], candidates=candidates, margin=config['epsilon_g'],
                    tolerance=config['epsilon_o'], max_splits=config['max_splits'],
                    max_seconds=config.get('max_seconds'))
    # Same derived fields as poc/run.py (which needs the original simulator checkout to start).
    result.update(config=config, geometry=dict(p_m=g.p.tolist(), anchors_m=g.anchors.tolist(),
                  d_m=b.d.tolist(), h=b.h.tolist(), sigma=b.sigma.tolist(), f_cap=b.cap, **g.constants))
    mu = float(np.linalg.eigvalsh(b.h.T@b.h)[0])
    result['saturation_width'] = dict(mu_min=mu,
        calibration=(b.env.upper_cap-b.env.lower_cap)*mu,
        corrections=(b.env.upper_cap+b.env.lower_cap)*float(sum(b.sigma))+2*b.b['e_lambda'],
        applies_to_saturated_easy_example=bool(np.allclose(result['selected']['w_minus'], b.env.lower_cap)))
    cover = result['remaining_boxes']+result['pruned_boxes']
    if cover:
        wk = np.array([b.env.plus(np.array(v['center']['c_plus'])+b.geometry.Lq*np.linalg.norm((np.array(v['hi'])-v['lo'])/2)) for v in cover])
        result['cover_weight_upper_max'] = wk.max(axis=0).tolist()
        result['cover_excludes_simultaneous_upper_saturation'] = bool(np.all(np.any(wk < b.env.upper_cap, axis=1)))
    result['witnesses'] = {key: g.clearance(result[key]['q'], int(b.b['N']), witnesses=True)[1]
                           for key in ['reference', 'selected']}
    return json.loads(json.dumps(result, allow_nan=False)), g, b


def headline(result):
    """Flat record of every quantity of a result that the article prints or uses.

    Works on a stored or a replayed result; needs only NumPy.  Derived values
    (rank and eigenvalues of the direction matrices, widths, slacks, the
    distance of the link-1 witness node from the tag) are computed here from
    the full stored float64 values.
    """
    config, geo = result['config'], result['geometry']
    env = config['envelope']
    h = np.array(geo['h'])
    ref, sel = result['reference'], result['selected']
    lb, ub = result['LB'], result['UB']
    rest = h[1:].T@h[1:]
    out = dict(
        outcome=result['theorem_outcome'], stop_reason=result['stop_reason'], epsilon_g=result['epsilon_g'],
        split_count=result['split_count'], evaluation_count=result['evaluation_count'],
        cache_hits=result['cache_hits'],
        phase_counts={p: sum(a['phase'] == p for a in result['evaluations'])
                      for p in ['reference', 'candidate', 'box center']},
        q0=ref['q'], qb=sel['q'], f0_minus=ref['f_minus'], f0_plus=ref['f_plus'],
        fb_minus=sel['f_minus'], fb_plus=sel['f_plus'],
        c_hat_q0=ref['c_hat'], c_hat_qb=sel['c_hat'],
        w_minus_q0=ref['w_minus'], w_minus_qb=sel['w_minus'], w_plus_qb=sel['w_plus'],
        LB=lb, UB=ub, gain_lower=result['guaranteed_gain_lower'],
        width=None if lb is None else ub-lb,
        exclusion_threshold=result['exclusion_threshold'], improvement_threshold=result['improvement_threshold'],
        exclusion_slack=result['exclusion_threshold']-ub, root_upper=result['root_upper'],
        f_cap=geo['f_cap'], p=geo['p_m'], anchor_offsets=(np.array(geo['anchors_m'])-geo['p_m']).tolist(),
        d=geo['d_m'], sigma=geo['sigma'], Lq=geo['Lq_m_per_rad'], joint_reach=geo['joint_reach_m'],
        moving_bodies=geo['moving_bodies'],
        # Gamma_infinity of the separate-envelope construction: (cap spread) * lambda_min(sum h h^T).
        calibration_width=(env['upper_cap']-env['lower_cap'])*float(np.linalg.eigvalsh(h.T@h)[0]),
        calibration_width_recorded=result['saturation_width']['calibration'],
        # Links 2, 3, 4 alone: rank of their directions and lambda_min of the matrix with the lower cap.
        rank_h234=int(np.linalg.matrix_rank(h[1:], tol=1e-12)),
        lambda_min_234_lower_cap=env['lower_cap']*float(np.linalg.eigvalsh(rest)[0]),
        cover_weight_upper_max=result.get('cover_weight_upper_max'),
        cover_excludes_simultaneous_upper_saturation=result.get('cover_excludes_simultaneous_upper_saturation'),
        search_elapsed_s=result['search_elapsed_s'])
    for key, tag in [('reference', 'q0'), ('selected', 'qb')]:
        w = result['witnesses'][key][0]
        out['link1_witness_node_'+tag] = w['node']
        out['link1_witness_tag_distance_m_'+tag] = float(np.linalg.norm(np.array(w['point_m'])-geo['p_m']))
    return out


def compare(stored, new):
    """Maximal absolute differences per top-level field and the paths of all exact mismatches."""
    numeric, mismatches = {}, []

    def walk(a, b, path, group, exact):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in SKIPPED_KEYS:
                    continue
                if k not in a or k not in b:
                    mismatches.append(path+'/'+k+' (missing)')
                    continue
                # Differences are grouped by top-level field and innermost key, e.g. evaluations.c_hat.
                walk(a[k], b[k], path+'/'+k, group.split('.')[0]+'.'+k if group else k,
                     exact or k in EXACT_KEYS)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append(path+' (length)')
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path+'/'+str(i), group, exact)
        elif (isinstance(a, float) or isinstance(b, float)) and not exact \
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
            numeric[group] = max(numeric.get(group, 0.0), abs(a-b))
        elif a != b or type(a) is not type(b):
            mismatches.append(path)

    def by_box(boxes):
        return sorted(boxes, key=lambda v: (v['lo'], v['hi']))
    a, b = dict(stored), dict(new)
    same_order = all([v['lo'] for v in a[k]] == [v['lo'] for v in b[k]] and
                     [v['hi'] for v in a[k]] == [v['hi'] for v in b[k]] for k in ['remaining_boxes', 'pruned_boxes'])
    for k in ['remaining_boxes', 'pruned_boxes']:
        a[k], b[k] = by_box(a[k]), by_box(b[k])
    # Boxes whose upper bounds agree to rounding may be split in a different order, which permutes
    # the evaluation sequence without changing the evaluated set; the order is reported separately
    # and the evaluations are matched by posture.
    same_sequence = [v['q'] for v in a['evaluations']] == [v['q'] for v in b['evaluations']]
    for r in (a, b):
        r['evaluations'] = sorted(r['evaluations'], key=lambda v: v['q'])
    walk(a, b, '', None, False)

    def clean(prefix):
        return not any(m.startswith(prefix) for m in mismatches)
    exact = dict(
        theorem_outcome=clean('/theorem_outcome'), stop_reason=clean('/stop_reason'),
        split_count=clean('/split_count'), evaluation_count=clean('/evaluation_count'),
        cache_hits=clean('/cache_hits'),
        reference_posture=clean('/reference/q'), selected_posture=clean('/selected/q'),
        evaluated_posture_set_and_phases=clean('/evaluations'),
        history_counts=clean('/history'),
        cover_boxes=clean('/remaining_boxes') and clean('/pruned_boxes'),
        # Informative only (not part of 'all'): identical order of evaluations and of the heap.
        evaluation_sequence=bool(same_sequence), cover_box_order=bool(same_order),
        witness_nodes_and_solids=clean('/witnesses'),
        moving_solids=clean('/geometry'), configuration=clean('/config'),
        derived_flags=clean('/saturation_width') and clean('/cover_excludes'),
        all=not mismatches)
    return numeric, exact, mismatches


def headline_difference(a, b):
    """Maximal absolute difference of the floating-point entries of two headline() records."""
    worst = 0.0
    for k in a:
        if k == 'search_elapsed_s':
            continue
        x, y = a[k], b[k]
        if isinstance(x, float) or (isinstance(x, list) and x and isinstance(x[0], (float, list))):
            worst = max(worst, float(np.max(np.abs(np.array(x, float)-np.array(y, float)))))
    return worst


def model_facts(g, stored):
    """Facts about the pinned model that the article states; evaluated on the loaded MuJoCo model."""
    import mujoco
    m = g.robot.model
    xml = Path(mujoco_path())
    manifest = stored['provenance']['model_manifest']
    verified = stored['provenance']['model_verified']
    bad = [f['path'] for f in manifest['files']
           if not (xml.parent/f['path']).exists()
           or hashlib.sha256((xml.parent/f['path']).read_bytes()).hexdigest() != f['sha256']]
    position = [bool(m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                     and m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                     and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                     and m.actuator_biasprm[a, 1] == -m.actuator_gainprm[a, 0])
                for a in range(m.nu)]
    name = lambda kind, i: mujoco.mj_id2name(m, kind, int(i))
    hand = [gid for gid in range(m.ngeom) if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
            and 'rubber_hand' in name(mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid])
            and name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]).startswith('left_')]
    types = {int(mujoco.mjtGeom.mjGEOM_MESH): 'mesh', int(mujoco.mjtGeom.mjGEOM_CYLINDER): 'cylinder'}
    torso = g.robot._bid['torso_link']
    return dict(
        g1_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        g1_xml_sha256_matches_stored=hashlib.sha256(xml.read_bytes()).hexdigest() == verified['g1_xml_sha256'],
        manifest_revision=manifest['requested_revision'], manifest_file_count=len(manifest['files']),
        manifest_files_mismatched=bad,
        actuator_count=int(m.nu), position_actuator_count=int(sum(position)),
        actuated_joint_count=len({int(m.actuator_trnid[a, 0]) for a in range(m.nu)}),
        hinge_joint_count=int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)),
        free_joint_count=int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
        moving_solids=[dict(geom_id=int(gid), body=name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]),
                            type=types[int(m.geom_type[gid])], convex_hull=hull is not None,
                            collision=bool(m.geom_contype[gid] or m.geom_conaffinity[gid]))
                       for gid, hull, _ in g.geoms],
        left_rubber_hand_geoms=[dict(geom_id=int(gid), body=name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]),
                                     group=int(m.geom_group[gid]),
                                     collision=bool(m.geom_contype[gid] or m.geom_conaffinity[gid]),
                                     in_L=gid in {x[0] for x in g.geoms}) for gid in hand],
        tag_body='torso_link', torso_origin_m=g.robot.data.xpos[torso].tolist(),
        torso_rotation_is_identity=bool(np.allclose(g.robot.data.xmat[torso].reshape(3, 3), np.eye(3), atol=1e-15)),
        controlled_joint_ranges={j: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)].tolist()
                                 for j in g.joints})


def mujoco_path():
    import os
    return os.environ['G1_XML']


def tag_is_fixed(g, config):
    """The tag and the torso do not move when the controlled joints sweep the corners of Q."""
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])
    worst = 0.0
    for q in [lo, hi, [lo[0], hi[1]], [hi[0], lo[1]], config['q0']]:
        g.pose(q)
        worst = max(worst, float(np.max(np.abs(g.robot.tag_position()-g.p))))
    return worst


def replay_map(g, b, config):
    """Sampled map of f^- of Fig. 4(a), exactly as in poc/run.py; compared with hard/map.npz."""
    n = config.get('map_grid', 35)
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])
    x, y = [np.linspace(lo[j], hi[j], n) for j in range(2)]
    z = np.array([[b.at([a, c])['f_minus'] for a in x] for c in y])
    stored = np.load(RESULTS/'hard'/'map.npz')
    threshold = json.loads((RESULTS/'hard'/'result.json').read_text())['improvement_threshold']
    return dict(grid=[int(len(x)), int(len(y))],
                axes_exact=bool(np.array_equal(x, stored['shoulder_pitch']) and np.array_equal(y, stored['elbow'])),
                max_abs_diff_f_minus=float(np.max(np.abs(z-stored['f_minus']))),
                stored_range=[float(stored['f_minus'].min()), float(stored['f_minus'].max())],
                samples_above_improvement_threshold=dict(stored=int(np.sum(stored['f_minus'] > threshold)),
                                                         replayed=int(np.sum(z > threshold))),
                hatched_samples_identical=bool(np.array_equal(z > threshold, stored['f_minus'] > threshold)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='*', choices=[*RUNS, []], help='runs to replay (default: all)')
    parser.add_argument('--no-map', action='store_true', help='skip the 35x35 map of Fig. 4(a)')
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args(argv)
    names = args.runs or list(RUNS)
    import mujoco
    import scipy
    report = json.loads(args.report.read_text()) if args.report.exists() else {}
    report.update(
        description='Replay of the static G1 searches with the geometry adapter; differences are '
                    'replayed minus stored values (see poc/replay_g1_static.py).',
        tolerance=TOLERANCE, trace_tolerance=TRACE_TOLERANCE,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         mujoco=mujoco.__version__, platform=platform.platform()))
    report.setdefault('runs', {})
    for name in names:
        stored = json.loads(stored_path(name).read_text())
        started = time.perf_counter()
        new, g, b = replay(name)
        elapsed = time.perf_counter()-started
        numeric, exact, mismatches = compare(stored, new)
        worst = max(numeric.values(), default=0.0)
        head_stored, head_new = headline(stored), headline(new)
        head = headline_difference(head_stored, head_new)
        report['runs'][name] = dict(
            stored_result=stored_path(name).relative_to(POC.parent).as_posix(),
            config_file='poc/'+RUNS[name][0], margin=new['epsilon_g'], max_splits=new['config']['max_splits'],
            replay_seconds=elapsed,
            headline_max_abs_diff=head, headline_within_tolerance=bool(head < TOLERANCE),
            trace_max_abs_diff=worst, trace_within_tolerance=bool(worst < TRACE_TOLERANCE),
            trace_max_abs_diff_by_field=numeric,
            exact=exact, exact_mismatches=mismatches[:50],
            stored=head_stored, replayed=head_new)
        if name in ('base', 'hard'):
            key = 'baseline' if name == 'base' else 'full_rank'
            report.setdefault('model', {})[key] = dict(model_facts(g, stored),
                                                       tag_displacement_over_Q_m=tag_is_fixed(g, new['config']))
        if name == 'hard' and not args.no_map:
            report['fig4_map'] = replay_map(g, b, new['config'])
        print('%-13s %-23s splits=%-5d evals=%-5d headline|diff|=%.3e trace|diff|=%.3e exact=%s (%.1f s)' % (
            name, new['theorem_outcome'], new['split_count'], new['evaluation_count'], head, worst,
            exact['all'], elapsed), flush=True)
        args.report.write_text(json.dumps(report, indent=1, allow_nan=False)+'\n')
    done = [n for n in RUNS if n in report['runs']]
    ok = all(report['runs'][n]['headline_within_tolerance'] and report['runs'][n]['trace_within_tolerance']
             and report['runs'][n]['exact']['all'] for n in done)
    print('report:', args.report, '| runs in report:', ', '.join(done), '|', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.path.insert(0, str(POC))
    sys.exit(main())
