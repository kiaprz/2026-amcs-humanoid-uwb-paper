"""Replay of the Section 6.3 studies without humanoid-uwb-sim.

Re-runs, with the pinned Menagerie model and plain MuJoCo, the standing
simulations, the diffraction search with the 2 mm budget, the uniform-budget
searches and the local-bound study. The originally stored files in
poc/results/reviewer are NOT touched: new outputs go to
poc/results/reviewer/replay_sensitivity/, and report.json there lists the
numerical differences against the stored originals.

The chain is replayed end to end: the uniform and local studies read the
budgets from the replayed standing result, not from the stored one.

Usage (from the repository root):
    G1_XML=g1_model/g1.xml PYTHONPATH=poc/menagerie_shim \
        python poc/replay_sensitivity.py
"""
import csv
import json
import platform
import sys
import time
from pathlib import Path
import mujoco
import numpy as np
import scipy
from bounds import Bounds, search
from diffraction_model import DiffractionEnvelope, torso_intersections
from g1_geometry import G1Geometry, ROOT
import dynamic_standing
import local_experiments
import static_sensitivity

STORED = ROOT/'poc/results/reviewer'
OUT = STORED/'replay_sensitivity'
TIMING = ('elapsed', 'seconds', 'source_dynamics')   # timings and the input-path label
TOLERANCE = 1e-7   # absolute; observed maximum 3.9e-8 (hull-based clearances times envelope slopes)


def leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path+'/'+str(k))
    elif isinstance(x, list):
        for k, v in enumerate(x):
            yield from leaves(v, path+'/'+str(k))
    else:
        yield path, x


def compare(new, old):
    """Leaf-wise comparison of two JSON trees; timing fields are skipped."""
    a = {p: v for p, v in leaves(new) if not any(t in p for t in TIMING)}
    b = {p: v for p, v in leaves(old) if not any(t in p for t in TIMING)}
    shared = sorted(set(a) & set(b))
    worst, worst_path, identical, numeric, other = 0., None, 0, 0, []
    for p in shared:
        x, y = a[p], b[p]
        if isinstance(x, float) and isinstance(y, (int, float)) and not isinstance(y, bool):
            numeric += 1
            identical += x == y
            if abs(x-y) > worst:
                worst, worst_path = abs(x-y), p
        elif x != y:
            other.append(dict(path=p, replay=x, stored=y))
    return dict(float_values=numeric, bitwise_identical_floats=int(identical),
                max_abs_difference=worst, max_abs_difference_at=worst_path,
                non_float_values=len(shared)-numeric,
                non_float_mismatches=len(other), first_mismatches=other[:5],
                only_in_replay=len(set(a)-set(b)), only_in_stored=len(set(b)-set(a)))


def trace_difference(rows, stored_csv):
    """Maximum absolute difference of the decimated trace against a stored CSV."""
    with open(stored_csv, newline='') as f:
        old = list(csv.DictReader(f))
    assert len(old) == len(rows)
    worst = 0.
    for new, row in zip(rows, old):
        for k, v in row.items():
            if k == 'phase':
                assert v == new[k]
            else:
                worst = max(worst, abs(float(v)-float(new[k])))
    return dict(rows=len(rows), max_abs_difference=worst)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path):
    return json.loads(Path(path).read_text())


def diffraction_2mm():
    """Diffraction search with the assumed 2 mm budget (as reviewer_experiments.physical)."""
    config = load(ROOT/'poc/config_hard.json')
    g = G1Geometry(config)
    torso = torso_intersections(g)
    env = DiffractionEnvelope(np.linalg.norm(g.anchors-g.p, axis=1), torso)
    candidates = [config['Q_upper'], [(a+b)/2 for a, b in zip(config['Q_lower'], config['Q_upper'])]]
    r = search(Bounds(g, config['budget'], env), config['Q_lower'], config['Q_upper'], config['q0'],
               candidates=candidates, margin=.02, tolerance=.01, max_splits=32, max_seconds=30)
    r['budget'] = config['budget']
    physical = dict(torso_intersection_lengths_m=torso.tolist(), link_caps_lower=env.lower_cap.tolist(),
                    link_caps_upper=env.upper_cap.tolist(), frequency_GHz=list(env.frequency_GHz),
                    edge_distance_m=list(env.edge_distance_m),
                    torso_attenuation_dB_per_m=list(env.torso_loss_dB_per_m))
    return r, physical


def run(out=OUT):
    started = time.perf_counter()
    report = dict(
        purpose='Replay of the Section 6.3 computations; differences are replay minus stored original',
        replay_versions=dict(python=platform.python_version(), numpy=np.__version__,
                             scipy=scipy.__version__, mujoco=mujoco.__version__, platform=platform.platform()),
        stored_versions={k: load(ROOT/'poc/results/hard/result.json')['provenance'][k]
                         for k in ['python', 'numpy', 'scipy', 'mujoco']},
        skipped_fields='keys containing "elapsed" or "seconds" (timings) and "source_dynamics" (input path)',
        tolerance=TOLERANCE,
        chain='uniform and local studies use the budgets of the replayed standing result',
        comparisons={}, seconds={})
    cmp = report['comparisons']

    # 1. Standing simulations (2 ms as used in the article; 1 ms as stored variant).
    for name, timestep, old_format in [('dynamics_static', .002, 'dynamics'),
                                       ('dynamics_static_dt1ms', .001, 'dynamics_dt1ms')]:
        t = time.perf_counter()
        result, rows = dynamic_standing.run(timestep=timestep)
        result['elapsed_s'] = time.perf_counter()-t
        save(out/name/'result.json', result)
        cmp[name+'/result.json'] = compare(result, load(STORED/name/'result.json'))
        cmp[name+'/trace.csv'] = trace_difference(rows, STORED/name/'trace.csv')
        # Older stored run of the same script (fewer fields): shared fields only.
        cmp[old_format+'/result.json (shared fields)'] = compare(result, load(STORED/old_format/'result.json'))
        report['seconds'][name] = time.perf_counter()-t
        print(name, cmp[name+'/result.json']['max_abs_difference'], flush=True)
    dynamics_file = out/'dynamics_static/result.json'

    # 2. Diffraction model with the assumed 2 mm budget.
    t = time.perf_counter()
    r, physical = diffraction_2mm()
    save(out/'diffraction_assumed_2mm.json', r)
    save(out/'physical_sensitivity_inputs.json', physical)
    cmp['diffraction_assumed_2mm.json'] = compare(r, load(STORED/'diffraction_assumed_2mm.json'))
    cmp['physical_sensitivity.json (model inputs)'] = compare(physical, load(STORED/'physical_sensitivity.json'))
    report['seconds']['diffraction_2mm'] = time.perf_counter()-t

    # 3. Uniform budgets from the held postures.
    t = time.perf_counter()
    static_sensitivity.run(out/'static_sensitivity', dynamics_file)
    for f in sorted((STORED/'static_sensitivity').glob('*.json')):
        if f.name != 'tests.json':
            cmp['static_sensitivity/'+f.name] = compare(load(out/'static_sensitivity'/f.name), load(f))
    report['seconds']['static_sensitivity'] = time.perf_counter()-t

    # 4. Local bounds along the segment, tracking errors and 32 stress cases.
    t = time.perf_counter()
    local_experiments.run(out/'local', dynamics_file)
    for f in sorted((STORED/'local').glob('*.json')):
        if f.name != 'tests.json':
            cmp['local/'+f.name] = compare(load(out/'local'/f.name), load(f))
    report['seconds']['local'] = time.perf_counter()-t

    numeric = [c for c in cmp.values() if 'float_values' in c]
    report['summary'] = dict(
        files_compared=len(cmp),
        max_abs_difference=max(c['max_abs_difference'] for c in cmp.values()),
        non_float_mismatches=sum(c['non_float_mismatches'] for c in numeric),
        float_values=sum(c['float_values'] for c in numeric),
        bitwise_identical_floats=sum(c['bitwise_identical_floats'] for c in numeric),
        interpretation='outcomes, counts and selected postures are compared exactly (non-float values); '
                       'floats are compared by absolute difference')
    report['elapsed_s'] = time.perf_counter()-started
    save(out/'report.json', report)
    return report


if __name__ == '__main__':
    summary = run()['summary']
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary['non_float_mismatches'] == 0 and summary['max_abs_difference'] <= TOLERANCE else 1)
