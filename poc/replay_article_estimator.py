"""Replay of the estimator, calibration-coverage and Monte Carlo experiments.

Re-runs, on the current machine, the programs behind the "Estimator mismatch"
paragraph (Section 6.2) and the calibration protocol (Section 2) of the
article, and compares them with the originally stored outputs:

    poc/results/reviewer/estimator/result.json   (estimator_experiments.py)
    poc/results/reviewer/calibration.json        (reviewer_experiments.calibration)
    poc/results/reviewer/calibration_ranges.csv
    poc/results/reviewer/monte_carlo.json        (reviewer_experiments.monte_carlo)
    poc/results/reviewer/monte_carlo_samples.csv

The original files are never overwritten. The replayed JSON files and the
difference report are written to poc/results/article_estimator/.

Usage (repository root, see REPRODUCTION.md):
    python poc/replay_article_estimator.py            # about 4 min
    python poc/replay_article_estimator.py --skip-mc  # about 15 s, keeps the
                                                      # previous Monte Carlo replay
"""
import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
import scipy
import mujoco
import estimator_experiments
import reviewer_experiments
from g1_geometry import G1Geometry, ROOT

STORED = ROOT/'poc/results/reviewer'
REPLAY = ROOT/'poc/results/article_estimator'
REVIEWER_SEED = 20261011  # seed of reviewer_experiments.py (both modes)


def difference(new, old):
    """Largest absolute difference of two equally shaped numeric structures."""
    return float(np.max(np.abs(np.asarray(new, float)-np.asarray(old, float))))


def table(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)


def replay_estimator():
    new = estimator_experiments.run(output=REPLAY/'estimator')
    old = json.loads((STORED/'estimator/result.json').read_text())
    rows = {}
    for case, postures in new['nonlinear_GN'].items():
        for label in ['q0', 'qb']:
            a, b = postures[label], old['nonlinear_GN'][case][label]
            rows[f'{case}/{label}'] = dict(
                MSE_m2=dict(stored=b['MSE_m2'], replay=a['MSE_m2'], difference=a['MSE_m2']-b['MSE_m2']),
                Monte_Carlo_standard_error_m2=dict(stored=b['Monte_Carlo_standard_error_m2'],
                    replay=a['Monte_Carlo_standard_error_m2'],
                    difference=a['Monte_Carlo_standard_error_m2']-b['Monte_Carlo_standard_error_m2']),
                max_abs_difference=dict(clearance_grid_m=difference(a['clearance_grid_m'], b['clearance_grid_m']),
                    weights=difference(a['weights'], b['weights']), bias_m=difference(a['bias_m'], b['bias_m']),
                    oracle_f_normalized=abs(a['oracle_f_normalized']-b['oracle_f_normalized']),
                    oracle_CRLB_trace_m2=abs(a['oracle_CRLB_trace_m2']-b['oracle_CRLB_trace_m2'])),
                solver_iterations=dict(stored=b['solver']['iterations'], replay=a['solver']['iterations']),
                solver_nonconverged=dict(stored=b['solver']['nonconverged'], replay=a['solver']['nonconverged']))
        rows[f'{case}/MSE_order'] = dict(stored=old['nonlinear_GN'][case]['MSE_order'], replay=postures['MSE_order'])
    return rows


def replay_calibration(g, work):
    reviewer_experiments.calibration(g, np.random.default_rng(REVIEWER_SEED), work)
    new = json.loads((work/'calibration.json').read_text())
    old = json.loads((STORED/'calibration.json').read_text())
    (REPLAY/'calibration.json').write_text(json.dumps(new, indent=2)+'\n')
    return dict(
        simultaneous_coverage_count=dict(stored=old['simultaneous_coverage_count'], replay=new['simultaneous_coverage_count']),
        calibration_all_cells_covered=dict(stored=old['calibration_all_cells_covered'], replay=new['calibration_all_cells_covered']),
        max_abs_difference=dict(
            true_weights=difference(new['true_weights'], old['true_weights']),
            weight_lower=difference(new['weight_lower'], old['weight_lower']),
            weight_upper=difference(new['weight_upper'], old['weight_upper']),
            binomial_95_interval=difference(new['binomial_95_interval'], old['binomial_95_interval']),
            calibration_ranges_csv=difference(table(work/'calibration_ranges.csv'), table(STORED/'calibration_ranges.csv'))))


def replay_monte_carlo(g, config, work):
    reviewer_experiments.monte_carlo(g, config, np.random.default_rng(REVIEWER_SEED), work)
    new = json.loads((work/'monte_carlo.json').read_text())
    old = json.loads((STORED/'monte_carlo.json').read_text())
    (REPLAY/'monte_carlo.json').write_text(json.dumps(new, indent=2)+'\n')
    integers = ['postures', 'replicates_per_posture', 'total', 'index_interval_failures', 'clearance_bracket_failures']
    linear = {}
    for a, b in zip(new['estimator_demonstration'], old['estimator_demonstration']):
        linear[str(a['q'])] = {k: dict(stored=b[k], replay=a[k], difference=a[k]-b[k])
                               for k in ['linear_Gaussian_CRLB_trace_m2', 'WLS_empirical_MSE_m2', 'covariance_relative_frobenius_error']}
    return dict({k: dict(stored=old[k], replay=new[k]) for k in integers},
        max_abs_difference=dict(
            minimum_lower_upper_slack=difference(new['minimum_lower_upper_slack'], old['minimum_lower_upper_slack']),
            monte_carlo_samples_csv=difference(table(work/'monte_carlo_samples.csv'), table(STORED/'monte_carlo_samples.csv'))),
        linear_WLS=linear)


def run(skip_mc=False):
    started = time.perf_counter()
    REPLAY.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'poc/config_hard.json').read_text())
    report_path = REPLAY/'replay_report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    report = dict(
        purpose='Differences between a re-run on this machine and the originally stored outputs; the stored files are unchanged.',
        stored_environment='Python 3.12.10, NumPy 1.26.4, SciPy 1.14.1, MuJoCo 3.2.6 (authors\' engine humanoid-uwb-sim)',
        replay_environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                mujoco=mujoco.__version__, geometry='poc/menagerie_shim adapter'),
        seeds=dict(estimator=estimator_experiments.SEED, calibration=REVIEWER_SEED, monte_carlo=REVIEWER_SEED))
    report['estimator'] = replay_estimator()
    with tempfile.TemporaryDirectory() as tmp:
        report['calibration'] = replay_calibration(G1Geometry(config), Path(tmp))
        if skip_mc:
            if 'monte_carlo' not in previous:
                raise SystemExit('No previous Monte Carlo replay to keep; run without --skip-mc.')
            report['monte_carlo'] = previous['monte_carlo']
        else:
            report['monte_carlo'] = replay_monte_carlo(G1Geometry(config), config, Path(tmp))
    report['elapsed_s'] = time.perf_counter()-started
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-mc', action='store_true', help='keep the Monte Carlo part of the previous replay report')
    r = run(parser.parse_args().skip_mc)
    for name, row in r['estimator'].items():
        if 'MSE_m2' in row:
            print(f"{name:18s} MSE stored {row['MSE_m2']['stored']:.12g} replay {row['MSE_m2']['replay']:.12g} diff {row['MSE_m2']['difference']:.2e}")
    print('calibration', json.dumps(r['calibration']))
    print('monte_carlo', json.dumps({k: v for k, v in r['monte_carlo'].items() if k != 'linear_WLS'}))
    print('elapsed', r['elapsed_s'])
