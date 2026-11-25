"""Checker for the estimator and calibration claims of the article (main12.tex).

Area covered:
  * Section 6.2, paragraph "Estimator mismatch" (nonlinear range estimator
    with omitted bias, seed 124) and the experiments that the paragraph and
    the Limitations paragraph attribute to the repository documentation
    (unbiased case, synthetic calibration-coverage experiment, finite
    Monte Carlo checks);
  * Section 2, paragraph "A calibration protocol" (Bonferroni-corrected
    chi-square variance intervals and the resulting weight bounds).

Every printed value is compared at its printed precision (nearest rounding)
with three sources: the originally stored result file, a fresh run of
estimator_experiments.compute() on this machine, and a reconstruction that
builds the range vectors directly from the formulas printed in the article.

Usage (repository root, see REPRODUCTION.md):
    python poc/check_article_estimator.py          # about 25 s
    python -m pytest poc/check_article_estimator.py -q
Prints one PASS/FAIL line per claim; the exit status is non-zero on failure.
The Monte Carlo replay (4 min) is not repeated here: its stored difference
report poc/results/article_estimator/replay_report.json is checked instead
(regenerate it with poc/replay_article_estimator.py).
"""
import csv
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy.stats import chi2, beta
import estimator_experiments as est
import reviewer_experiments
from g1_geometry import G1Geometry, ROOT

STORED = ROOT/'poc/results/reviewer'
REPLAY = ROOT/'poc/results/article_estimator/replay_report.json'
S62 = 'Sec. 6.2 "Estimator mismatch"'
S2 = 'Sec. 2 "A calibration protocol"'
S7 = 'Sec. 7 "Limitations"'
# The geometry adapter (poc/menagerie_shim) reproduces the authors' engine to
# about 4e-10 m. In the synthetic calibration the torso path length of link 3
# enters an exponential loss, so its weights differ by a relative 7e-9 from
# the stored ones; all other differences are at rounding level.
GEOMETRY_TOLERANCE = 1e-7

# Values printed in the article ------------------------------------------------
REPLICATES = 20000
SIGMA_M = 0.03                            # Y_i noise scale 0.03/sqrt(w_i)
RHO_M = np.array([0.03, 0.0, 0.50, 0.0])  # strong-bias vector rho
BIAS_CLEARANCE_M = 0.08
START_OFFSET_M = np.array([0.02, -0.02, 0.02])
ITERATION_LIMIT = 100
SEED = 124
Q0, QB = [-0.6, 1.0], [-0.3, 1.45]        # full-rank configuration, Sec. 6.2
ANCHOR_OFFSETS_M = [[1.8, 1.2, -0.4], [1.8, -0.3, -0.4], [-1.8, 1.2, -0.4], [0, 0, 1.8]]
PRINTED = dict(q0=dict(MSE='14.89', SE='0.101'), qb=dict(MSE='29.83', SE='0.116'))  # in 1e-3 m^2


def psi_minus(c):
    return np.clip(0.1+4*c, 0.1, 0.9)


def psi_plus(c):
    return np.clip(0.25+5*c, 0.2, 1.0)


def as_printed(value, text, scale=1e-3):
    """True if value/scale, rounded to the decimals of `text`, prints as `text`."""
    decimals = len(text.split('.')[1]) if '.' in text else 0
    return f'{value/scale:.{decimals}f}' == text


def reconstruct():
    """Strong-bias experiment built directly from the article's formulas."""
    config = json.loads((ROOT/'poc/config_hard.json').read_text())
    g = G1Geometry(config)
    z = np.random.default_rng(SEED).normal(size=(REPLICATES, 4))  # shared by both postures
    rows = {}
    for label, q in [('q0', Q0), ('qb', QB)]:
        c = np.asarray(g.clearance(q, 256))           # grid clearance, N_i = 256
        w = (psi_minus(c)+psi_plus(c))/2
        bias = RHO_M*np.clip((BIAS_CLEARANCE_M-c)/BIAS_CLEARANCE_M, 0, 1)
        y = np.linalg.norm(g.p-g.anchors, axis=1)+bias+SIGMA_M/np.sqrt(w)*z
        x, status = est.gauss_newton(y, g.anchors, w, g.p+START_OFFSET_M, ITERATION_LIMIT)
        squared = np.sum((x-g.p)**2, axis=1)
        h = (g.p-g.anchors)/np.linalg.norm(g.p-g.anchors, axis=1)[:, None]
        rows[label] = dict(c=c, w=w, bias=bias, n=len(squared), MSE=float(squared.mean()),
                           SE=float(squared.std(ddof=1)/np.sqrt(len(squared))), status=status,
                           f=float(np.linalg.eigvalsh(np.einsum('i,ij,ik->jk', w, h, h))[0]))
    return g, config, rows


def estimator_claims(add):
    stored = json.loads((STORED/'estimator/result.json').read_text())
    live = est.compute()
    g, config, rec = reconstruct()
    strong = {'stored file': stored['nonlinear_GN']['strong_mismatch'],
              'fresh run': live['nonlinear_GN']['strong_mismatch']}

    add(S62, 'full-rank configuration: q0=(-0.6,1.0), qb=(-0.3,1.45), anchor 2 at p+(1.8,-0.3,-0.4) m',
        config['q0'] == Q0 and config['Q_upper'] == QB and np.allclose(g.anchors-g.p, ANCHOR_OFFSETS_M, atol=1e-12)
        and all(rows[k]['q'] == q for rows in strong.values() for k, q in [('q0', Q0), ('qb', QB)]),
        f"config_hard.json q0={config['q0']} Q_upper={config['Q_upper']}")
    add(S62, '20 000 independent range vectors per posture',
        stored['replicates'] == live['replicates'] == REPLICATES and rec['q0']['n'] == rec['qb']['n'] == REPLICATES,
        f"stored {stored['replicates']}, fresh {live['replicates']}")
    add(S62, 'lambda_ref = (0.03 m)^-2',
        all(abs(r['lambda_ref_per_observation_m_minus2']-SIGMA_M**-2) < 1e-9 and r['reference_range_standard_deviation_m'] == SIGMA_M
            for r in (stored, live)), f"{live['lambda_ref_per_observation_m_minus2']:.6f} m^-2")
    ok = True; worst = 0.0
    for rows in strong.values():
        for label in ['q0', 'qb']:
            c = np.array(rows[label]['clearance_grid_m'])
            worst = max(worst, np.abs(rows[label]['weights']-(psi_minus(c)+psi_plus(c))/2).max(),
                        np.abs(c-rec[label]['c']).max())
    add(S62, 'w_i(q) = [psi^-(c_i) + psi^+(c_i)]/2 for the clip envelopes at the grid clearance',
        worst < 1e-9, f'max deviation of weights/clearances (stored, fresh) from the article formula {worst:.1e}')
    worst = max(np.abs(np.array(rows[label]['bias_m'])-RHO_M*np.clip((BIAS_CLEARANCE_M-np.array(rows[label]['clearance_grid_m']))/BIAS_CLEARANCE_M, 0, 1)).max()
                for rows in strong.values() for label in ['q0', 'qb'])
    add(S62, 'strong bias B_i(q) = rho_i clip((0.08 - c_i)/0.08; 0, 1), rho = (0.03, 0, 0.50, 0) m',
        worst < 1e-12 and est.BIAS_CAPS_M['strong_mismatch'] == RHO_M.tolist() and est.BIAS_CLEARANCE_M == BIAS_CLEARANCE_M
        and live['bias_caps_m']['strong_mismatch'] == RHO_M.tolist(), f'max deviation {worst:.1e}')
    add(S62, 'Gauss-Newton starts at x = p + (0.02, -0.02, 0.02) m',
        live['initial_offset_from_tag_m'] == START_OFFSET_M.tolist() and tuple(est.START_OFFSET_M) == tuple(START_OFFSET_M),
        f"fresh run {live['initial_offset_from_tag_m']}")
    iterations = [rows[label]['solver']['iterations'] for rows in strong.values() for label in ['q0', 'qb']]
    add(S62, 'backtracking and a 100-iteration limit',
        live['iteration_limit'] == est.MAX_ITERATIONS == ITERATION_LIMIT and max(iterations) == ITERATION_LIMIT
        and all('backtracking' in rows[label]['solver']['line_search'] for rows in strong.values() for label in ['q0', 'qb']),
        f'iterations used (stored q0, qb; fresh q0, qb) {iterations}; the limit is reached at q0')
    add(S62, 'seed 124, the same standard normal draws at both postures',
        stored['seed'] == live['seed'] == est.SEED == SEED, 'reconstruction uses one default_rng(124) array for both postures')
    for label, name in [('q0', 'q0'), ('qb', 'qb')]:
        for key, field, what in [('MSE', 'MSE_m2', 'mean squared error'), ('SE', 'Monte_Carlo_standard_error_m2', 'Monte Carlo standard error')]:
            values = {source: rows[label][field] for source, rows in strong.items()}
            values['article formulas'] = rec[label][key]
            add(S62, f'{what} at {name}: {PRINTED[label][key]}e-3 m^2',
                all(as_printed(v, PRINTED[label][key]) for v in values.values())
                and max(values.values())-min(values.values()) < 1e-9,
                ', '.join(f'{source} {v*1e3:.6f}e-3' for source, v in values.items()))
    add(S62, 'all terminal squared errors are averaged (non-converged replicates included)',
        all(rows['q0']['solver']['nonconverged'] == 1 and rows['qb']['solver']['nonconverged'] == 0 for rows in strong.values())
        and rec['q0']['status']['nonconverged'] == 1,
        'one of 20 000 replicates at q0 stops at the iteration limit and stays in the mean')
    gap = rec['qb']['MSE']-rec['q0']['MSE']; se = np.hypot(rec['q0']['SE'], rec['qb']['SE'])
    add(S62, 'ignored bias reverses the ordering favoured by the information model',
        all(rows['MSE_order'] == 'q0_better' and rows['oracle_information_order'] == 'qb_better'
            and rows['qb']['oracle_f_normalized'] > rows['q0']['oracle_f_normalized']
            and rows['qb']['oracle_CRLB_trace_m2'] < rows['q0']['oracle_CRLB_trace_m2'] for rows in strong.values())
        and rec['qb']['f'] > rec['q0']['f'] and gap > 10*se,
        f"f(q0)={rec['q0']['f']:.4f} < f(qb)={rec['qb']['f']:.4f}, but MSE(qb)-MSE(q0)={gap*1e3:.2f}e-3 m^2 (> 10 combined standard errors)")

    # Unbiased case, reported by the repository documentation.
    readme = (ROOT/'REPRODUCTION.md').read_text(encoding='utf-8')
    unbiased = {source: r['nonlinear_GN']['correct'] for source, r in [('stored file', stored), ('fresh run', live)]}
    ok = all(rows['MSE_order'] == 'qb_better' and not np.any(rows['q0']['bias_m']) and not np.any(rows['qb']['bias_m'])
             for rows in unbiased.values())
    texts = [f"{unbiased['stored file'][label][field]*1e3:.3f}" for label in ['q0', 'qb'] for field in ['MSE_m2', 'Monte_Carlo_standard_error_m2']]
    ok &= all(abs(unbiased['fresh run'][label]['MSE_m2']-unbiased['stored file'][label]['MSE_m2']) < 1e-9 for label in ['q0', 'qb'])
    add(S62, 'the repository documentation reports the unbiased case',
        ok and all(t in readme for t in texts),
        f"B=0: MSE {texts[0]}e-3 (q0) > {texts[2]}e-3 m^2 (qb), ordering agrees with the information model; values found in REPRODUCTION.md")
    return g


def calibration_claims(add, g):
    stored = json.loads((STORED/'calibration.json').read_text())
    n, cells, delta = stored['samples_per_cell'], stored['cells'], stored['delta']
    nu = n-1
    knots = np.array(stored['knots_m']); truth = np.array(stored['true_weights'])
    add(S2, 'm_c cells with n_c >= 2 residuals each, nu = n_c - 1, delta_2 in (0,1)',
        cells == truth.size == 4*len(knots) and n >= 2 and 0 < delta < 1,
        f'synthetic experiment: m_c={cells} (17 clearances x 4 links), n_c={n}, nu={nu}, delta_2={delta}')
    residuals = np.empty((len(knots), 4, n))
    with (STORED/'calibration_ranges.csv').open(newline='') as f:
        rows = list(csv.reader(f))[1:]
    for clearance, link, replicate, value in rows:
        residuals[int(np.argmin(np.abs(knots-float(clearance)))), int(link)-1, int(replicate)] = float(value)
    s2 = residuals.var(axis=2, ddof=1)
    low_q, high_q = chi2.ppf(delta/(2*cells), nu), chi2.ppf(1-delta/(2*cells), nu)
    variance_low, variance_high = nu*s2/high_q, nu*s2/low_q            # interval printed in the article
    lambda_ref = SIGMA_M**-2
    weight_low, weight_high = 1/variance_high/lambda_ref, 1/variance_low/lambda_ref  # reciprocals, reversed order
    deviation = max(np.abs(weight_low/np.array(stored['weight_lower'])-1).max(),
                    np.abs(weight_high/np.array(stored['weight_upper'])-1).max())
    add(S2, 'variance interval [nu s^2/chi2_{1-delta_2/(2 m_c),nu}, nu s^2/chi2_{delta_2/(2 m_c),nu}]; reciprocals in reversed order divided by lambda_ref bound the weights',
        len(rows) == cells*n and deviation < 1e-9 and np.all(variance_low < variance_high),
        f'recomputed from the {len(rows)} stored residuals; max relative deviation from stored weight bounds {deviation:.1e}')
    true_variance = SIGMA_M**2/truth
    add(S2, 'information per observation is the reciprocal of the variance; the intervals cover the true cell variances and weights',
        bool(np.all((variance_low <= true_variance) & (true_variance <= variance_high)) and np.all((weight_low <= truth) & (truth <= weight_high)))
        and stored['calibration_all_cells_covered'] is True, f'{cells} of {cells} cells covered in the stored calibration data set')
    per_cell = chi2.cdf(high_q, nu)-chi2.cdf(low_q, nu)
    exact = per_cell**cells
    add(S2, 'the intervals of all cells hold simultaneously with probability at least 1 - delta_2',
        abs(per_cell-(1-delta/cells)) < 1e-12 and 1-cells*(1-per_cell) >= 1-delta-1e-12 and exact >= 1-delta,
        f'per-cell coverage 1-delta_2/m_c={per_cell:.6f}; Bonferroni bound {1-cells*(1-per_cell):.4f}; independent cells: {exact:.5f} >= {1-delta}')
    count, repeats = stored['simultaneous_coverage_count'], stored['repeated_calibrations']
    interval = [beta.ppf(.025, count, repeats-count+1), beta.ppf(.975, count+1, repeats-count)]
    readme = (ROOT/'REPRODUCTION.md').read_text(encoding='utf-8')
    add(f'{S62} / {S7}', 'synthetic calibration-coverage experiment, reported by the repository documentation',
        repeats == 2000 and count == 1905 and stored['empirical_simultaneous_coverage'] == count/repeats
        and interval[0] <= exact <= interval[1] and interval[1] >= 1-delta
        and np.allclose(interval, stored['binomial_95_interval'], atol=1e-9) and '1905' in readme and '0.9525' in readme,
        f'{count}/{repeats} = {count/repeats} simultaneous coverage; 95% binomial interval [{interval[0]:.4f}, {interval[1]:.4f}] contains the exact value {exact:.4f}')
    with tempfile.TemporaryDirectory() as tmp:
        reviewer_experiments.calibration(g, np.random.default_rng(20261011), Path(tmp))
        fresh = json.loads((Path(tmp)/'calibration.json').read_text())
    deviation = max(np.abs(np.array(fresh[k])-np.array(stored[k])).max() for k in ['true_weights', 'weight_lower', 'weight_upper'])
    add(f'{S62} / {S7}', 'calibration-coverage experiment reproduces from its seed (20261011) on this machine',
        fresh['simultaneous_coverage_count'] == count and fresh['calibration_all_cells_covered'] and deviation < GEOMETRY_TOLERANCE,
        f"fresh run {fresh['simultaneous_coverage_count']}/{fresh['repeated_calibrations']}; max weight-bound difference {deviation:.1e} (link 3 only, torso path length of the geometry adapter)")


def monte_carlo_claims(add):
    stored = json.loads((STORED/'monte_carlo.json').read_text())
    samples = np.loadtxt(STORED/'monte_carlo_samples.csv', delimiter=',', skiprows=1)
    f, f_minus, f_plus = samples[:, 2], samples[:, 3], samples[:, 4]
    add(S7, 'finite Monte Carlo check of the index bounds (evaluated cases only)',
        stored['postures'] == 80 and stored['replicates_per_posture'] == 8 and stored['total'] == len(samples) == 640
        and stored['index_interval_failures'] == 0 and stored['clearance_bracket_failures'] == 0
        and bool(np.all((f_minus <= f) & (f <= f_plus))) and len(np.unique(samples[:, 0])) == 80,
        f'{len(samples)} stored realizations at 80 postures: f^- <= f <= f^+ in all; smallest slacks {np.min(f-f_minus):.4f}, {np.min(f_plus-f):.4f}')
    estimator = json.loads((STORED/'estimator/result.json').read_text())
    z = [row['empirical_minus_CRLB_in_standard_errors'] for row in estimator['linear_WLS']]
    add(S7, 'Monte Carlo check of the unbiased linear Gaussian estimator against the Cramer-Rao bound',
        [row['q'] for row in estimator['linear_WLS']] == [Q0, QB] and all(abs(v) < 3 for v in z)
        and all(row['replicates'] == REPLICATES for row in estimator['linear_WLS']),
        f'empirical WLS MSE minus CRLB trace: {z[0]:+.2f} and {z[1]:+.2f} Monte Carlo standard errors')
    if not REPLAY.exists():
        add(S7, 'replay of the estimator, calibration and Monte Carlo experiments on this machine', False,
            'poc/results/article_estimator/replay_report.json is missing; run poc/replay_article_estimator.py')
        return
    replay = json.loads(REPLAY.read_text())
    mc = replay['monte_carlo']
    worst_estimator = max(abs(row[k]['difference']) for row in replay['estimator'].values() if 'MSE_m2' in row
                          for k in ['MSE_m2', 'Monte_Carlo_standard_error_m2'])
    worst_mc = max(mc['max_abs_difference'].values())
    worst_linear = max(abs(v['difference']) for row in mc['linear_WLS'].values() for v in row.values())
    add(S7, 'replay of the estimator, calibration and Monte Carlo experiments on this machine',
        all(mc[k]['replay'] == mc[k]['stored'] for k in ['postures', 'replicates_per_posture', 'total', 'index_interval_failures', 'clearance_bracket_failures'])
        and mc['index_interval_failures']['replay'] == 0 and worst_mc < GEOMETRY_TOLERANCE and worst_linear < 1e-9 and worst_estimator < 1e-9
        and replay['calibration']['simultaneous_coverage_count']['replay'] == 1905
        and max(replay['calibration']['max_abs_difference'].values()) < GEOMETRY_TOLERANCE,
        f'replay vs stored: MSE/SE differences <= {worst_estimator:.1e} m^2, Monte Carlo index samples <= {worst_mc:.1e}, linear WLS <= {worst_linear:.1e}; 0 bound failures in 640 replayed realizations')


def collect():
    results = []

    def add(location, claim, passed, detail=''):
        results.append((location, claim, bool(passed), detail))
    g = estimator_claims(add)
    calibration_claims(add, g)
    monte_carlo_claims(add)
    return results


def test_article_estimator_and_calibration_claims():
    failed = [(location, claim, detail) for location, claim, passed, detail in collect() if not passed]
    assert not failed, failed


if __name__ == '__main__':
    results = collect()
    for location, claim, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'} [{location}] {claim} -- {detail}")
    failures = sum(not passed for _, _, passed, _ in results)
    print(f'{len(results)-failures}/{len(results)} claims pass')
    sys.exit(1 if failures else 0)
