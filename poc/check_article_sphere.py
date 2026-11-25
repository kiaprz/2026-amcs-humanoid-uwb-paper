"""Check every number of the synthetic sphere example against the article.

Covers the "Illustrative instance" paragraph, Example 1 and Fig. 1 (Section 3)
and Section 6 before Subsection 6.1 (Table 1, "Analytic instance data",
Examples 2 and 3) of main12.tex. Each printed value is compared with

* the exact rational enclosures of verified_sphere.py (primary source), and
* the float64 reconstruction of paper_example.py (identities, decisions),

under the rounding convention of Section 6: interval endpoints and lower
(upper) bounds are rounded downwards (upwards); other displayed values are
rounded to the nearest last digit. The stored outputs under poc/results and
the data files of Fig. 1 are compared with the recomputed ones.

Usage (from the repository root):
    python poc/check_article_sphere.py          # one PASS/FAIL line per claim
    python -m pytest poc/check_article_sphere.py -q
The script exits with a non-zero status if any claim fails.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'figsrc'))

import numpy as np                                    # noqa: E402
import paper_example                                  # noqa: E402
import verified_sphere                                # noqa: E402
from bounds import Envelope                           # noqa: E402

TOL = 1e-12        # agreement of float64 values with exact ones and with identities


# ---------------------------------------------------------------- rounding rules
def digits_of(printed):
    return len(printed.split('.')[1]) if '.' in printed else 0


def rounded(value, digits, mode):
    """Exact rounding of a Fraction: 'down', 'up' or 'nearest'."""
    scaled = F(value)*10**digits
    integer = dict(down=math.floor, up=math.ceil, nearest=round)[mode](scaled)
    return F(integer, 10**digits)


def ends(x):
    """(lower, upper) Fractions of an exported enclosure, a pair or a number."""
    if isinstance(x, dict):
        return F(x['lower_rational']), F(x['upper_rational'])
    if isinstance(x, (tuple, list)):
        return F(x[0]), F(x[1])
    return F(x), F(x)


def displays_as(x, printed, mode):
    """Both ends of the enclosure x round to the printed decimal string."""
    if mode == 'exact':
        return all(v == F(printed) for v in ends(x))
    return all(rounded(v, digits_of(printed), mode) == F(printed) for v in ends(x))


def text(x):
    lo, hi = ends(x)
    return '%.12g' % float(lo) if lo == hi else '[%.12g, %.12g]' % (float(lo), float(hi))


def close(a, b, tol=TOL):
    return abs(float(a)-float(b)) <= tol


def same(a, b, tol=TOL):
    """Recursive comparison of JSON-like data; floats within tol."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        return a == b
    return abs(a-b) <= tol


def table(source):
    return np.array([[float(x) for x in line.split()] for line in source.strip().split('\n')[1:]])


# ------------------------------------------------------------------ claim list
@lru_cache(maxsize=None)
def collect():
    """Return the list of (location, claim, passed, obtained) records."""
    records = []

    def claim(location, statement, passed, obtained=''):
        records.append((location, statement, bool(passed), str(obtained)))

    def shown(location, statement, x, printed, mode):
        claim(location, '%s = %s (%s)' % (statement, printed, mode), displays_as(x, printed, mode), text(x))

    exact = verified_sphere.verify()
    flt = paper_example.paper_validation()
    inst, nom = flt['instance'], exact['nominal']
    X = exact['budgets']

    # ---- Section 3, "Illustrative instance" -------------------------------
    loc = 'Sec. 3, Illustrative instance'
    article_budgets = dict(A=('.012', '.002', '.010', 1000, '.0001'), B=('.004', '.001', '.002', 10000, '.00001'))
    claim(loc, 'budgets A/B: e_i, eps_i, L^eta*eta_bar, N_i, e_phi as printed; e_lambda = 1e-6 (both programs)',
          {n: tuple(v) for n, *v in verified_sphere.BUDGETS} == article_budgets
          and all(tuple(F(str(x)) for x in paper_example.BUDGETS[n]) == tuple(F(x) for x in article_budgets[n])
                  for n in 'AB')
          and verified_sphere.E_LAMBDA.lo == verified_sphere.E_LAMBDA.hi == F(1, 10**6)
          and paper_example.budget('A')['e_lambda'] == 1e-6, verified_sphere.BUDGETS)
    env, cs = Envelope(), np.linspace(-.08, .12, 2001)
    ok = (np.allclose(env.plus(cs), np.minimum(1, np.maximum(.2, 1+40*cs)), atol=TOL, rtol=0)
          and np.allclose(env.minus(cs), np.minimum(.9, np.maximum(0, 18*(cs-.01))), atol=TOL, rtol=0))
    for k in range(-80, 121):
        c = F(k, 1000)
        ok = ok and verified_sphere.weights(verified_sphere.I(c), False).lo == min(1, max(F(1, 5), 1+40*c))
        ok = ok and verified_sphere.weights(verified_sphere.I(c), True).lo == min(F(9, 10), max(0, 18*(c-F(1, 100))))
    claim(loc, 'psi+(c) = min{1, max{0.2, 1+40c}}, psi-(c) = min{0.9, max{0, 18(c-0.01)}} (both programs)', ok)
    claim(loc, 'anchors a_i = 2 v_i: d_i = 2 m', np.allclose(inst['d_m'], 2, atol=TOL, rtol=0), inst['d_m'])
    heights = inst['anchor_heights_m']
    claim(loc, 'tag at 1 m: three anchors at 0.33 m, the fourth at 3 m',
          all(displays_as(F(h), '0.33', 'nearest') for h in heights[:3]) and close(heights[3], 3), heights)
    claim(loc, "h_i^T h_i' = -1/3", close(inst['direction_gram_offdiagonal'], -1/3), inst['direction_gram_offdiagonal'])
    claim(loc, 'sum_i h_i h_i^T = (4/3) I_3', close(inst['information_sum_diagonal'], 4/3), inst['information_sum_diagonal'])
    claim(loc, 'weights (alpha,gamma,gamma,gamma): eigenvalues alpha+gamma/3 and 4gamma/3; lambda_min = gamma/3+alpha',
          True, 'asserted in paper_example.instance_data')
    claim(loc, 'u_perp is a unit vector orthogonal to v_1; centre on a circle of radius 0.3 m around 0.55 v_1',
          True, 'asserted in paper_example.instance_data')
    claim(loc, 'for q = 0 the sphere lies on segment 1 at 0.25 m from the tag',
          close(inst['tag_to_sphere_at_q0_m'], .25) and close(inst['c1_at_q0'], -.05), inst['tag_to_sphere_at_q0_m'])
    claim(loc, 'q_A = arcsin(1/6): nominal clearance of link 1 is zero',
          close(inst['q_A'], math.asin(1/6)) and close(inst['c1_at_q_A'], 0), inst['c1_at_q_A'])

    # ---- Example 1 ---------------------------------------------------------
    loc = 'Sec. 3, Example 1'
    ref = X['A']['reference']
    shown(loc, 'c_1^-(q0) [m]', ref['c_minus'][0], '-0.0611', 'down')
    shown(loc, 'c_1^+(q0) [m]', ref['c_plus'][0], '-0.0399', 'up')
    claim(loc, 'w_1 in [0, 0.2]', displays_as(ref['w_minus'][0], '0', 'exact') and displays_as(ref['w_plus'][0], '0.2', 'exact'),
          text((ends(ref['w_minus'][0])[0], ends(ref['w_plus'][0])[1])))
    claim(loc, 'c_i^-(q0) >= 0.188 m for i = 2,3,4 (rounded down)',
          all(displays_as(c, '0.188', 'down') for c in ref['c_minus'][1:]), text(ref['c_minus'][1]))
    claim(loc, 'w_i in [0.9, 1] for i = 2,3,4',
          all(displays_as(w, '0.9', 'exact') for w in ref['w_minus'][1:])
          and all(displays_as(w, '1', 'exact') for w in ref['w_plus'][1:]))
    shown(loc, 'F(q0) lower endpoint f^-(q0)', X['A']['reference_minus'], '0.2729', 'down')
    shown(loc, 'F(q0) upper endpoint f^+(q0)', X['A']['reference_plus'], '0.5654', 'up')
    claim(loc, 'SPEB <= 3/(0.2729 lambda_ref): 0.2729 is f^-(q0) rounded down',
          F('0.2729') <= ends(X['A']['reference_minus'])[0], text(X['A']['reference_minus']))

    # ---- Fig. 1 --------------------------------------------------------------
    loc = 'Sec. 3, Fig. 1'
    import make_figs
    figsrc = ROOT/'figsrc'
    tex = make_figs.instance(make_figs.LANG['EN'])
    claim(loc, 'figure source fig_instance_n10_EN.tex is what make_figs.py generates',
          tex == (figsrc/'fig_instance_n10_EN.tex').read_text(encoding='utf-8'))
    q_A = math.asin(1/6)
    circles = ['(axis cs:%.5f,%.5f) circle[radius=0.05]' % (.55-.3*math.cos(q), .3*math.sin(q)) for q in (0, q_A, math.pi/6)]
    claim(loc, '(a) sphere of radius 0.05 at q = 0, q_A = arcsin(1/6) (tangent to link 1) and pi/6',
          all(c in tex for c in circles) and close(.3*math.sin(q_A), .05), '; '.join(circles))
    env_text, phi_text = paper_example.figure_data()
    stored_env, stored_phi = table((figsrc/'plot_env.dat').read_text()), table((figsrc/'plot_phi.dat').read_text())
    claim(loc, '(b) plot_env.dat = envelopes psi-, psi+ on [-0.06, 0.10] m',
          stored_env.shape == (321, 3) and np.allclose(stored_env, table(env_text), atol=1.5e-6, rtol=0)
          and np.allclose(stored_env[:, 1], env.minus(stored_env[:, 0]), atol=1e-5, rtol=0)
          and np.allclose(stored_env[:, 2], env.plus(stored_env[:, 0]), atol=1e-5, rtol=0), stored_env.shape)
    q = stored_phi[:, 0]
    claim(loc, '(c) plot_phi.dat = Phi-, Phi+ and budget-A bounds f-, f+ on [0, pi/6]',
          stored_phi.shape == (241, 5) and np.allclose(stored_phi, table(phi_text), atol=1.5e-6, rtol=0)
          and close(q[-1], math.pi/6, 1e-6)
          and np.allclose(stored_phi[:, 1], .3+env.minus(.3*np.sin(q)-.05), atol=1e-5, rtol=0)
          and np.allclose(stored_phi[:, 2], 1/3+env.plus(.3*np.sin(q)-.05), atol=1e-5, rtol=0)
          and np.allclose(stored_phi[[0, -1], 3:], [[float(F(X['A'][k]['lower_rational'])) for k in ks] for ks in
                          (('reference_minus', 'reference_plus'), ('endpoint_minus', 'cap'))], atol=1e-6, rtol=0),
          stored_phi.shape)
    flat = stored_phi[q <= .239, 3]
    claim(loc, '(c) plotted f- is constant on [0, 0.239] rad and increases afterwards',
          np.all(flat == flat[0]) and np.all(stored_phi[q >= .2415, 3] > flat[0]), 'f- = %.6f' % flat[0])

    # ---- Section 6, text before Table 1 ------------------------------------
    loc = 'Sec. 6, first paragraph'
    for name, gain, margin in [('A', '0.5986', '0.007641'), ('B', '0.6394', '0.04305')]:
        x = X[name]
        lo = lambda key: ends(x[key])[0]
        hi = lambda key: ends(x[key])[1]
        claim(loc, 'gain lower bound exceeds %s (budget %s; rounded down)' % (gain, name),
              lo('gain') > F(gain) and displays_as(x['gain'], gain, 'down')
              and lo('gain') <= lo('endpoint_minus')-hi('reference_plus'), text(x['gain']))
        claim(loc, 'one-box exclusion margin at eps_g = 0.3 exceeds %s (budget %s; rounded down)' % (margin, name),
              lo('exclusion_slack') > F(margin) and displays_as(x['exclusion_slack'], margin, 'down')
              and lo('exclusion_slack') <= lo('reference_minus')+F('0.3')-hi('one_box_UB'), text(x['exclusion_slack']))
        claim(loc, 'budget %s: float64 decisions on full endpoints agree with the exact ones' % name,
              flt[name]['improvement'] == 'guaranteed improvement' and flt[name]['exclusion'] == 'excluded improvement'
              and close(flt[name]['guaranteed_gain_lower'], lo('gain')) and close(flt[name]['exclusion_slack'], lo('exclusion_slack')),
              (flt[name]['improvement'], flt[name]['exclusion']))
    stored = json.loads((HERE/'results/reviewer/verified_sphere.json').read_text())
    strip = lambda r: {k: v for k, v in r.items() if k != 'elapsed_s'}
    claim(loc, 'endpoints enclosed in exact rational interval arithmetic: stored verified_sphere.json is reproduced',
          strip(stored) == strip(exact) and 'exact Fraction' in exact['arithmetic'], 'poc/results/reviewer/verified_sphere.json')
    copies = ['results/paper_validation.json']+['results/%s/paper_validation.json' % d
                                                for d in ('bnb', 'excluded', 'hard', 'inconclusive')]
    fresh = json.loads(json.dumps(flt))
    claim(loc, 'stored float64 replays paper_validation.json (5 copies) are reproduced',
          all(same(json.loads((HERE/p).read_text()), fresh) for p in copies), ', '.join(copies))

    # ---- Analytic instance data ----------------------------------------------
    loc = 'Sec. 6, Analytic instance data'
    along, lateral = inst['along_v1_range_m'], inst['lateral_range_m']
    claim(loc, 'coordinate of z(q) along v_1: 0.55-0.3cos q in [0.25, 0.30]',
          close(along[0], .25) and along[0] >= .25-TOL and along[1] <= .30 and close(along[1], .55-.3*math.cos(math.pi/6)), along)
    claim(loc, 'distance from the line of segment 1 is 0.3 sin q; c_1(q,0) = 0.3 sin q - 0.05',
          close(lateral[0], 0) and close(lateral[1], .15) and displays_as(nom['c_q0'][0], '-0.05', 'exact')
          and displays_as(nom['c_end'][0], '0.1', 'exact'), 'asserted on 2001 angles; c_1 in %s' % [text(nom['c_q0'][0]), text(nom['c_end'][0])])
    claim(loc, 'v_i^T v_1 = -1/3 for i = 2,3,4', np.allclose(inst['v_i_dot_v_1'], -1/3, atol=TOL, rtol=0), inst['v_i_dot_v_1'])
    claim(loc, 'u_perp^T v_2 = u_perp^T v_3 = sqrt(2)/3, u_perp^T v_4 = -2 sqrt(2)/3',
          np.allclose(inst['u_perp_dot_v'], [math.sqrt(2)/3, math.sqrt(2)/3, -2*math.sqrt(2)/3], atol=TOL, rtol=0), inst['u_perp_dot_v'])
    claim(loc, 'z(q)^T v_i < 0 for i = 2,3,4', inst['max_projection_on_other_links'] < 0, inst['max_projection_on_other_links'])
    claim(loc, '|z(q)|^2 = 0.3925-0.33cos q >= 0.0625; c_i(q,0) = |z(q)|-0.05 >= 0.2 m',
          close(inst['norm_z_squared_min'], .0625) and close(inst['other_clearance_min_m'], .2)
          and all(displays_as(c, '0.2', 'exact') for c in nom['c_q0'][1:]), text(nom['c_q0'][1]))
    claim(loc, "|z'(q)| = 0.3, so L_i^q = 0.3 m/rad", close(inst['Lq_m_per_rad'], .3)
          and verified_sphere.LQ.lo == verified_sphere.LQ.hi == F(3, 10) and np.all(paper_example.SphereExample().Lq == .3),
          inst['Lq_m_per_rad'])
    claim(loc, 'Phi-(q) = 0.3+psi-(c_1(q,0)), Phi+(q) = 1/3+psi+(c_1(q,0))', True, 'asserted on 2001 angles in paper_example.instance_data')
    claim(loc, 'Phi- is constant on [0, arcsin 0.2]', close(inst['Phi_minus_constant_until'], math.asin(.2)), inst['Phi_minus_constant_until'])
    claim(loc, 'sup Phi+ = 4/3', ends(nom['Phi_plus_sup']) == (F(4, 3), F(4, 3)) and close(inst['Phi_plus_sup'], 4/3), text(nom['Phi_plus_sup']))
    claim(loc, 'sup Phi- = 1.2', displays_as(nom['Phi_minus_sup'], '1.2', 'exact') and close(inst['Phi_minus_sup'], 1.2), text(nom['Phi_minus_sup']))
    claim(loc, 'Gamma_inf = 2/15 ~ 0.1333 for Q = C', ends(nom['Gamma_infinity']) == (F(2, 15), F(2, 15))
          and displays_as(nom['Gamma_infinity'], '0.1333', 'nearest') and close(inst['Gamma_infinity'], 2/15), text(nom['Gamma_infinity']))

    # ---- Table 1 -------------------------------------------------------------
    printed = dict(A=dict(sigma='0.010004', lo='0.2729', hi='0.5654', end='1.1639', cap='1.3734', width='0.2094', rhs='0.8931'),
                   B=dict(sigma='0.004001', lo='0.2891', hi='0.5462', end='1.1855', cap='1.3494', width='0.1637', rhs='0.3184'))
    for name in 'AB':
        loc, x, p, f = 'Sec. 6, Table 1 [budget %s]' % name, X[name], printed[name], flt[name]
        shown(loc, 'sigma_i', x['sigma'], p['sigma'], 'exact')
        shown(loc, 'f^-(q0)', x['reference_minus'], p['lo'], 'down')
        shown(loc, 'f^+(q0)', x['reference_plus'], p['hi'], 'up')
        shown(loc, 'f^-(pi/6)', x['endpoint_minus'], p['end'], 'down')
        shown(loc, 'calibration ceiling', x['cap'], p['cap'], 'up')
        for r in x['refinements'][1:]:
            shown(loc, 'UB-LB for M = %d' % r['M'], r['width'], p['width'], 'nearest')
        shown(loc, 'Gamma_inf', nom['Gamma_infinity'], '0.1333', 'nearest')
        a = x['apriori']
        rho = ends(a['rho_bar'])
        claim(loc, 'caption: M = 256, rho_bar = rho_P = pi/3072', a['M'] == 256 and a['rho_bar'] == a['rho_P']
              and rho[0] <= F(math.pi)/3072*(1+F(1, 10**15)) and rho[1] >= F(math.pi)/3072*(1-F(1, 10**15))
              and rho[1]-rho[0] < F(1, 10**25) and close(f['apriori']['rho_bar'], math.pi/3072), text(a['rho_bar']))
        claim(loc, 'caption: Lambda_1^+ = 40, Lambda_1^- = 18, Lambda_i^+- = 0 for i = 2,3,4',
              a['Lambda_plus'] == [40, 0, 0, 0] and a['Lambda_minus'] == [18, 0, 0, 0]
              and f['apriori']['Lambda_plus'] == [40, 0, 0, 0] and f['apriori']['Lambda_minus'] == [18, 0, 0, 0],
              (a['Lambda_plus'], a['Lambda_minus']))
        shown(loc, 'right-hand side of (12)', a['rhs'], p['rhs'], 'up')
        exact_row = [x['reference_minus'], x['reference_plus'], x['endpoint_minus'], x['cap'], x['refinements'][-1]['width'], a['rhs']]
        claim(loc, 'float64 replay agrees with the exact values and displays the printed row',
              all(close(v, ends(e)[0]) for v, e in zip(f['table_actual'], exact_row))
              and f['table_displayed'] == [float(p[k]) for k in ('lo', 'hi', 'end', 'cap', 'width', 'rhs')]
              and close(f['sigma'][0], ends(x['sigma'])[0]), f['table_displayed'])

    # ---- Example 2 -----------------------------------------------------------
    loc, x, f = 'Sec. 6, Example 2', X['A'], flt['A']
    end = x['endpoint']
    shown(loc, 'c_1(q1,0) [m]', nom['c_end'][0], '0.1', 'exact')
    shown(loc, 'c_1^-(q1) [m]', end['c_minus'][0], '0.0889', 'nearest')
    claim(loc, 'w_i^-(q1) = 0.9 for all links', all(displays_as(w, '0.9', 'exact') for w in end['w_minus']))
    sigma = ends(x['sigma'])[0]
    claim(loc, 'f^-(q1) = 1.2 - e_lambda - 0.9 sum_i sigma_i = 1.1639846 (exact)',
          displays_as(x['endpoint_minus'], '1.1639846', 'exact')
          and ends(x['endpoint_minus'])[0] == F('1.2')-F(1, 10**6)-F('0.9')*4*sigma, text(x['endpoint_minus']))
    shown(loc, 'f^+(q0) = 0.5653471... (digits truncated)', x['reference_plus'], '0.5653471', 'down')
    claim(loc, 'P = {q0, q1}: improvement by more than 0.5986 certified in every realization',
          f['improvement'] == 'guaranteed improvement' and ends(x['gain'])[0] > F('0.5986'), text(x['gain']))
    claim(loc, 'f^- is constant on [0, 0.239] rad (0.239 rounded down: f^- changes at 0.240)',
          F(x['f_minus_constant_on_0_to']) == F('0.239') and x['f_minus_changes_at_next_milliradian'],
          x['f_minus_constant_on_0_to'])
    for name in 'AB':
        rs = X[name]['refinements']
        claim(loc, 'budget %s: computed UB equals the ceiling for M = 1,4,16,64,256' % name,
              [r['M'] for r in rs] == [1, 4, 16, 64, 256] and all(r['UB_is_cap'] for r in rs)
              and all(r['UB'] == flt[name]['f_cap'] for r in flt[name]['refinements']), [r['UB_is_cap'] for r in rs])
        claim(loc, 'budget %s: from M = 4 on LB = f^-(q1), attained at every centre m_j >= 0.42 (all lower weights 0.9)' % name,
              not rs[0]['LB_is_endpoint'] and all(r['LB_is_endpoint'] and r['centres_ge_042'] > 0
                                                  and r['centres_ge_042_attain_LB'] for r in rs[1:])
              and F(X[name]['lower_weights_saturated_from']) <= F('0.42')
              and all(r['centres_ge_042_attain_LB'] and r['LB'] == flt[name]['endpoint']['f_minus']
                      for r in flt[name]['refinements'][1:]),
              'weights 0.9 proved on [%s, pi/6]; centres >= 0.42: %s' % (X[name]['lower_weights_saturated_from'],
                                                                           [r['centres_ge_042'] for r in rs]))
        width = X[name]['width_formula']
        claim(loc, 'budget %s: width Gamma_inf + sum_i (wcap_i + w_i^-(q1)) sigma_i + 2 e_lambda ~ %s equals UB-LB'
              % (name, printed[name]['width']), displays_as(width, printed[name]['width'], 'nearest')
              and all(r['width'] == width for r in rs[1:]) and close(flt[name]['width_formula'], ends(width)[0]), text(width))
        claim(loc, 'budget %s: bound (12) is conservative (right-hand side > UB-LB)' % name,
              ends(X[name]['apriori']['rhs'])[0] > ends(width)[1], '%s > %s' % (text(X[name]['apriori']['rhs']), text(width)))
    claim(loc, 'budget A: 0.42 rad is the saturation threshold rounded up (a lower weight is below 0.9 at 0.41)',
          F(x['lower_weights_saturated_from']) == F('0.42') and x['unsaturated_one_centiradian_below'],
          x['lower_weights_saturated_from'])

    # ---- Example 3 -----------------------------------------------------------
    loc, box = 'Sec. 6, Example 3', x['one_box']
    shown(loc, 'exclusion threshold f^-(q0)+eps_g', x['exclusion_threshold'], '0.5729882', 'exact')
    argument = ends(box['argument'][0])
    claim(loc, 'c_1^+(m_1)+L_1^q rho_1 < -0.0249 m (m_1 = rho_1 = 0.025; rounded up)',
          argument[1] < F('-0.0249') and displays_as(box['argument'][0], '-0.0249', 'up'), text(box['argument'][0]))
    claim(loc, 'envelope argument does not exceed -0.02 m and w_11^K = 0.2',
          argument[1] <= F('-0.02') and displays_as(box['weights'][0], '0.2', 'exact'), text(box['weights'][0]))
    shown(loc, 'u_1^K = 0.5653471... (digits truncated)', box['uK'], '0.5653471', 'down')
    claim(loc, 'u_1^K < f^-(q0)+0.3: improvement excluded with the single box (float64 search: no split)',
          ends(box['uK'])[1] < ends(x['exclusion_threshold'])[0] and box['uK'] == x['one_box_UB']
          and f['exclusion'] == 'excluded improvement' and f['exclusion_splits'] == 0
          and close(f['one_box']['upper'], ends(box['uK'])[0]) and close(f['one_box']['argument'][0], argument[0]),
          '%s < %s' % (text(box['uK']), text(x['exclusion_threshold'])))
    shown(loc, 'L_f [1/rad]', x['L_f'], '48.48', 'nearest')
    claim(loc, 'the Lipschitz bound requires 159 equal intervals (also title and abstract)',
          x['lipschitz_only_intervals'] == 159 and f['lipschitz_only_intervals'] == 159
          and close(f['L_f'], ends(x['L_f'])[0], 1e-9), x['lipschitz_only_intervals'])
    claim(loc, 'Phi+(q0) = 8/15 ~ 0.5333', ends(nom['Phi_plus_q0']) == (F(8, 15), F(8, 15))
          and displays_as(nom['Phi_plus_q0'], '0.5333', 'nearest') and close(inst['Phi_plus_q0'], 8/15), text(nom['Phi_plus_q0']))
    claim(loc, 'f^-(q0) <= Phi-(q0) = 0.3', displays_as(nom['Phi_minus_q0'], '0.3', 'exact')
          and ends(x['reference_minus'])[1] <= F('0.3') and close(inst['Phi_minus_q0'], .3), text(nom['Phi_minus_q0']))
    claim(loc, 'no certificate of this form excludes eps_g < 7/30 = Phi+(q0)-Phi-(q0)',
          ends(nom['exclusion_margin_floor']) == (F(7, 30), F(7, 30)) and close(inst['exclusion_margin_floor'], 7/30),
          text(nom['exclusion_margin_floor']))
    return tuple(records)


def report():
    failures = 0
    for location, statement, passed, obtained in collect():
        failures += not passed
        print('%s  %s: %s%s' % ('PASS' if passed else 'FAIL', location, statement,
                                 '  [computed: %s]' % obtained if obtained else ''))
    print('%d claims checked, %d failed' % (len(collect()), failures))
    return failures


def pytest_generate_tests(metafunc):
    if 'record' in metafunc.fixturenames:
        records = collect()
        metafunc.parametrize('record', records, ids=['%s: %s' % r[:2] for r in records])


def test_article_claim(record):
    location, statement, passed, obtained = record
    assert passed, '%s: %s [computed: %s]' % (location, statement, obtained)


if __name__ == '__main__':
    sys.exit(1 if report() else 0)
