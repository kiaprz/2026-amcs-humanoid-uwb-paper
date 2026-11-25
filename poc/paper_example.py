"""Independent reconstruction of the article sphere example, budgets A/B.

Float64 replay of the illustrative instance (Section 3) and of Section 6
before Subsection 6.1: Table 1, the analytic instance data and Examples 1-3.
The exact rational enclosures of the same quantities are in verified_sphere.py;
check_article_sphere.py compares both with the numbers printed in the article.
"""
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
import numpy as np
from bounds import Bounds, search

# name: e_i [m], epsilon_i, L_i^eta*eta_bar [m], N_i, e_phi [m]
BUDGETS = dict(A=(.012, .002, .010, 1000, 1e-4), B=(.004, .001, .002, 10000, 1e-5))
# Table 1 as printed: f-(q0), f+(q0), f-(pi/6), ceiling, UB-LB, right-hand side of
# the a priori width bound. Lower bounds are rounded down, upper bounds up, the
# width to the nearest digit (rounding convention of Section 6).
TABLE_MODES = ['down', 'up', 'down', 'up', 'nearest', 'up']
TABLE = dict(A=[.2729, .5654, 1.1639, 1.3734, .2094, .8931],
             B=[.2891, .5462, 1.1855, 1.3494, .1637, .3184])


def shown(value, digits, mode):
    """Round a float64 value for display: mode is 'down', 'up' or 'nearest'."""
    rounding = dict(down=ROUND_FLOOR, up=ROUND_CEILING, nearest=ROUND_HALF_EVEN)[mode]
    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=rounding))


class SphereExample:
    def __init__(self):
        self.v = np.array([[2*np.sqrt(2)/3, 0, -1/3], [-np.sqrt(2)/3, np.sqrt(6)/3, -1/3],
                           [-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3], [0, 0, 1]])
        self.p = np.zeros(3)
        self.anchors = 2*self.v
        self.Lq = np.full(4, 0.3)
        self.u = -(self.v[3]+self.v[0]/3)
        self.u /= np.linalg.norm(self.u)

    def centre(self, q):
        return (0.55-0.3*np.cos(q))*self.v[0]+0.3*np.sin(q)*self.u

    def clearance(self, q, N):
        z = self.centre(float(q[0]))
        points = np.linspace(0, 1, N+1)[None, :, None]*self.anchors[:, None, :]
        return np.linalg.norm(points-z, axis=2).min(axis=1)-0.05

    def analytic(self, q):
        z = self.centre(q)
        t = np.clip(self.anchors@z/4, 0, 1)
        return np.linalg.norm(t[:, None]*self.anchors-z, axis=1)-0.05


def budget(name):
    e, eps, relative, N, ep = BUDGETS[name]
    return dict(range_vector_m=e, direction_offset=eps, relative_geometry_m=relative,
                N=N, e_phi=ep, e_lambda=1e-6)


def nominal_indices(b, q):
    """Nominal envelope indices Phi-(q), Phi+(q) from the eigenvalue definition."""
    c = b.geometry.analytic(q)
    return b.eigen(b.env.minus(c)), b.eigen(b.env.plus(c))


def instance_data(tol=1e-12):
    """Identities of the illustrative instance and of the analytic instance data."""
    g = SphereExample()
    b = Bounds(g, budget('A'))
    v, u, h = g.v, g.u, b.h
    gram = h@h.T
    assert np.allclose(gram, (4*np.eye(4)-1)/3, atol=tol)            # h_i.h_i' = -1/3
    assert np.allclose(np.einsum('ij,ik->jk', h, h), 4/3*np.eye(3), atol=tol)
    assert np.allclose(b.d, 2, atol=tol)
    assert abs(np.linalg.norm(u)-1) < tol and abs(u@v[0]) < tol
    products = dict(v_i_dot_v_1=(v[1:]@v[0]).tolist(), u_perp_dot_v=(v[1:]@u).tolist())
    assert np.allclose(products['v_i_dot_v_1'], -1/3, atol=tol)
    assert np.allclose(products['u_perp_dot_v'], [np.sqrt(2)/3, np.sqrt(2)/3, -2*np.sqrt(2)/3], atol=tol)
    # Weights (alpha, gamma, gamma, gamma): eigenvalues alpha+gamma/3 and 4 gamma/3 (twice).
    for alpha, gamma in [(0, .9), (.2, 1), (.9, .9), (.37, .81), (1, 1)]:
        eigenvalues = np.linalg.eigvalsh(np.einsum('i,ijk->jk', [alpha, gamma, gamma, gamma], b.outer))
        assert np.allclose(eigenvalues, [alpha+gamma/3, 4*gamma/3, 4*gamma/3], atol=tol)
        assert abs(b.eigen(np.array([alpha, gamma, gamma, gamma]))-(gamma/3+alpha)) < tol
    q_A = np.arcsin(1/6)
    assert np.allclose(g.centre(0), .25*v[0], atol=tol)
    assert abs(g.analytic(0)[0]+.05) < tol and abs(g.analytic(q_A)[0]) < tol
    grid = np.linspace(0, np.pi/6, 2001)
    along, lateral, speed, projections, others = [], [], [], [], []
    for q in grid:
        z, c = g.centre(q), g.analytic(q)
        assert abs(np.linalg.norm(z-.55*v[0])-.3) < tol                # circle of radius 0.3 m
        along.append(z@v[0])
        lateral.append(np.linalg.norm(z-(z@v[0])*v[0]))
        assert abs(along[-1]-(.55-.3*np.cos(q))) < tol and abs(lateral[-1]-.3*np.sin(q)) < tol
        assert abs(c[0]-(.3*np.sin(q)-.05)) < tol                      # c_1(q,0)
        projections.append(float(np.max(v[1:]@z)))                     # z(q).v_i < 0, i=2,3,4
        assert abs(z@z-(.3925-.33*np.cos(q))) < tol
        assert np.allclose(c[1:], np.sqrt(z@z)-.05, atol=tol)          # c_i(q,0), i=2,3,4
        others.append(float(c[1]))
        speed.append(np.linalg.norm(.3*np.sin(q)*v[0]+.3*np.cos(q)*u))  # |z'(q)|
        minus, plus = nominal_indices(b, q)
        assert abs(minus-(.3+b.env.minus(c[0]))) < tol and abs(plus-(1/3+b.env.plus(c[0]))) < tol
    assert max(projections) < 0 and np.allclose(speed, .3, atol=tol)
    # Lipschitz constant L_i^q = 0.3 m/rad, checked on the sampled clearances.
    assert np.all(abs(np.diff([g.analytic(q) for q in grid], axis=0)) <= .3*np.diff(grid)[:, None]+tol)
    q_flat = np.arcsin(.2)
    phi = np.array([nominal_indices(b, q) for q in grid])
    assert np.all(abs(phi[grid <= q_flat, 0]-.3) < tol) and nominal_indices(b, q_flat+1e-6)[0] > .3+1e-7
    assert np.all(np.diff(phi, axis=0) >= -tol)                        # suprema attained at pi/6
    phi0, phi1 = nominal_indices(b, 0), nominal_indices(b, np.pi/6)
    return dict(anchor_heights_m=(1+g.anchors[:, 2]).tolist(), tag_height_m=1.0,
                d_m=b.d.tolist(), direction_gram_offdiagonal=float(gram[0, 1]),
                information_sum_diagonal=float(np.einsum('ij,ik->jk', h, h)[0, 0]),
                q_A=float(q_A), c1_at_q_A=float(g.analytic(q_A)[0]),
                c1_at_q0=float(g.analytic(0)[0]), c1_at_endpoint=float(g.analytic(np.pi/6)[0]),
                tag_to_sphere_at_q0_m=float(np.linalg.norm(g.centre(0))),
                along_v1_range_m=[float(min(along)), float(max(along))],
                lateral_range_m=[float(min(lateral)), float(max(lateral))],
                max_projection_on_other_links=max(projections),
                other_clearance_min_m=min(others), norm_z_squared_min=float(g.centre(0)@g.centre(0)),
                Lq_m_per_rad=float(max(speed)), Phi_minus_constant_until=float(q_flat),
                Phi_minus_q0=phi0[0], Phi_plus_q0=phi0[1], Phi_minus_sup=phi1[0], Phi_plus_sup=phi1[1],
                Gamma_infinity=phi1[1]-phi1[0], exclusion_margin_floor=phi0[1]-phi0[0], **products)


def lipschitz_only_intervals(b, upper, threshold, limit=10000):
    """Smallest number of equal intervals of [0, upper] for which the Lipschitz
    bounds u^L alone give UB <= threshold (Example 3)."""
    for M in range(1, limit+1):
        rho = upper/(2*M)
        if all(min(b.cap, b.at([(2*j+1)*rho])['f_plus']+b.Lf*rho) <= threshold for j in range(M)):
            return M
    raise RuntimeError('no Lipschitz-only exclusion within the limit')


def width_rhs(b, gamma, M=256):
    """Right-hand side of the a priori width bound for M equal intervals of
    [0, pi/6] with P = {q0} and the centres (beta = 0)."""
    g, env = b.geometry, b.env
    radius = np.pi/(6*M*2)                         # box radius = covering radius of P
    ec = 2*b.b['e_phi']+b.d/(2*b.b['N'])+b.b['relative_geometry_m']
    low, high = g.analytic(0)-ec, g.analytic(np.pi/6)+ec+2*b.Lq*radius
    # Lipschitz constants of the envelopes on the clearance interval of each link:
    # the slope if the interval meets the sloped part, zero if the envelope is constant.
    lam_plus = np.where((low < env.upper_offset) & (high > env.upper_offset-(env.upper_cap-env.upper_floor)/env.upper_slope),
                        env.upper_slope, 0.)
    lam_minus = np.where((low < env.lower_offset+(env.lower_cap-env.lower_floor)/env.lower_slope) & (high > env.lower_offset),
                         env.lower_slope, 0.)
    rhs = (gamma+(lam_plus+lam_minus)@ec+4*b.b['e_lambda']+2*env.upper_cap*np.sum(b.sigma)
           +2*(lam_plus@b.Lq)*radius+(lam_minus@b.Lq)*radius)
    return dict(M=M, rho_bar=radius, rho_P=radius, epsilon_c=ec.tolist(), Lambda_plus=lam_plus.tolist(),
                Lambda_minus=lam_minus.tolist(), rhs=float(rhs))


def paper_validation():
    g = SphereExample()
    results = dict(instance=instance_data())
    gamma = results['instance']['Gamma_infinity']
    for name in BUDGETS:
        N = BUDGETS[name][3]
        b = Bounds(g, budget(name))
        ref, end = b.at([0]), b.at([np.pi/6])
        expected = TABLE[name]
        apriori = width_rhs(b, gamma)
        rhs = apriori['rhs']
        actual = [ref['f_minus'], ref['f_plus'], end['f_minus'], b.cap, b.cap-end['f_minus'], rhs]
        displayed = [shown(a, 4, mode) for a, mode in zip(actual, TABLE_MODES)]
        assert displayed == expected, (name, actual)
        improve = search(b, [0], [np.pi/6], [0], candidates=[[np.pi/6]])
        exclude = search(b, [0], [.05], [0], margin=.3)
        unresolved = search(b, [0], [.05], [0], margin=0, max_splits=8)
        assert improve['theorem_outcome'] == 'guaranteed improvement'
        assert exclude['theorem_outcome'] == 'excluded improvement' and exclude['split_count'] == 0
        assert unresolved['theorem_outcome'] == 'inconclusive'
        # Example 3: the single box K_1 = Q = [0, 0.05] and the Lipschitz-only alternative.
        centre = b.at([.025])
        argument = np.array(centre['c_plus'])+b.Lq*.025
        one_box = dict(argument=argument.tolist(), weights=b.env.plus(argument).tolist(),
                       upper=b.box(np.array([0.]), np.array([.05]), centre))
        assert one_box['upper'] == exclude['UB']
        refinements = []
        for M in [1, 4, 16, 64, 256]:
            centers = (np.arange(M)+.5)*np.pi/(6*M)
            values = [b.at([q]) for q in centers]
            LB = max(ref['f_minus'], max(v['f_minus'] for v in values))
            UB = max(b.box(np.array([k*np.pi/(6*M)]), np.array([(k+1)*np.pi/(6*M)]), values[k]) for k in range(M))
            far = [v for q, v in zip(centers, values) if q >= .42]
            attained = all(v['f_minus'] == end['f_minus'] and v['w_minus'] == [.9]*4 for v in far)
            if M >= 4:
                assert shown(UB-LB, 4, 'nearest') == expected[4]
                assert LB == end['f_minus'] and far and attained
            assert UB == b.cap
            refinements.append(dict(M=M, LB=LB, UB=UB, centres_ge_042=len(far),
                                    centres_ge_042_attain_LB=attained))
        # Grid minimum's mathematical bracket checked against independent analytic geometry.
        max_gap = 0.
        for q in np.linspace(0, np.pi/6, 31):
            exact, grid = g.analytic(q), g.clearance([q], N)
            assert np.all(grid+1e-12 >= exact)
            assert np.all(grid-exact <= 2/(2*N)+1e-12)
            max_gap = max(max_gap, float(np.max(grid-exact)))
        results[name] = dict(reference=ref, endpoint=end, f_cap=b.cap, table_actual=actual,
                             table_displayed=displayed, table_expected=expected,
                             table_rounding=TABLE_MODES, sigma=b.sigma.tolist(), L_f=float(b.Lf),
                             improvement=improve['theorem_outcome'],
                             guaranteed_gain_lower=improve['guaranteed_gain_lower'],
                             exclusion=exclude['theorem_outcome'], inconclusive=unresolved['theorem_outcome'],
                             exclusion_threshold=exclude['exclusion_threshold'],
                             exclusion_UB=exclude['UB'], exclusion_splits=exclude['split_count'],
                             exclusion_slack=exclude['exclusion_threshold']-exclude['UB'], one_box=one_box,
                             lipschitz_only_intervals=lipschitz_only_intervals(b, .05, exclude['exclusion_threshold']),
                             refinements=refinements,
                             width_formula=float(gamma+(b.env.upper_cap+np.array(end['w_minus']))@b.sigma+2e-6),
                             Gamma_infinity_analytic=gamma, apriori=apriori, width_rhs=rhs,
                             empirical_max_grid_gap_m=max_gap)
    return results


def figure_data():
    """Text of the data files of Fig. 1: (plot_env.dat, plot_phi.dat).

    Panel (b): envelopes on 321 clearances; panel (c): nominal envelope indices
    and the certified bounds of budget A on 241 angles.
    """
    b = Bounds(SphereExample(), budget('A'))
    env = ['c l u']+['%.5f %.5f %.5f' % (c, b.env.minus(c), b.env.plus(c)) for c in np.linspace(-.06, .10, 321)]
    phi = ['q phim phip fmA fpA']
    for q in np.linspace(0, np.pi/6, 241):
        a, c = b.at([q]), .3*np.sin(q)-.05
        phi.append('%.6f %.6f %.6f %.6f %.6f' % (q, .3+b.env.minus(c), 1/3+b.env.plus(c), a['f_minus'], a['f_plus']))
    return '\n'.join(env)+'\n', '\n'.join(phi)+'\n'


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='write the JSON report to this file instead of printing it')
    args = parser.parse_args()
    report = json.dumps(paper_validation(), indent=2)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report+'\n')
