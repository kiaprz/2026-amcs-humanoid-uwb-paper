"""Exact rational interval replay of the analytic sphere, without NumPy.

Integer arithmetic supplies outward enclosures. sqrt is bracketed using
integer square roots; sine/cosine use a Taylor polynomial and its absolute
remainder bound. This certifies arithmetic, conditional on the stated model
and rational input budgets, rather than calibrating a real radio channel.
"""
from fractions import Fraction as F
from math import isqrt, factorial
from dataclasses import dataclass
from functools import lru_cache
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
import json
import time


@dataclass(frozen=True)
class I:
    lo: F
    hi: F

    def __init__(self, lo, hi=None):
        object.__setattr__(self, 'lo', F(lo))
        object.__setattr__(self, 'hi', F(lo if hi is None else hi))
        assert self.lo <= self.hi

    def __add__(self, other):
        b = interval(other)
        return I(self.lo+b.lo, self.hi+b.hi)
    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+-interval(other)

    def __rsub__(self, other):
        return interval(other)+-self

    def __mul__(self, other):
        b = interval(other)
        v = [a*c for a in (self.lo, self.hi) for c in (b.lo, b.hi)]
        return I(min(v), max(v))
    __rmul__ = __mul__

    def __truediv__(self, other):
        b = interval(other)
        assert not b.lo <= 0 <= b.hi
        return self*I(1/b.hi, 1/b.lo)

    def square(self):
        v = [self.lo**2, self.hi**2]
        return I(0 if self.lo <= 0 <= self.hi else min(v), max(v))

    def sqrt(self, digits=45):
        assert self.lo >= 0
        scale = 10**digits
        lower = isqrt(self.lo.numerator*scale**2//self.lo.denominator)
        upper = isqrt(self.hi.numerator*scale**2//self.hi.denominator)
        if upper*upper != self.hi*scale**2:   # keep exact roots exact
            upper += 1
        return I(F(lower, scale), F(upper, scale))

    def clip(self, low, high):
        return I(max(F(low), min(F(high), self.lo)), max(F(low), min(F(high), self.hi)))

    def outward(self, digits=50):
        """Enclosing interval with endpoints rounded outward to short decimals."""
        scale = 10**digits
        return I(F((self.lo*scale).__floor__(), scale), F((self.hi*scale).__ceil__(), scale))

    def export(self):
        def dec(v, mode):
            with localcontext() as c:
                c.prec = 25
                c.rounding = mode
                return str(Decimal(v.numerator)/Decimal(v.denominator))
        return dict(lower_rational=str(self.lo), upper_rational=str(self.hi),
                    lower_decimal=dec(self.lo, ROUND_FLOOR), upper_decimal=dec(self.hi, ROUND_CEILING))


def interval(x):
    return x if isinstance(x, I) else I(x)


@lru_cache(maxsize=None)
def trig(x, cosine=False):
    x = F(x)
    assert abs(x) <= 1
    n = 24
    value = sum(((-1)**k*x**(2*k+(0 if cosine else 1))/factorial(2*k+(0 if cosine else 1))
                 for k in range(n)), F(0))
    # Taylor theorem: every derivative has absolute value at most one.
    order = 2*n if cosine else 2*n+1
    error = abs(x)**order/factorial(order)
    return I(value-error, value+error)


def grid_clearance(N, sine, cosine):
    zeta = I('0.55')-I('0.3')*cosine
    lateral = I('0.3')*sine
    # Nearest node is among nodes bracketing the uncertain projection.
    first = (zeta.lo*N/2).__floor__()
    last = (zeta.hi*N/2).__ceil__()
    distances = [((I(F(2*k, N))-zeta).square()+lateral.square()).sqrt()-I('.05')
                 for k in range(first, last+1)]
    c1 = I(min(v.lo for v in distances), min(v.hi for v in distances))
    # Other rays have negative projections (proved in the article).
    other = (zeta.square()+lateral.square()).sqrt()-I('.05')
    return [c1, other, other, other]


def weights(c, lower):
    return (I(18)*(c-I('.01'))).clip(0, '.9') if lower else (I(1)+I(40)*c).clip('.2', 1)


def eig(w):
    # Tetrahedral directions: three equal weights gamma, one alpha.
    assert all(v == w[1] for v in w[2:])
    a, g = w[0], w[1]
    x, y = a+g/3, g*F(4, 3)
    return I(min(x.lo, y.lo), min(x.hi, y.hi))


def pi_interval(terms=40, digits=40):
    """Enclose pi by Machin's formula; alternating series give two-sided bounds."""
    def atan_inverse(n):
        partial = [F(0)]
        for k in range(terms+1):
            partial.append(partial[-1]+F((-1)**k, (2*k+1)*n**(2*k+1)))
        return I(min(partial[-2:]), max(partial[-2:]))
    value = 16*atan_inverse(5)-4*atan_inverse(239)
    # Outward rounding to a short denominator keeps later powers small.
    return value.outward(digits)


PI = pi_interval()
E_LAMBDA = I('.000001')
LQ = I('.3')                      # L_i^q [m/rad], all links
KNOTS = {True: (F('.01'), F('.06'), 18), False: (F('-.02'), F(0), 40)}  # sloped part of psi-/psi+
BUDGETS = [('A', '.012', '.002', '.010', 1000, '.0001'),
           ('B', '.004', '.001', '.002', 10000, '.00001')]


def sincos(q):
    """Enclose (sin q, cos q) for an interval q in [0, 1] (both are monotone there)."""
    q = interval(q)
    assert 0 <= q.lo and q.hi <= 1
    return (I(trig(q.lo).lo, trig(q.hi).hi).outward(),
            I(trig(q.hi, True).lo, trig(q.lo, True).hi).outward())


def nominal_clearance(sine, cosine):
    """Exact nominal clearances c_i(q,0): link 1 and the three other links."""
    zeta = I('0.55')-I('0.3')*cosine
    lateral = I('0.3')*sine
    other = (zeta.square()+lateral.square()).sqrt()-I('.05')
    return [lateral-I('.05'), other, other, other]


def lowest(values):
    return I(min(v.lo for v in values), min(v.hi for v in values))


def highest(values):
    return I(max(v.lo for v in values), max(v.hi for v in values))


def slope_on(lower, lo, hi):
    """Lipschitz constant of psi- (lower=True) or psi+ on the interval [lo, hi]."""
    first, last, slope = KNOTS[lower]
    return slope if lo < last and hi > first else 0


class Budget:
    def __init__(self, e, eps, rel, N, ep):
        self.N, self.e_phi, self.rel = N, I(ep), I(rel)
        self.grid = I(F(1, N))                       # d_i/(2 N_i) with d_i = 2 m
        self.sigma = I(e)/2+I(eps)*(2+I(eps))
        self.cap = I(F(4, 3))+E_LAMBDA+4*self.sigma  # all upper weight caps equal 1
        self.Lf = 4*I(40)*LQ*(1+self.sigma)
        self.eps_c = 2*self.e_phi+self.grid+self.rel

    def at(self, sine, cosine):
        c = grid_clearance(self.N, sine, cosine)
        cm = [v-self.e_phi-self.grid-self.rel for v in c]
        cp = [v+self.e_phi+self.rel for v in c]
        wm, wp = [weights(v, True) for v in cm], [weights(v, False) for v in cp]
        return dict(c_hat=c, c_minus=cm, c_plus=cp, w_minus=wm, w_plus=wp,
                    f_minus=eig(wm)-E_LAMBDA-sum(wm, I(0))*self.sigma,
                    f_plus=eig(wp)+E_LAMBDA+sum(wp, I(0))*self.sigma)

    def box(self, centre, rho):
        """Box bound min(u^K, u^L) from the centre evaluation and the radius."""
        argument = [v+LQ*rho for v in centre['c_plus']]
        w = [weights(v, False) for v in argument]
        uK = eig(w)+E_LAMBDA+sum(w, I(0))*self.sigma
        uL = lowest([self.cap, centre['f_plus']+self.Lf*rho])
        return dict(argument=argument, weights=w, uK=uK, uL=uL, upper=lowest([uK, uL]))

    def constant_until(self, ref):
        """Largest k/1000 such that f- is proved constant on [0, k/1000].

        The projection of the sphere centre on segment 1 stays inside the
        segment, so a grid node lies within d/(2N) of it and the computed
        clearance is at most sqrt((0.3 sin q)^2+(d/(2N))^2)-0.05, which is
        nondecreasing in q; the other lower weights stay at the cap 0.9.
        """
        assert all(w == I('.9') for w in ref['w_minus'][1:])
        k = 0
        while True:
            lateral = I('.3')*trig(F(k+1, 1000))
            c = (lateral.square()+self.grid.square()).sqrt()-I('.05')
            if weights(c-self.e_phi-self.grid-self.rel, True).hi > 0:
                break
            k += 1
        changed = self.at(*sincos(F(k+1, 1000)))['f_minus']
        return F(k, 1000), changed.lo > ref['f_minus'].hi

    def saturated_from(self):
        """Smallest k/100 such that all lower weights equal 0.9 on [k/100, pi/6].

        The grid minimum is never below the nominal clearance 0.3 sin q-0.05,
        which is increasing in q.
        """
        k = (PI.lo/6*100).__floor__()
        def saturated(j):
            c = I('.3')*trig(F(j, 100))-I('.05')
            return weights(c-self.e_phi-self.grid-self.rel, True) == I('.9')
        assert saturated(k)
        while saturated(k-1):
            k -= 1
        below = self.at(*sincos(F(k-1, 100)))['w_minus'][0]
        return F(k, 100), below.hi < F('.9')

    def lipschitz_intervals(self, upper, threshold, limit=10000):
        """Smallest number M of equal intervals of [0, upper] for which the
        Lipschitz bounds u^L alone satisfy max_j u_j^L <= threshold."""
        for M in range(1, limit+1):
            rho = F(upper)/(2*M)
            values = []
            for j in range(M):
                uL = lowest([self.cap, self.at(*sincos((2*j+1)*rho))['f_plus']+self.Lf*rho])
                values.append(uL)
                if uL.lo > threshold.hi:      # rigorous failure of this M
                    break
            else:
                assert highest(values).hi <= threshold.lo
                return M
            assert uL.lo > threshold.hi
        raise AssertionError('no Lipschitz-only exclusion within the limit')

    def refinement(self, M, ref, endpoint):
        """LB and UB for M equal intervals of [0, pi/6]; P = {q0} and the centres."""
        rho = PI/(12*M)
        centres = [(rho*(2*j+1)).outward(20) for j in range(M)]
        values = [self.at(*sincos(q)) for q in centres]
        LB = highest([ref['f_minus']]+[v['f_minus'] for v in values])
        UB = highest([self.box(v, rho)['upper'] for v in values])
        # Centres at or beyond 0.42 rad: all lower weights 0.9 and f- = f-(pi/6).
        far = [v for q, v in zip(centres, values) if q.lo >= F('.42')]
        return dict(M=M, LB=LB, UB=UB, width=UB-LB, centres_ge_042=len(far),
                    LB_is_endpoint=LB == endpoint['f_minus'], UB_is_cap=UB == self.cap,
                    centres_ge_042_attain_LB=all(v['f_minus'] == endpoint['f_minus']
                                                 and all(w == I('.9') for w in v['w_minus'])
                                                 for v in far))


def export(value):
    if isinstance(value, I):
        return value.export()
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: export(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [export(v) for v in value]
    return value


def verify():
    started = time.perf_counter()
    output = dict(arithmetic='exact Fraction intervals; integer sqrt brackets; Taylor remainder bounds',
                  status='arithmetic certificate conditional on analytic geometry and declared budgets', budgets={})
    # Nominal envelope indices (Section 5): Phi+- are nondecreasing in q, because
    # every nominal clearance is, so their suprema over [0, pi/6] are attained at pi/6.
    start, end = nominal_clearance(I(0), I(1)), nominal_clearance(I('.5'), I(3).sqrt()/2)
    phi = {(name, lower): eig([weights(c, lower) for c in cs])
           for name, cs in [('q0', start), ('end', end)] for lower in [True, False]}
    gamma = phi['end', False]-phi['end', True]
    output['nominal'] = export(dict(
        c_q0=start, c_end=end, Phi_minus_q0=phi['q0', True], Phi_plus_q0=phi['q0', False],
        Phi_minus_sup=phi['end', True], Phi_plus_sup=phi['end', False], Gamma_infinity=gamma,
        # Example 3: no certificate of this form excludes a margin below this value.
        exclusion_margin_floor=phi['q0', False]-phi['q0', True], pi=PI))
    for name, e, eps, rel, N, ep in BUDGETS:
        b = Budget(e, eps, rel, N, ep)
        ref = b.at(I(0), I(1))
        endpoint = b.at(I('.5'), I(3).sqrt()/2)
        gain = endpoint['f_minus']-ref['f_plus']
        # One-box bound, Q=[0,0.05], centre=radius=0.025.
        centre = b.at(*sincos('.025'))
        one = b.box(centre, I('.025'))
        ub = one['uK']
        threshold = ref['f_minus']+I('.3')
        exclusion_slack = threshold-ub
        assert gain.lo > F('.59')
        assert exclusion_slack.lo > 0
        refinements = [b.refinement(M, ref, endpoint) for M in [1, 4, 16, 64, 256]]
        # Right-hand side of the a priori width bound for M=256 (beta=0, so
        # L_kappa=epsilon_kappa=0); the covering radius of P equals the box radius.
        rho = PI/(12*256)
        slopes = {lower: [slope_on(lower, lo.lo-b.eps_c.hi, (hi+b.eps_c+2*LQ*rho).hi)
                          for lo, hi in zip(start, end)] for lower in [True, False]}
        rhs = (gamma+sum(slopes[True]+slopes[False])*b.eps_c+4*E_LAMBDA+2*4*b.sigma
               +2*sum(slopes[False])*LQ*rho+sum(slopes[True])*LQ*rho)
        constant, changes = b.constant_until(ref)
        saturated, unsaturated_below = b.saturated_from()
        output['budgets'][name] = export(dict(
            reference_minus=ref['f_minus'], reference_plus=ref['f_plus'],
            endpoint_minus=endpoint['f_minus'], cap=b.cap, gain=gain,
            one_box_UB=ub, exclusion_slack=exclusion_slack,
            improvement_gt_0_59=True, exclusion_margin_0_3=True,
            sigma=b.sigma, L_f=b.Lf, epsilon_c=b.eps_c,
            reference=ref, endpoint=endpoint, exclusion_threshold=threshold,
            one_box=dict(centre=centre, argument=one['argument'], weights=one['weights'],
                         uK=one['uK'], uL=one['uL']),
            lipschitz_only_intervals=b.lipschitz_intervals('.05', threshold),
            refinements=refinements,
            width_formula=gamma+sum((I(1)+w for w in endpoint['w_minus']), I(0))*b.sigma+2*E_LAMBDA,
            apriori=dict(M=256, rho_bar=rho, rho_P=rho, Lambda_plus=slopes[False],
                         Lambda_minus=slopes[True], rhs=rhs),
            f_minus_constant_on_0_to=constant, f_minus_changes_at_next_milliradian=changes,
            lower_weights_saturated_from=saturated, unsaturated_one_centiradian_below=unsaturated_below))
    output['elapsed_s'] = time.perf_counter()-started
    return output


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(Path(__file__).resolve().parent/'results/reviewer/verified_sphere.json'))
    args = p.parse_args()
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    result = verify(); dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {v: b[v]['lower_decimal'] for v in ['gain','exclusion_slack']} for k,b in result['budgets'].items()}, indent=2))