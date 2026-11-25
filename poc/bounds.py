"""AMCS equations cpm, sigma, fpm, uK, LBUB and decision (beta=0).

All outputs are conditional on supplied error bounds. Float64 is NOT
verified arithmetic; this module does not issue a formal certificate.
"""
from dataclasses import dataclass
import heapq
import time
import numpy as np


@dataclass
class Envelope:
    lower_floor: float = 0.0
    lower_cap: float = 0.9
    lower_slope: float = 18.0
    lower_offset: float = 0.01
    upper_floor: float = 0.2
    upper_cap: float = 1.0
    upper_slope: float = 40.0
    upper_offset: float = 0.0

    def __post_init__(self):
        vals = list(vars(self).values())
        if not np.all(np.isfinite(vals)):
            raise ValueError('Nonfinite envelope')
        if not (0 <= self.lower_floor <= self.lower_cap <= self.upper_cap
                and self.lower_floor <= self.upper_floor <= self.upper_cap
                and self.lower_slope > 0 and self.upper_slope > 0):
            raise ValueError('Invalid envelope')
        knots = [self.lower_offset + (v-self.lower_floor)/self.lower_slope
                 for v in [self.lower_floor, self.lower_cap]]
        knots += [self.upper_offset + (v-self.upper_cap)/self.upper_slope
                  for v in [self.upper_floor, self.upper_cap]]
        if np.any(self.minus(np.array(knots)) > self.plus(np.array(knots))):
            raise ValueError('Lower envelope exceeds upper envelope')

    def minus(self, c):
        return np.clip(self.lower_floor + self.lower_slope*(c-self.lower_offset),
                       self.lower_floor, self.lower_cap)

    def plus(self, c):
        return np.clip(self.upper_cap + self.upper_slope*(c-self.upper_offset),
                       self.upper_floor, self.upper_cap)


class Bounds:
    def __init__(self, geometry, budget, envelope=None):
        self.geometry, self.b = geometry, budget
        self.env = envelope or Envelope()
        self.d = np.linalg.norm(geometry.anchors-geometry.p, axis=1)
        self.h = (geometry.p-geometry.anchors)/self.d[:, None]
        self.outer = np.einsum('ij,ik->ijk', self.h, self.h)
        for name in ['e_phi', 'e_lambda', 'relative_geometry_m', 'range_vector_m',
                     'direction_offset', 'N']:
            if not np.isfinite(budget[name]) or budget[name] < 0:
                raise ValueError('Invalid budget '+name)
        if budget['N'] < 1 or int(budget['N']) != budget['N']:
            raise ValueError('N must be a positive integer')
        if np.any(budget['range_vector_m'] >= self.d):
            raise ValueError('e_i must be less than d_i')
        self.sigma = budget['range_vector_m']/self.d + budget['direction_offset']*(2+budget['direction_offset'])
        self.Lq = np.asarray(geometry.Lq)
        self.Lf = np.sum(self.env.upper_slope*self.Lq*(1+self.sigma))
        w = np.broadcast_to(self.env.upper_cap, (4,)).copy()
        self.cap = self.eigen(w)+budget['e_lambda']+w@self.sigma

    def eigen(self, w):
        return float(np.linalg.eigvalsh(np.einsum('i,ijk->jk', w, self.outer))[0])

    def at(self, q):
        c = self.geometry.clearance(q, int(self.b['N']))
        cm = c-self.b['e_phi']-self.d/(2*self.b['N'])-self.b['relative_geometry_m']
        cp = c+self.b['e_phi']+self.b['relative_geometry_m']
        wm, wp = self.env.minus(cm), self.env.plus(cp)
        fm = self.eigen(wm)-self.b['e_lambda']-wm@self.sigma
        fp = self.eigen(wp)+self.b['e_lambda']+wp@self.sigma
        return dict(q=np.asarray(q).tolist(), c_hat=c.tolist(), c_minus=cm.tolist(),
                    c_plus=cp.tolist(), w_minus=wm.tolist(), w_plus=wp.tolist(),
                    f_minus=float(fm), f_plus=float(fp))

    def box(self, lo, hi, center_result=None):
        m = (lo+hi)/2
        r = np.linalg.norm((hi-lo)/2)
        a = center_result or self.at(m)
        w = self.env.plus(np.array(a['c_plus'])+self.Lq*r)
        uk = self.eigen(w)+self.b['e_lambda']+w@self.sigma
        ul = min(self.cap, a['f_plus']+self.Lf*r)
        return float(min(uk, ul))


def decision(lb, ub, reference, margin):
    if lb > reference['f_plus']+margin:
        return 'guaranteed improvement'
    if ub <= reference['f_minus']+margin:
        return 'excluded improvement'
    return 'inconclusive'


def search(bounds, lo, hi, q0, candidates=(), margin=0.0, tolerance=0.01, max_splits=256,
           max_seconds=None, progress=None):
    started = time.perf_counter()
    lo, hi, q0 = map(lambda v: np.asarray(v, float), (lo, hi, q0))
    if (lo.shape != hi.shape or q0.shape != lo.shape or np.any(lo > hi)
            or not np.all(np.isfinite([lo, hi, q0])) or margin < 0 or tolerance < 0
            or max_splits < 0 or (max_seconds is not None and max_seconds <= 0)):
        raise ValueError('Invalid search domain/settings')
    def admissible(q):
        return q.shape == lo.shape and np.all(np.isfinite(q)) and np.all((q >= lo) & (q <= hi))
    if not admissible(q0):
        raise ValueError('q0 outside Q')
    cache, evaluations = {}, []
    cache_hits = 0
    def evaluate(q, phase):
        nonlocal cache_hits
        key = tuple(q)
        if key in cache:
            cache_hits += 1
            return cache[key]
        a = bounds.at(q)
        a['phase'] = phase
        cache[key] = a
        evaluations.append(a)
        return a
    ref = evaluate(q0, 'reference')
    best = ref
    reference_s = time.perf_counter()-started
    common = dict(certification_status='conditional PoC; unverified float64 arithmetic', beta=0,
                  epsilon_g=margin, epsilon_o=tolerance, reference=ref,
                  exclusion_threshold=ref['f_minus']+margin,
                  improvement_threshold=ref['f_plus']+margin)
    # Algorithm 1: return BEFORE candidate generation or root-center evaluation.
    if bounds.cap <= ref['f_minus']+margin:
        return dict(common, theorem_outcome='excluded improvement', stop_reason='calibration ceiling',
                    selected=ref, LB=None, UB=bounds.cap, guaranteed_gain_lower=None,
                    history=[dict(splits=0, LB=None, UB=bounds.cap, active_boxes=0,
                                  evaluations=1, elapsed_s=time.perf_counter()-started)],
                    evaluations=evaluations, pruned_boxes=[], remaining_boxes=[],
                    root_upper=None, evaluation_count=1, cache_hits=0, split_count=0,
                    search_elapsed_s=time.perf_counter()-started,
                    phase_seconds=dict(reference=reference_s, candidates=0., branch_and_bound=0.))
    if callable(candidates):
        candidates = candidates()
    candidate_start = time.perf_counter()
    for q in candidates:
        q = np.asarray(q, float)
        if not admissible(q):
            raise ValueError('candidate outside Q')
        a = evaluate(q, 'candidate')
        if a['f_minus'] > best['f_minus']:
            best = a
    heap = []
    serial = 0
    candidate_s = time.perf_counter()-candidate_start
    bnb_start = time.perf_counter()
    def add(l, h, parent_upper=None):
        nonlocal serial, best
        a = evaluate((l+h)/2, 'box center')
        if a['f_minus'] > best['f_minus']:
            best = a
        raw = bounds.box(l, h, a)
        u = raw if parent_upper is None else min(raw, parent_upper)
        record = dict(lo=l.tolist(), hi=h.tolist(), upper=u, raw_upper=raw,
                      parent_upper=parent_upper, center=a)
        heapq.heappush(heap, (-u, serial, l, h, record))
        serial += 1
    add(lo, hi)
    root_upper = -heap[0][0]
    history = []
    pruned = []
    for step in range(max_splits+1):
        lb = best['f_minus']
        pruned.extend(dict(item[4], pruned_at_LB=lb)
                      for item in heap if -item[0] <= lb)
        heap = [item for item in heap if -item[0] > lb]
        heapq.heapify(heap)
        ub = max(lb, -heap[0][0] if heap else lb)
        outcome = decision(lb, ub, ref, margin)
        history.append(dict(splits=step, LB=lb, UB=ub, active_boxes=len(heap),
                            evaluations=len(evaluations), elapsed_s=time.perf_counter()-started))
        if progress is not None and step % 64 == 0:
            progress(history[-1])
        if outcome != 'inconclusive':
            stop = 'decision inequality'
        elif ub-lb <= tolerance:
            stop = 'optimality tolerance'
        elif step == max_splits:
            stop = 'split limit'
        elif max_seconds is not None and time.perf_counter()-started >= max_seconds:
            stop = 'time limit'
        elif not heap:
            stop = 'empty active cover'
        else:
            stop = None
        if stop is not None:
            break
        neg_parent, _, l, h, _ = heapq.heappop(heap)
        dim = int(np.argmax(h-l))
        mid = (l[dim]+h[dim])/2
        h1, l2 = h.copy(), l.copy()
        h1[dim], l2[dim] = mid, mid
        add(l.copy(), h1, -neg_parent)
        add(l2, h.copy(), -neg_parent)
    return dict(common, theorem_outcome=outcome, stop_reason=stop,
                selected=best, LB=lb, UB=ub, guaranteed_gain_lower=lb-ref['f_plus'],
                history=history, evaluations=evaluations,
                pruned_boxes=pruned,
                remaining_boxes=[x[4] for x in heap], root_upper=root_upper,
                evaluation_count=len(evaluations), cache_hits=cache_hits, split_count=step,
                search_elapsed_s=time.perf_counter()-started,
                phase_seconds=dict(reference=reference_s, candidates=candidate_s,
                                   branch_and_bound=time.perf_counter()-bnb_start))
