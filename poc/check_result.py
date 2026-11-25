"""Independent arithmetic replay of exported decision; not error verification."""
import argparse
import json
import numpy as np


def check(path):
    r = json.loads(path.read_text())
    config, geometry = r['config'], r['geometry']
    # Decision uses round-trip float64 JSON values, never paper/plot rounding.
    assert r['exclusion_threshold'] == r['reference']['f_minus']+r['epsilon_g']
    assert r['improvement_threshold'] == r['reference']['f_plus']+r['epsilon_g']
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])
    h, sigma = np.array(geometry['h']), np.array(geometry['sigma'])
    by_q = {tuple(a['q']):a for a in r['evaluations']}
    for a in r['evaluations']:
        q = np.array(a['q'])
        assert np.all((lo <= q) & (q <= hi)), 'Candidate not in Q'
        env, budget = config['envelope'], config['budget']
        c = np.array(a['c_hat'])
        np.testing.assert_allclose(a['c_minus'], c-budget['e_phi']-np.array(geometry['d_m'])/(2*budget['N'])-budget['relative_geometry_m'], atol=1e-14)
        np.testing.assert_allclose(a['c_plus'], c+budget['e_phi']+budget['relative_geometry_m'], atol=1e-14)
        np.testing.assert_allclose(a['w_minus'], np.clip(env['lower_floor']+env['lower_slope']*(np.array(a['c_minus'])-env['lower_offset']),env['lower_floor'],env['lower_cap']), atol=1e-14)
        np.testing.assert_allclose(a['w_plus'], np.clip(env['upper_cap']+env['upper_slope']*(np.array(a['c_plus'])-env['upper_offset']),env['upper_floor'],env['upper_cap']), atol=1e-14)
        for sign, direction in [('minus', -1), ('plus', 1)]:
            w = np.array(a['w_'+sign])
            J = sum(wi*np.outer(v, v) for wi, v in zip(w, h))
            f = np.linalg.eigvalsh(J)[0]+direction*(config['budget']['e_lambda']+w@sigma)
            assert abs(f-a['f_'+sign]) < 1e-12
    if r.get('stop_reason') == 'calibration ceiling':
        assert r['LB'] is None and r['evaluation_count'] == 1 and r['split_count'] == 0
        assert r['evaluations'][0]['q'] == config['q0']
        assert geometry['f_cap'] <= r['reference']['f_minus']+r['epsilon_g']
        assert r['UB'] == geometry['f_cap']
        return dict(arithmetic_replay='PASS', theorem_outcome='excluded improvement', leaf_count=0,
                    formal_certificate=False, reason='Ceiling arithmetic only; supplied errors unverified')
    lb = max(a['f_minus'] for a in r['evaluations'])
    assert abs(lb-r['LB']) < 1e-12
    leaves = r['remaining_boxes']+r['pruned_boxes']
    volume = sum(float(np.prod(np.array(x['hi'])-x['lo'])) for x in leaves)
    assert abs(volume-np.prod(hi-lo)) < 1e-10, 'Final leaves do not cover Q in volume'
    def raw_upper(xl, xh):
        a = by_q[tuple((xl+xh)/2)]
        rad = np.linalg.norm((xh-xl)/2)
        env = config['envelope']
        w = np.clip(env['upper_cap']+env['upper_slope']*(np.array(a['c_plus'])+np.array(geometry['Lq_m_per_rad'])*rad-env['upper_offset']),env['upper_floor'],env['upper_cap'])
        UK = np.linalg.eigvalsh(sum(wi*np.outer(v,v) for wi,v in zip(w,h)))[0]+config['budget']['e_lambda']+w@sigma
        Lf = np.sum(env['upper_slope']*np.array(geometry['Lq_m_per_rad'])*(1+sigma))
        return min(UK, geometry['f_cap'], a['f_plus']+Lf*rad)
    for i, x in enumerate(leaves):
        xl, xh = np.array(x['lo']), np.array(x['hi'])
        assert np.all(lo <= xl) and np.all(xh <= hi)
        if 'center' in x:
            # Reconstruct ancestor bounds from evaluated centres, rather than trust
            # the exported parent_upper number in isolation.
            pl, ph, inherited = lo.copy(), hi.copy(), None
            while not (np.array_equal(pl,xl) and np.array_equal(ph,xh)):
                raw_parent = raw_upper(pl,ph)
                inherited = raw_parent if inherited is None else min(inherited,raw_parent)
                dim = int(np.argmax(ph-pl));mid = (pl[dim]+ph[dim])/2
                if xh[dim] <= mid:ph[dim] = mid
                elif xl[dim] >= mid:pl[dim] = mid
                else:raise AssertionError('Leaf is not part of the dyadic split tree')
            assert (inherited is None and x['parent_upper'] is None) or abs(inherited-x['parent_upper']) < 1e-12
            a = x['center']
            center = np.array(a['q'])
            assert np.allclose(center, (xl+xh)/2, atol=1e-14)
            rad = np.linalg.norm((xh-xl)/2)
            env = config['envelope']
            argument = np.array(a['c_plus'])+np.array(geometry['Lq_m_per_rad'])*rad
            w = np.clip(env['upper_cap']+env['upper_slope']*(argument-env['upper_offset']), env['upper_floor'],env['upper_cap'])
            UK = np.linalg.eigvalsh(sum(wi*np.outer(v,v) for wi,v in zip(w,h)))[0]+config['budget']['e_lambda']+w@sigma
            Lf = np.sum(env['upper_slope']*np.array(geometry['Lq_m_per_rad'])*(1+sigma))
            raw = min(UK,geometry['f_cap'],a['f_plus']+Lf*rad)
            assert abs(raw-x['raw_upper']) < 1e-12
            expected_upper = raw if x['parent_upper'] is None else min(raw,x['parent_upper'])
            assert abs(expected_upper-x['upper']) < 1e-12
        for y in leaves[i+1:]:
            assert np.any(np.minimum(xh, y['hi']) <= np.maximum(xl, y['lo'])), 'Interior overlap'
    for x in r['pruned_boxes']:
        assert x['upper'] <= x['pruned_at_LB'] <= r['LB']
    ub = max([lb]+[x['upper'] for x in r['remaining_boxes']])
    assert abs(ub-r['UB']) < 1e-12
    ref, margin = r['reference'], r['epsilon_g']
    expected = ('guaranteed improvement' if lb > ref['f_plus']+margin else
                'excluded improvement' if ub <= ref['f_minus']+margin else 'inconclusive')
    assert expected == r['theorem_outcome']
    return dict(arithmetic_replay='PASS', theorem_outcome=expected, leaf_count=len(leaves),
                formal_certificate=False, reason='Supplied error budgets and SDF values not verified by replay')


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    print(json.dumps(check(p.parse_args().result), indent=2))
