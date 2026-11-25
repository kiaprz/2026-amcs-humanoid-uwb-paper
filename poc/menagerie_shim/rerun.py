"""Quick replay of the static G1 searches, printed next to the stored values.

Usage (from the repository root):
    python poc/menagerie_shim/rerun.py base|hard|bnb
poc/replay_g1_static.py performs the complete comparison and writes a report.
"""
import sys, json, time
sys.path.insert(0, 'poc')
import numpy as np
from bounds import Bounds, Envelope, search
from g1_geometry import G1Geometry

def run(cfgfile, margin=None, max_splits=None, stored=None):
    config = json.load(open(cfgfile))
    if margin is not None: config['epsilon_g'] = margin
    if max_splits is not None: config['max_splits'] = max_splits
    g = G1Geometry(config)
    b = Bounds(g, config['budget'], Envelope(**config['envelope']))
    rng = np.random.default_rng(config['seed'])
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])
    def candidates():
        return [lo, hi, (lo+hi)/2]+list(rng.uniform(lo, hi, (config['candidate_count'], len(lo))))
    t = time.perf_counter()
    r = search(b, lo, hi, config['q0'], candidates=candidates, margin=config['epsilon_g'],
               tolerance=config['epsilon_o'], max_splits=config['max_splits'], max_seconds=config.get('max_seconds'))
    el = time.perf_counter()-t
    s = json.load(open(stored)) if stored else {}
    keys = ['theorem_outcome', 'LB', 'UB', 'split_count', 'evaluation_count', 'guaranteed_gain_lower']
    print(cfgfile, 'margin', config['epsilon_g'], 'time %.1fs' % el)
    for k in keys:
        print('  %-24s new=%-22s stored=%s' % (k, r.get(k), s.get(k)))
    print('  selected', r['selected']['q'], 'stored', s.get('selected', {}).get('q'))
    print('  f0', r['reference']['f_minus'], r['reference']['f_plus'], 'stored', s.get('reference', {}).get('f_minus'), s.get('reference', {}).get('f_plus'))
    print('  fb', r['selected']['f_minus'], r['selected']['f_plus'], 'stored', s.get('selected', {}).get('f_minus'), s.get('selected', {}).get('f_plus'))
    print('  w- ref', np.round(r['reference']['w_minus'], 4), 'w- sel', np.round(r['selected']['w_minus'], 4), flush=True)

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'base':
        run('poc/config.json', 0.02, None, 'poc/results/result.json')
        run('poc/config.json', 0.4, 8, 'poc/results/inconclusive/result.json')
        run('poc/config.json', 0.8, None, 'poc/results/excluded/result.json')
    elif which == 'hard':
        run('poc/config_hard.json', None, None, 'poc/results/hard/result.json')
    else:
        run('poc/config_bnb.json', None, None, 'poc/results/bnb/result.json')
