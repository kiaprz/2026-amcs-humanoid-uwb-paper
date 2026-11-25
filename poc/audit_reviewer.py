"""Recompute new float64 results and audit original decisions independently.

Run after reviewer_experiments.py. Only verified_sphere.verify establishes
arithmetic enclosures; other checks test consistency of assumed inputs.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_decision import audit
from verified_sphere import verify
from comparison_models import comparisons
from diffraction_model import DiffractionEnvelope
from g1_geometry import G1Geometry, ROOT
from bounds import Bounds, Envelope, decision
from local_bounds import LocalBounds

# Stored values come from the authors' platform; a recomputation elsewhere
# differs through platform-dependent rounding of the mesh geometry (observed
# differences below 4e-8, see the replay reports).
REPLAY_TOL=1e-7


def run():
    out=ROOT/'poc/results/reviewer';started=time.perf_counter()
    originals=['result.json','excluded/result.json','inconclusive/result.json','hard/result.json','bnb/result.json']
    original={name:audit(ROOT/'poc/results'/name) for name in originals}
    cfg=json.loads((ROOT/'poc/config_hard.json').read_text())
    ps=json.loads((out/'physical_sensitivity.json').read_text())
    physical={}
    for name in ps['runs']:
        r=json.loads((out/(name+'.json')).read_text());budget=r['budget']
        g=G1Geometry(dict(cfg,budget=budget))
        env=DiffractionEnvelope(np.linalg.norm(g.anchors-g.p,axis=1),ps['torso_intersection_lengths_m'])
        b=Bounds(g,budget,env)
        for e in r['evaluations']:
            rebuilt=b.at(e['q'])
            for field in ['f_minus','f_plus']:
                assert abs(rebuilt[field]-e[field])<REPLAY_TOL,(name,field)
        assert decision(r['LB'],r['UB'],r['reference'],r['epsilon_g'])==r['theorem_outcome']
        physical[name]=dict(clearance_and_index_recompute='PASS',evaluations=len(r['evaluations']),
                            outcome=r['theorem_outcome'],formal_certificate=False)
    static = {}
    summary_path = out/'static_sensitivity/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for name in summary['runs']:
            r = json.loads((summary_path.parent/(name+'.json')).read_text())
            cfg = json.loads((ROOT/'poc'/r['config_file']).read_text())
            g = G1Geometry(dict(cfg, budget=r['budget']))
            env = (DiffractionEnvelope(np.linalg.norm(g.anchors-g.p, axis=1), ps['torso_intersection_lengths_m'])
                   if 'diffraction' in name else Envelope(**cfg['envelope']))
            b = Bounds(g, r['budget'], env)
            for e in r['evaluations']:
                rebuilt = b.at(e['q'])
                for field in ['f_minus', 'f_plus']:
                    assert abs(rebuilt[field]-e[field]) < REPLAY_TOL, (name, field)
            for box in r['remaining_boxes']:
                raw = b.box(np.array(box['lo']), np.array(box['hi']), box['center'])
                assert abs(raw-box['raw_upper']) < REPLAY_TOL, name
            if r['LB'] is None:
                assert b.cap <= r['exclusion_threshold'] and r['theorem_outcome']=='excluded improvement'
            else:
                assert decision(r['LB'], r['UB'], r['reference'], r['epsilon_g'])==r['theorem_outcome']
            static[name] = dict(recompute='PASS', evaluations=len(r['evaluations']), outcome=r['theorem_outcome'])
    local = {}
    summary_path = out/'local/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cfg = json.loads((ROOT/'poc/config_hard.json').read_text())
        for name in summary['runs']:
            r = json.loads((summary_path.parent/(name+'.json')).read_text())
            g = G1Geometry(dict(cfg, budget=r['budget']))
            env = (DiffractionEnvelope(np.linalg.norm(g.anchors-g.p, axis=1), ps['torso_intersection_lengths_m'])
                   if r['envelope_name']=='diffraction' else Envelope(**cfg['envelope']))
            b = LocalBounds(g, r['budget'], env, tracking_rad=r['tracking_rad'])
            for e in r['evaluations']:
                rebuilt = b.at(e['q'])
                for field in ['c_minus', 'c_plus', 'f_minus', 'f_plus']:
                    assert np.allclose(rebuilt[field], e[field], atol=REPLAY_TOL, rtol=0), (name, field)
            for box in r['remaining_boxes']:
                raw = b.box(np.array(box['lo']), np.array(box['hi']), box['center'])
                assert abs(raw-box['raw_upper']) < REPLAY_TOL, name
            assert decision(r['LB'], r['UB'], r['reference'], r['epsilon_g'])==r['theorem_outcome']
            local[name] = dict(recompute='PASS', evaluations=len(r['evaluations']), outcome=r['theorem_outcome'])
        stress = json.loads((summary_path.parent/'stress.json').read_text())
        assert len(stress)==summary['stress_check']['cases']==32
        for row in stress:
            assert row['rigid_frame_m'] <= summary['budget']['relative_geometry_m']
            assert np.all(np.array(row['clearance_grid']) <= np.array(row['upper'])+1e-12)
        local['stress_records'] = dict(cases=len(stress), stored_upper_checks='PASS')
    estimator = json.loads((out/'estimator/result.json').read_text())
    assert abs(estimator['lambda_ref_per_observation_m_minus2']-1/.03**2)<1e-11
    for name, rows in estimator['nonlinear_GN'].items():
        assert rows['oracle_information_order']=='qb_better'
        for posture in ['q0','qb']:
            row=rows[posture]
            lo,hi=row['approximate_95_MC_interval_m2']
            assert abs((hi-lo)/2-1.96*row['Monte_Carlo_standard_error_m2'])<1e-12
        assert (rows['qb']['MSE_m2']<rows['q0']['MSE_m2']) == (rows['MSE_order']=='qb_better')
    files=list((ROOT/'poc').glob('*.py'))+list((ROOT/'poc').glob('config*.json'))
    files+=[ROOT/'g1_model/g1.xml']
    data=list(out.glob('*.json'))+list(out.glob('*.csv'))+list(out.glob('dynamics*/*.json'))+list(out.glob('dynamics*/*.csv'))
    data+=list(out.glob('static_sensitivity/*.json'))
    data+=list(out.glob('local/*.json'))+list(out.glob('estimator/*.json'))
    data=[p for p in data if p.name!='verification.json']
    def fingerprint(paths):
        result = {}
        for p in paths:
            raw = p.read_bytes()
            if p.suffix == '.csv':
                raw = raw.replace(b'\r\n', b'\n')
            result[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return result
    return dict(original_decision_audits=original,physical_recompute=physical,static_sensitivity_recompute=static,
        local_recompute=local,estimator_summary_checks='PASS',
        verified_sphere=verify(),comparison_models=comparisons(),
        source_sha256=fingerprint(files),result_sha256=fingerprint(data),
        result_hash_policy="CSV line endings normalized to LF, matching Git checkout; other files hashed byte-for-byte",
        elapsed_s=time.perf_counter()-started,
        limitations='Verified rational arithmetic applies only to the analytic sphere; empirical tests and dynamics do not establish uniform calibration, global eta or arithmetic guarantees for G1.')


if __name__=='__main__':
    report=run();(ROOT/'poc/results/reviewer/verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('original, physical and held-posture re-evaluations PASS; exact interval decisions PASS')
