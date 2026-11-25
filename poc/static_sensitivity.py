"""Held-posture base-motion sensitivity; finite trajectories are not eta bounds.

Compare the supplied 2mm budget with torso/tag offsets relative to the
original fixed-base nominal geometry. Transition samples are excluded.
All geometry checks below use unverified float64 arithmetic.
"""
import json
from pathlib import Path
import time
import numpy as np
from bounds import Bounds, Envelope, search
from diffraction_model import DiffractionEnvelope, torso_intersections
from g1_geometry import G1Geometry, ROOT


def width_parts(b, q):
    a = b.at(q)
    c = np.array(a['c_hat'])
    nominal = b.eigen(b.env.plus(c))-b.eigen(b.env.minus(c))
    no_motion = (b.eigen(b.env.plus(c+b.b['e_phi']))
                 -b.eigen(b.env.minus(c-b.b['e_phi']-b.d/(2*b.b['N']))))
    bracket = b.eigen(np.array(a['w_plus']))-b.eigen(np.array(a['w_minus']))
    direction = (np.array(a['w_plus'])+a['w_minus'])@b.sigma
    parts = dict(calibration_at_grid_clearance=nominal,
                 clearance_discretization_and_rounding=no_motion-nominal,
                 relative_geometry=bracket-no_motion,
                 direction=float(direction), eigenvalue_budget=2*b.b['e_lambda'])
    assert abs(sum(parts.values())-(a['f_plus']-a['f_minus'])) < 1e-12
    return dict(index_interval=[a['f_minus'],a['f_plus']],width=a['f_plus']-a['f_minus'],
                decomposition=parts, interpretation='exact algebraic decomposition of computed interval; calibration term uses grid clearance, not Gamma_infinity')


def run(out=ROOT/'poc/results/reviewer/static_sensitivity', dynamics_file=None):
    """Write the nine runs to `out`; `dynamics_file` selects the standing result
    (default: the stored dynamics_static/result.json beside `out`)."""
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    dynamics_file=Path(dynamics_file) if dynamics_file else out.parent/'dynamics_static/result.json'
    dynamics=json.loads(dynamics_file.read_text())
    phases={p:{k:max(t['phases'][p]['maxima'][k] for t in dynamics['trials'])
              for k in dynamics['trials'][0]['phases'][p]['maxima']}
            for p in dynamics['trials'][0]['phases']}
    budget_motion=dict(relative_geometry_m=dynamics['static_nominal_frame_budget_m'],
                       range_vector_m=dynamics['static_nominal_tag_budget_m'])
    report=dict(phases=phases,held_posture_windows_s=dynamics['static_windows_s'],
                substituted_base_motion=budget_motion,runs={},width_terms={},
                scope='Nominal base-motion sensitivity using held-target samples only. Joint target errors are reported but not incorporated. No bound on unobserved postures, trajectories, disturbances or hardware.')
    incremental_motion=dict(relative_geometry_m=max(phases[p]['settled_reference_frame_displacement_bound_m'] for p in ['reference_hold','candidate_hold']),
                            range_vector_m=max(phases[p]['settled_reference_tag_displacement_m'] for p in ['reference_hold','candidate_hold']))
    report['held_incremental_motion']=incremental_motion
    cases=[('first_gain','config.json',.02),('first_middle','config.json',.4),
           ('first_exclusion','config.json',.8),('restricted_gain','config_hard.json',.02),
           ('restricted_exclusion','config_bnb.json',.36),('diffraction_gain','config_hard.json',.02),
           ('incremental_restricted_gain','config_hard.json',.02),
           ('incremental_restricted_exclusion','config_bnb.json',.36),
           ('incremental_diffraction_gain','config_hard.json',.02)]
    for name,file,margin in cases:
        cfg=json.loads((ROOT/'poc'/file).read_text())
        budget=dict(cfg['budget'],**(incremental_motion if name.startswith('incremental_') else budget_motion))
        g=G1Geometry(dict(cfg,budget=budget))
        env=(DiffractionEnvelope(np.linalg.norm(g.anchors-g.p,axis=1),torso_intersections(g))
             if 'diffraction' in name else Envelope(**cfg['envelope']))
        b=Bounds(g,budget,env)
        candidates=[cfg['Q_lower'],cfg['Q_upper'],np.mean([cfg['Q_lower'],cfg['Q_upper']],axis=0)]
        if file=='config.json':
            old=json.loads((ROOT/'poc/results/result.json').read_text())
            candidates += [a['q'] for a in old['evaluations'] if a['phase']=='candidate']
        r=search(b,cfg['Q_lower'],cfg['Q_upper'],cfg['q0'],candidates=candidates,
                 margin=margin,tolerance=.01,max_splits=32,max_seconds=60)
        r['budget']=budget; r['config_file']=file; r['geometry_constants']=g.constants
        (out/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        report['runs'][name]={k:r[k] for k in ['theorem_outcome','LB','UB','exclusion_threshold',
                                'improvement_threshold','split_count','evaluation_count','stop_reason','search_elapsed_s']}
        report['runs'][name]['reference']=r['reference']
        report['runs'][name]['selected']=r['selected']
        report['runs'][name]['point_widths']={label:width_parts(b,q) for label,q in [('q0',cfg['q0']),('qb',cfg['Q_upper'])]}
        if name in ['first_gain','restricted_gain']:
            original=cfg['budget']; base_b=Bounds(G1Geometry(cfg),original,env)
            zero=dict(original,relative_geometry_m=0,range_vector_m=0,direction_offset=0)
            nominal_b=Bounds(G1Geometry(dict(cfg,budget=zero)),zero,env)
            if name=='first_gain':
                assert np.allclose(nominal_b.at(cfg['Q_upper'])['w_minus'],env.lower_cap)
                gamma_upper=gamma_lower=b.eigen(np.broadcast_to(env.upper_cap,4))-b.eigen(np.broadcast_to(env.lower_cap,4))
                ref=base_b.at(cfg['q0']); sm=b.eigen(np.broadcast_to(env.lower_cap,4)); sp=b.eigen(np.broadcast_to(env.upper_cap,4))
                report['middle_margin_obstruction']=dict(S_minus=sm,S_plus=sp,
                    improvement_threshold=ref['f_plus']+.4,exclusion_threshold=ref['f_minus']+.4,
                    improvement_test_impossible=sm<=ref['f_plus']+.4,
                    exclusion_test_impossible=sp>ref['f_minus']+.4,
                    justification='Both nominal envelope maxima attain their common caps at the endpoint; matrix monotonicity bounds all postures.')
            else:
                gamma_lower=0.
                gamma_upper=nominal_b.cap-max(nominal_b.at(q)['f_minus'] for q in candidates)
            def terms(bound):
                return dict(relative_geometry=float(np.sum(env.upper_slope+env.lower_slope)*bound.b['relative_geometry_m'])
                            if np.ndim(env.upper_slope) else float(4*(env.upper_slope+env.lower_slope)*bound.b['relative_geometry_m']),
                    discretization_and_distance_budget=float(np.sum((env.upper_slope+env.lower_slope)*(2*bound.b['e_phi']+bound.d/(2*bound.b['N'])))),
                    direction=float(2*np.broadcast_to(env.upper_cap,4)@bound.sigma),
                    eigenvalue_budget=4*bound.b['e_lambda'])
            report['width_terms'][name]=dict(Gamma_interval=[gamma_lower,gamma_upper],
                   assumed_budget=terms(base_b),held_nominal_budget=terms(b),
                   interpretation='Partition-independent additive upper-bound terms of Eq. (12); their size is not a lower bound on the actual interval width.')
        print(name,r['theorem_outcome'],r['LB'],r['UB'],flush=True)
    cfg=json.loads((ROOT/'poc/config_hard.json').read_text())
    b=Bounds(G1Geometry(dict(cfg,budget=dict(cfg['budget'],**budget_motion))),dict(cfg['budget'],**budget_motion),Envelope(**cfg['envelope']))
    nominal_lower_cap=b.eigen(np.broadcast_to(b.env.lower_cap,4))
    qb=b.at(cfg['Q_upper'])
    assert np.all(b.env.plus(np.array(qb['c_hat'])-b.b['e_phi']-b.d/(2*b.b['N'])+b.b['relative_geometry_m'])==b.env.upper_cap)
    report['held_nominal_obstruction']=dict(S_minus_upper=nominal_lower_cap+b.b['e_lambda'],
        box_upper_floor_from_saturated_qb=b.cap-2*b.b['e_lambda'],
        improvement_threshold_at_002=report['runs']['restricted_gain']['improvement_threshold'],
        exclusion_threshold_at_036=report['runs']['restricted_exclusion']['exclusion_threshold'],
        reason='Lower weights never exceed their common cap. Even a continuous-clearance lower bracket at qb plus the held geometric budget saturates all upper envelopes; every box covering qb has an upper bound at least the saturated eigenvalue plus direction corrections (allowing two eigenvalue budgets).')
    report['source_dynamics']=Path(dynamics_file).resolve().relative_to(ROOT).as_posix()
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'poc/results/reviewer/static_sensitivity'))
    p.add_argument('--dynamics',default=None,help='standing result.json (default: dynamics_static beside --output)')
    args=p.parse_args()
    started=time.perf_counter();run(Path(args.output).resolve(),args.dynamics and Path(args.dynamics).resolve());print('elapsed',time.perf_counter()-started)
