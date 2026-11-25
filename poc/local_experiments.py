"""Comparison of uniform and local clearance bounds with tracking boxes."""
import json
import time
import numpy as np
import mujoco
from bounds import Envelope, search, decision
from diffraction_model import DiffractionEnvelope, torso_intersections
from g1_geometry import G1Geometry, ROOT, cylinder_sdf, hull_sdf
from local_bounds import LocalBounds


def actual_clearance(g,q,delta_q,rotation,translation,N=512):
    g.pose(np.asarray(q)+delta_q)
    m,d = g.robot.model,g.robot.data
    torso = g.robot._bid['torso_link']
    t0,R0 = d.xpos[torso].copy(),d.xmat[torso].reshape(3,3).copy()
    # Torso motion is a rigid perturbation of the original nominal pose.
    tag = t0+rotation@(g.p-t0)+translation
    points = tag[None,None,:]+np.linspace(0,1,N+1)[None,:,None]*(g.anchors-tag)[:,None,:]
    result = np.full(4,np.inf)
    for gid,hull,_ in g.geoms:
        t = t0+rotation@(d.geom_xpos[gid]-t0)+translation
        R = rotation@d.geom_xmat[gid].reshape(3,3)
        local = (points-t)@R
        for i in range(4):
            values = hull_sdf(local[i],hull) if hull is not None else cylinder_sdf(local[i],*m.geom_size[gid,:2])
            result[i] = min(result[i],float(values.min()))
    return result,tag,t0,R0


def run(out=ROOT/'poc/results/reviewer/local',dynamics_file=None):
    """Write the local-bound runs to `out`; `dynamics_file` selects the standing
    result (default: the stored dynamics_static/result.json beside `out`)."""
    from scipy.spatial.transform import Rotation
    from pathlib import Path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'poc/config_hard.json').read_text())
    standing=json.loads(Path(dynamics_file or out.parent/'dynamics_static/result.json').read_text())
    budget=dict(cfg['budget'],relative_geometry_m=standing['static_nominal_frame_budget_m'],
                range_vector_m=standing['static_nominal_tag_budget_m'])
    g=G1Geometry(dict(cfg,budget=budget));tau=.0081
    summary=dict(budget=budget,tracking_per_coordinate_rad=tau,
                 target_Q=[cfg['Q_lower'],cfg['Q_upper']],realized_Q=[(np.array(cfg['Q_lower'])-tau).tolist(),(np.array(cfg['Q_upper'])+tau).tolist()],
                 reference=cfg['q0'],candidate=cfg['Q_upper'],runs={},
                 scope='Fixed nominal torso; rigid motion plus two controlled-joint errors. Other joint/model/mount/anchor errors not established. Inputs are finite-trajectory sensitivity magnitudes, not hardware bounds.')
    for j,lo,hi in zip(cfg['joints'],np.array(cfg['Q_lower'])-tau,np.array(cfg['Q_upper'])+tau):
        jid=mujoco.mj_name2id(g.robot.model,mujoco.mjtObj.mjOBJ_JOINT,j)
        assert g.robot.model.jnt_range[jid,0]<=lo<=hi<=g.robot.model.jnt_range[jid,1]
    summary['realized_joint_limits_checked']=True
    envelopes=dict(clip=Envelope(**cfg['envelope']),
                   diffraction=DiffractionEnvelope(np.linalg.norm(g.anchors-g.p,axis=1),torso_intersections(g)))
    for name,env in envelopes.items():
        for tracking in [0.,tau]:
            b=LocalBounds(g,budget,env,tracking_rad=tracking)
            r=search(b,cfg['Q_lower'],cfg['Q_upper'],cfg['q0'],candidates=[cfg['Q_upper'],np.mean([cfg['Q_lower'],cfg['Q_upper']],axis=0)],
                     margin=.02,tolerance=.01,max_splits=32,max_seconds=60)
            r.update(budget=budget,tracking_rad=tracking,config_file='config_hard.json',envelope_name=name)
            key=name+('_tracked' if tracking else '_local')
            (out/(key+'.json')).write_text(json.dumps(r,indent=2)+'\n')
            summary['runs'][key]={k:r[k] for k in ['theorem_outcome','LB','UB','split_count','evaluation_count','search_elapsed_s','stop_reason','guaranteed_gain_lower']}
            summary['runs'][key]['reference']=r['reference'];summary['runs'][key]['selected']=r['selected']
            print(key,r['theorem_outcome'],r['guaranteed_gain_lower'],flush=True)
    # Boundary-directed stress cases: joint-box vertices and extreme rigid motions.
    b=LocalBounds(g,budget,envelopes['diffraction'],tracking_rad=tau)
    cases=[];rng=np.random.default_rng(124)
    radius=max(np.linalg.norm(a-g.robot.data.xpos[g.robot._bid['torso_link']]) for a in g.anchors)
    full=LocalBounds(g,budget,envelopes['diffraction'],tracking_rad=tau,prune=False)
    for k in range(32):
        q=(np.array(cfg['q0']) if k<8 else np.array(cfg['Q_upper']) if k<16
           else rng.uniform(cfg['Q_lower'],cfg['Q_upper']))
        delta=tau*np.array([1 if k&1 else -1,1 if k&2 else -1])
        axis=np.eye(3)[k%3]*(-1 if k&4 else 1)
        angle=(budget['relative_geometry_m']-budget['range_vector_m'])/(radius+np.linalg.norm(cfg['tag_offset_m']))*.98
        R=Rotation.from_rotvec(axis*angle).as_matrix()
        g.pose(q)
        t=g.robot.data.xpos[g.robot._bid['torso_link']].copy()
        shift=budget['range_vector_m']*np.array([1 if k&1 else -1,1 if k&2 else -1,1 if k&4 else -1])/np.sqrt(3)
        translation=g.p-t-R@(g.p-t)+shift  # rotation about the tag plus extreme tag translation
        H=np.linalg.norm(translation)+np.linalg.norm(R-np.eye(3),2)*max(np.linalg.norm(a-t) for a in g.anchors)
        assert H<=budget['relative_geometry_m']+1e-12
        bound=b.at(q)
        # Lemma 2 omission rule against full evaluation at this posture.
        unpruned=full.at(q)
        for field in ['c_minus','c_plus','f_minus','f_plus']:
            assert np.allclose(bound[field],unpruned[field],atol=1e-12,rtol=0),field
        c,tag,_,_=actual_clearance(g,q,delta,R,translation)
        lengths=np.linalg.norm(g.anchors-tag,axis=1)
        # The dense-grid clearance brackets continuous truth from below.
        assert np.all(c-lengths/(2*512)>=np.array(bound['c_minus'])-1e-12)
        assert np.all(c<=np.array(bound['c_plus'])+1e-12)
        assert np.linalg.norm(tag-g.p)<=budget['range_vector_m']+1e-12
        cases.append(dict(q=q.tolist(),joint_error_rad=delta.tolist(),torso_rotation=R.tolist(),torso_translation_m=translation.tolist(),
                          rigid_frame_m=float(H),tag_displacement_m=float(np.linalg.norm(tag-g.p)),omitted_cells=bound['pruned_cells'],
                          omission_max_abs_difference=float(max(np.max(np.abs(np.array(bound[f])-np.array(unpruned[f]))) for f in ['c_minus','c_plus'])),
                          clearance_grid=c.tolist(),lower=bound['c_minus'],upper=bound['c_plus']))
    # Compare analytical cell pruning with full SDF evaluation at the same nodes.
    for q in [cfg['q0'],cfg['Q_upper'],np.mean([cfg['Q_lower'],cfg['Q_upper']],axis=0)]:
        a,z=b.at(q),full.at(q)
        for field in ['c_minus','c_plus','f_minus','f_plus']:
            assert np.allclose(a[field],z[field],atol=1e-12,rtol=0),field
    summary['stress_check']=dict(cases=len(cases),violations=0,sampling='joint-box vertices; extreme tag shifts and axis rotations; endpoints and random target postures',
                                scope='Finite stress check of analytic enclosure implementation; not a certificate of arithmetic or coverage',pruned_full_agreement=True,pruned_full_agreement_cases=len(cases))
    (out/'stress.json').write_text(json.dumps(cases,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'poc/results/reviewer/local'))
    p.add_argument('--dynamics',default=None,help='standing result.json (default: dynamics_static beside --output)')
    args=p.parse_args()
    t=time.perf_counter();run(Path(args.output).resolve(),args.dynamics and Path(args.dynamics).resolve());print('elapsed',time.perf_counter()-t)
