"""Estimator checks: linear WLS and nonlinear range GN with omitted bias.

The information reference assumes known/subtracted fixed posturewise bias.
GN deliberately omits that bias in mismatch cases. This is not an EFIM for
unrestricted unknown NLOS offsets and does not validate hardware envelopes.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from g1_geometry import G1Geometry, ROOT
from bounds import Bounds, Envelope


def gauss_newton(y,anchors,w,initial,max_iterations=100):
    x=np.broadcast_to(initial,(len(y),3)).copy()
    last=np.full(len(y),np.inf)
    for k in range(max_iterations):
        v=x[:,None,:]-anchors
        ranges=np.linalg.norm(v,axis=2)
        H=v/ranges[:,:,None]
        normal=np.einsum('nki,k,nkj->nij',H,w,H)
        rhs=np.einsum('nki,k,nk->ni',H,w,y-ranges)
        delta=np.linalg.solve(normal,rhs[:,:,None])[:,:,0]
        cost=np.sum(w*(y-ranges)**2,axis=1)
        step=np.ones(len(y))
        for _ in range(16):
            proposal=x+step[:,None]*delta
            new_ranges=np.linalg.norm(proposal[:,None,:]-anchors,axis=2)
            higher=np.sum(w*(y-new_ranges)**2,axis=1)>cost+1e-15
            if not np.any(higher):break
            step[higher]*=.5
        x+=step[:,None]*delta;last=np.linalg.norm(step[:,None]*delta,axis=1)
        if np.all(last<1e-10):break
    return x,dict(iterations=k+1,nonconverged=int(np.sum(last>=1e-8)),maximum_last_step_m=float(last.max()),line_search='objective-decreasing backtracking, up to 16 halvings')


def mse_summary(errors):
    squared=np.sum(errors**2,axis=1)
    mean=float(squared.mean());se=float(squared.std(ddof=1)/np.sqrt(len(squared)))
    return dict(MSE_m2=mean,Monte_Carlo_standard_error_m2=se,
                approximate_95_MC_interval_m2=[mean-1.96*se,mean+1.96*se])


# Experiment constants exactly as printed in the article (Section 6.2,
# paragraph "Estimator mismatch").
SEED=124
SIGMA_REF_M=.03
START_OFFSET_M=(.02,-.02,.02)
MAX_ITERATIONS=100
BIAS_CLEARANCE_M=.08
BIAS_CAPS_M=dict(correct=[0,0,0,0],mild_mismatch=[.03,0,.08,0],strong_mismatch=[.03,0,.5,0])


def compute(replicates=20000):
    """Return the estimator report without writing anything.

    The linear WLS rows are copied from the stored Monte Carlo report
    (poc/results/reviewer/monte_carlo.json), as in the original run.
    """
    reviewer=ROOT/'poc/results/reviewer'
    cfg=json.loads((ROOT/'poc/config_hard.json').read_text());g=G1Geometry(cfg)
    b=Bounds(g,cfg['budget'],Envelope(**cfg['envelope']))
    sigma_ref=SIGMA_REF_M;lambda_ref=1/sigma_ref**2
    original=json.loads((reviewer/'monte_carlo.json').read_text())
    rng=np.random.default_rng(SEED);standard_noise=rng.normal(size=(replicates,4))
    initial=g.p+np.array(START_OFFSET_M);distances=np.linalg.norm(g.anchors-g.p,axis=1)
    report=dict(reference_range_standard_deviation_m=sigma_ref,lambda_ref_per_observation_m_minus2=lambda_ref,
                replicates=replicates,seed=SEED,linear_WLS=[],nonlinear_GN={},
                initial_offset_from_tag_m=list(START_OFFSET_M),iteration_limit=MAX_ITERATIONS,
                bias_caps_m=BIAS_CAPS_M,bias_clearance_m=BIAS_CLEARANCE_M,
                bias_rule='b_i(q)=cap_i*clip((0.08-c_i(q))/0.08,0,1), fixed while estimating position at each posture',
                oracle_scope='Gaussian information with known variance and known/subtracted fixed bias; GN omits bias. Unknown unrestricted bias would need a different EFIM.',
                scope='Synthetic model-mismatch demonstration; no hardware calibration or estimator-error certificate')
    for label,q,k in [('q0',cfg['q0'],0),('qb',cfg['Q_upper'],1)]:
        a=b.at(q);c=np.array(a['c_hat']);w=(b.env.minus(c)+b.env.plus(c))/2
        information=lambda_ref*np.einsum('i,ijk->jk',w,b.outer)
        covariance=np.linalg.inv(information)
        row=original['estimator_demonstration'][k]
        se=float(np.sqrt(2*np.trace(covariance@covariance)/row['replicates']))
        report['linear_WLS'].append(dict(posture=label,**row,analytic_MC_standard_error_m2=se,
            empirical_minus_CRLB_in_standard_errors=(row['WLS_empirical_MSE_m2']-row['linear_Gaussian_CRLB_trace_m2'])/se))
        noise=standard_noise*sigma_ref/np.sqrt(w)
        for name,caps in BIAS_CAPS_M.items():
            bias=np.array(caps)*np.clip((BIAS_CLEARANCE_M-c)/BIAS_CLEARANCE_M,0,1)
            estimates,status=gauss_newton(distances+bias+noise,g.anchors,w,initial,MAX_ITERATIONS)
            error=estimates-g.p
            report['nonlinear_GN'].setdefault(name,{})[label]=dict(q=q,clearance_grid_m=c.tolist(),weights=w.tolist(),
                 bias_m=bias.tolist(),oracle_f_normalized=float(np.linalg.eigvalsh(information)[0]/lambda_ref),
                 oracle_CRLB_trace_m2=float(np.trace(covariance)),solver=status,**mse_summary(error))
    for rows in report['nonlinear_GN'].values():
        rows['MSE_order']=('qb_better' if rows['qb']['MSE_m2']<rows['q0']['MSE_m2'] else 'q0_better')
        rows['oracle_information_order']='qb_better'
    return report


def run(replicates=20000,output=None):
    """Compute the report and store it (default: the stable reviewer path)."""
    out=Path(output) if output else ROOT/'poc/results/reviewer/estimator'
    out.mkdir(parents=True,exist_ok=True)
    report=compute(replicates)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=None,help='directory for result.json (default poc/results/reviewer/estimator)')
    t=time.perf_counter();r=run(output=parser.parse_args().output)
    for name,rows in r['nonlinear_GN'].items():print(name,rows['q0']['MSE_m2'],rows['qb']['MSE_m2'],rows['MSE_order'])
    print('elapsed',time.perf_counter()-t)
