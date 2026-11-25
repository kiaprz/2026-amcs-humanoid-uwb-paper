"""Reproducible reviewer checks: coverage, Monte Carlo and physical sensitivity.

Statistical experiments and finite dynamics are empirical checks. Their
results never upgrade the G1 float64 implementation to a formal certificate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import chi2, beta
from bounds import Bounds, Envelope, search
from diffraction_model import DiffractionEnvelope, torso_intersections
from g1_geometry import G1Geometry, ROOT


def save(out,name,result):
    (out/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def calibration(g,rng,out):
    distances=np.linalg.norm(g.anchors-g.p,axis=1)
    torso=torso_intersections(g)
    env=DiffractionEnvelope(distances,torso)
    knots=np.linspace(-.05,.4,17); n=120; delta=.05; cells=4*len(knots)
    scale=np.sqrt(2*distances/(.299792458/6.5*.3*(distances-.3)))
    truth=np.array([10**(-(env.loss(-c*scale)+torso*75)/10) for c in knots])
    sigma=.03/np.sqrt(truth)
    observations=rng.normal(0,sigma[:,:,None],(len(knots),4,n))
    variance=observations.var(axis=2,ddof=1); nu=n-1
    low=chi2.ppf(delta/(2*cells),nu); high=chi2.ppf(1-delta/(2*cells),nu)
    wlow=.03**2*low/(nu*variance); whigh=.03**2*high/(nu*variance)
    # Repeated sample-variance draws under the Gaussian protocol, not hardware.
    draws=rng.chisquare(nu,size=(2000,len(knots),4))
    covered=((draws>=low)&(draws<=high)).all(axis=(1,2))
    successes=int(covered.sum())
    coverage_ci=[float(beta.ppf(.025,successes,2001-successes)) if successes else 0,
                 float(beta.ppf(.975,successes+1,2000-successes)) if successes<2000 else 1]
    with (out/'calibration_ranges.csv').open('w',newline='') as f:  # newline='': no CRLF doubling on Windows
        writer=csv.writer(f); writer.writerow(['clearance_m','link','replicate','range_residual_m'])
        for j,c in enumerate(knots):
            for i in range(4):
                writer.writerows((c,i+1,k,x) for k,x in enumerate(observations[j,i]))
    save(out,'calibration.json',dict(protocol='independent Gaussian range residuals at fixed known geometry; sample variance centred by sample mean; online mean offset assumed calibrated; unknown variance orthogonal to range score',
        samples_per_cell=n,cells=cells,delta=delta,simultaneous_nominal_coverage=1-delta,
        reference_variance_m2=.03**2,reference_variance_status='known synthetic input; hardware reference would need a joint confidence interval',
        knots_m=knots.tolist(),true_weights=truth.tolist(),weight_lower=wlow.tolist(),weight_upper=whigh.tolist(),
        calibration_all_cells_covered=bool(((wlow<=truth)&(truth<=whigh)).all()),
        repeated_calibrations=2000,simultaneous_coverage_count=successes,
        empirical_simultaneous_coverage=successes/2000,binomial_95_interval=coverage_ci,
        extension='Only the 68 measured cells are covered by this statistical guarantee. Uniform clearance envelopes additionally require a validated within-cell variation bound and coverage of nuisance states; then apply monotone hulls.',
        status='synthetic statistical demonstration; not hardware calibration; chi-square quantiles evaluated in float64'))
    return env,torso


def physical(g,config,rng,out):
    env,torso=calibration(g,rng,out)
    variants=[('diffraction_assumed_2mm',config['budget'])]
    dynamic=json.loads((out/'dynamics/result.json').read_text())
    empirical=dynamic['empirical_relative_geometry_m']
    budget=dict(config['budget'],relative_geometry_m=empirical,
                range_vector_m=dynamic['empirical_eta_tag_m'])
    variants.append(('diffraction_dynamic_sensitivity',budget))
    candidates=[config['Q_upper'],[(a+b)/2 for a,b in zip(config['Q_lower'],config['Q_upper'])]]
    summary=dict(torso_intersection_lengths_m=torso.tolist(),link_caps_lower=env.lower_cap.tolist(),link_caps_upper=env.upper_cap.tolist(),
        frequency_GHz=list(env.frequency_GHz),edge_distance_m=list(env.edge_distance_m),
        torso_attenuation_dB_per_m=list(env.torso_loss_dB_per_m),
        assumptions='single equivalent knife-edge; SNR-proportional ranging information at fixed bandwidth; effective torso path loss 50--100 dB/m; independent Gaussian observations',
        limitations='The edge position, antenna/material losses, multipath, noise and band response are assumed; not a ray-tracing or measured UWB channel. Torso loss is a sensitivity assumption, not validated material data.',runs={})
    for name,budget in variants:
        variant_g=G1Geometry(dict(config,budget=budget))
        b=Bounds(variant_g,budget,env)
        r=search(b,config['Q_lower'],config['Q_upper'],config['q0'],candidates=candidates,
                 margin=.02,tolerance=.01,max_splits=32,max_seconds=30)
        r['budget']=budget
        save(out,name+'.json',r)
        summary['runs'][name]={k:r[k] for k in ['theorem_outcome','LB','UB','evaluation_count','split_count','search_elapsed_s','stop_reason']}
        summary['runs'][name]['reference']=r['reference']
        summary['runs'][name]['selected']=r['selected']
    save(out,'physical_sensitivity.json',summary)


def monte_carlo(g,config,rng,out,postures=80,replicates=8):
    b=Bounds(g,config['budget'],Envelope(**config['envelope']))
    original_p=g.p.copy(); original_a=g.anchors.copy()
    checks=[]; bound_failures=0; clearance_failures=0; extrema=[float('inf'),float('inf')]
    for j in range(postures):
        q=rng.uniform(config['Q_lower'],config['Q_upper'])
        bound=b.at(q)
        for k in range(replicates):
            # Uniform norm-bounded joint vector of tag and four anchor translations.
            eta=rng.normal(size=15); eta*=.002*rng.random()**(1/15)/np.linalg.norm(eta)
            g.p=original_p+eta[:3]; g.anchors=original_a+eta[3:].reshape(4,3)
            c=g.clearance(q,512)
            w=rng.uniform(b.env.minus(c),b.env.plus(c))
            h=(g.p-g.anchors)/np.linalg.norm(g.p-g.anchors,axis=1)[:,None]
            offset=rng.normal(size=(4,3)); offset*=rng.random((4,1))**(1/3)*.001/np.linalg.norm(offset,axis=1)[:,None]
            directions=h+offset
            f=float(np.linalg.eigvalsh(np.einsum('i,ij,ik->jk',w,directions,directions))[0])
            slack=[f-bound['f_minus'],bound['f_plus']-f]
            bound_failures+=int(min(slack)<0)
            clearance_failures+=int(np.any(c<bound['c_minus']) or np.any(c>bound['c_plus']))
            extrema=np.minimum(extrema,slack).tolist()
            checks.append(dict(posture=j,replicate=k,f=f,f_minus=bound['f_minus'],f_plus=bound['f_plus']))
        g.p=original_p.copy(); g.anchors=original_a.copy()
        if (j+1)%10==0:print(f'Monte Carlo: {j+1}/{postures} postures',flush=True)
    total=postures*replicates
    # A separate estimator demonstration in an exactly linearized Gaussian model.
    covariance_reports=[]
    for q in [config['q0'],config['Q_upper']]:
        a=b.at(q); c=np.array(a['c_hat']); w=(b.env.minus(c)+b.env.plus(c))/2
        covariance=np.linalg.inv(np.einsum('i,ijk->jk',w,b.outer))*.03**2
        ranges=rng.normal(size=(20000,4))*.03/np.sqrt(w)
        A=np.linalg.solve(np.einsum('i,ijk->jk',w,b.outer),b.h.T*w)
        errors=ranges@A.T
        empirical=np.cov(errors,rowvar=False,ddof=1)
        covariance_reports.append(dict(q=q,linear_Gaussian_CRLB_trace_m2=float(np.trace(covariance)),
            WLS_empirical_MSE_m2=float(np.mean(np.sum(errors**2,axis=1))),
            covariance_relative_frobenius_error=float(np.linalg.norm(empirical-covariance)/np.linalg.norm(covariance)),
            replicates=20000,scope='known weights, unbiased exactly linear Gaussian observation model; not nonlinear NLOS ranging'))
    with (out/'monte_carlo_samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    save(out,'monte_carlo.json',dict(postures=postures,replicates_per_posture=replicates,total=total,
        index_interval_failures=bound_failures,clearance_bracket_failures=clearance_failures,
        minimum_lower_upper_slack=extrema,
        zero_failure_95_upper_cluster_rate=1-.05**(1/postures) if bound_failures==0 else None,
        sampling='uniform Q; uniform 15D ball of norm 2mm; weights uniform in declared envelopes; direction offsets in 1mm/m balls',
        truth_clearance='512-interval geometric grid, empirical higher-resolution check; not an exact continuous truth oracle',
        failure_rate_scope='80 independent posture clusters, each with 8 dependent-through-q realizations; upper rate refers to any failure per cluster, not 640 iid trials or worst-case Q/hardware',
        estimator_demonstration=covariance_reports))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['physical','mc'],required=True)
    p.add_argument('--output',default='poc/results/reviewer');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'poc/config_hard.json').read_text());g=G1Geometry(config)
    rng=np.random.default_rng(20261011);started=time.perf_counter()
    (physical if args.mode=='physical' else monte_carlo)(g,config,rng,out)
    save(out,args.mode+'_execution.json',dict(seed=20261011,elapsed_s=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__))
