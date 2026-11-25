"""Original search runner: python poc/run.py [--config ...] [--output ...].

Writes result.json, the CSV tables and the diagnostic plots of one search.
Without --output it overwrites the stored results in poc/results; use
poc/replay_g1_static.py to compare a fresh run with the stored files.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'amcs-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy
import mujoco

from bounds import Bounds, Envelope, search
from g1_geometry import G1Geometry, ROOT
from paper_example import paper_validation, SphereExample


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def model_provenance():
    """Check the bundled G1 model against poc/model_manifest.json (SHA-256)."""
    manifest = json.loads((ROOT/'poc/model_manifest.json').read_text())
    root = Path(os.environ['G1_XML']).resolve().parent
    digest = {f['path']: hashlib.sha256((root/f['path']).read_bytes()).hexdigest()
              if (root/f['path']).is_file() else None for f in manifest['files']}
    bad = [f['path'] for f in manifest['files'] if digest[f['path']] != f['sha256']]
    if bad:
        raise RuntimeError('G1 model files differ from the pinned manifest: '+', '.join(bad))
    return manifest, dict(verified=True, repository=manifest['repository'],
        requested_revision=manifest['requested_revision'], manifest_schema=manifest['manifest_schema'],
        g1_xml_sha256=digest['g1.xml'], model_license_sha256=digest.get('MODEL_LICENSE'),
        file_count=len(manifest['files']), all_manifest_files_verified=True)


def host_hardware():
    fields = ['machdep.cpu.brand_string', 'hw.model', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.memsize']
    values = []
    for field in fields:
        try:
            v = subprocess.run(['sysctl', '-n', field], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            v = None
        values.append(v)
    return dict(cpu=values[0] or platform.processor() or None, model=values[1],
        physical_cores=int(values[2]) if values[2] else None,
        logical_cores=int(values[3]) if values[3] else os.cpu_count(),
        memory_GiB=int(values[4])/2**30 if values[4] else None,
        os=platform.platform(), note='Read-only sysctl; unavailable fields remain null in restricted environments.')


def figures(g, b, result, out):
    colors = ['#d97706', '#2563eb', '#16a34a', '#9333ea']
    fig = plt.figure(figsize=(12, 6))
    for panel, (q, title) in enumerate([(result['reference']['q'], 'Reference q0'),
                                        (result['selected']['q'], 'Selected posture')], 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        moving = set(g.robot.model.geom_bodyid[gid] for gid, _, _ in g.geoms)
        for tris, bid in g.triangles(q):
            ax.add_collection3d(Poly3DCollection(tris, linewidths=0,
                facecolor='#e3933a' if bid in moving else '#acb8c7', alpha=1.0))
        for i, a in enumerate(g.anchors):
            ax.plot(*np.array([g.p, a]).T, color=colors[i], linewidth=1.6)
            ax.scatter(*a, color=colors[i], s=45)
            ax.text(*a, str(i+1))
        ax.scatter(*g.p, color='black', s=65, marker='D', depthshade=False)
        ax.text(*(g.p+np.array([0, -.08, .04])), 'TAG', color='black')
        clearance, witnesses = g.clearance(q, int(b.b['N']), witnesses=True)
        hit = witnesses[0]
        point = np.array(hit['point_m'])
        ax.scatter(*point, color='red', s=100, marker='x', depthshade=False)
        ax.text(*(point+np.array([0, .04, .04])), 'min grid clearance, link 1', color='red', fontsize=8)
        ax.plot(*np.array([g.p, g.anchors[0]]).T, color='red', lw=2.5, ls='--')
        ax.set(xlim=(-2.1, 2.1), ylim=(-1.5, 1.5), zlim=(0, 2.9),
               xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=title+'\nq='+str(np.round(q, 3)))
        ax.set_box_aspect((4.2, 3, 2.9))
        ax.view_init(18, 28)
    fig.suptitle('Unitree G1: fixed torso/tag and anchors; moving left arm in orange')
    fig.tight_layout()
    fig.savefig(out/'postures.png', dpi=170)
    for ax in fig.axes:
        ax.set(xlim=(-.4, .5), ylim=(-.5, .5), zlim=(.65, 1.5))
        ax.set_box_aspect((.9, 1, .85))
    fig.suptitle('Unitree G1: torso and left arm close-up (anchor rays extend beyond frame)')
    fig.savefig(out/'postures_closeup.png', dpi=170)
    plt.close(fig)

    # Straight path is contained in the explicitly convex Q box. No execution claim.
    t = np.linspace(0, 1, 25)
    path = np.array(result['reference']['q'])[None, :]*(1-t[:, None])+np.array(result['selected']['q'])[None, :]*t[:, None]
    values = [b.at(q) for q in path]
    with (out/'path.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', *g.joints, 'f_minus', 'f_plus', *['c_hat_'+str(i+1) for i in range(4)]])
        for a, q, v in zip(t, path, values):
            w.writerow([a, *q, v['f_minus'], v['f_plus'], *v['c_hat']])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    fm, fp = [np.array([v[key] for v in values]) for key in ['f_minus', 'f_plus']]
    ax[0].fill_between(t, fm, fp, alpha=.18, color='#2563eb')
    ax[0].plot(t, fm, label='f minus')
    ax[0].plot(t, fp, label='f plus')
    ax[0].axhline(result['improvement_threshold'], color='black', ls=':', label='improvement threshold')
    ax[0].set(xlabel='q0 -> selected (path parameter)', ylabel='normalized information', title='Conditional posture bounds')
    ax[0].legend(fontsize=8)
    for i in range(4):
        ax[1].plot(t, [v['c_hat'][i] for v in values], color=colors[i], label='link '+str(i+1))
        ax[1].fill_between(t, [v['c_minus'][i] for v in values], [v['c_plus'][i] for v in values], color=colors[i], alpha=.12)
    ax[1].axhline(0, color='black', ls=':')
    ax[1].set(xlabel='path parameter', ylabel='clearance [m]', title='Signed clearance and bounds')
    ax[1].legend(fontsize=8)
    hist = result['history']
    ax[2].plot([v['splits'] for v in hist], [v['LB'] for v in hist], 'o-', label='LB')
    ax[2].plot([v['splits'] for v in hist], [v['UB'] for v in hist], 'o-', label='UB')
    ax[2].axhline(result['improvement_threshold'], color='black', ls=':')
    ax[2].set(xlabel='branch-and-bound splits', ylabel='normalized information', title='Global bounds on Q')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out/'bounds.png', dpi=170)
    plt.close(fig)
    if len(g.joints) != 2:
        return  # The search supports 3D; this visualization is a two-joint map.
    n = result['config'].get('map_grid', 35)
    lo, hi = np.array(result['config']['Q_lower']), np.array(result['config']['Q_upper'])
    x, y = [np.linspace(lo[j], hi[j], n) for j in range(2)]
    z = np.array([[b.at([a, c])['f_minus'] for a in x] for c in y])
    np.savez_compressed(out/'map.npz', shoulder_pitch=x, elbow=y, f_minus=z)
    from plot_map import plot_map
    plot_map(x, y, z, result, out)


def paper_plot(out):
    g = SphereExample()
    b = Bounds(g, dict(N=1000, e_phi=1e-4, e_lambda=1e-6, relative_geometry_m=.010,
                       range_vector_m=.012, direction_offset=.002))
    q = np.linspace(0, np.pi/6, 101)
    vals = [b.at([x]) for x in q]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    c = np.linspace(-.07, .12, 200)
    ax[0].plot(c, b.env.minus(c), '--', label='psi minus')
    ax[0].plot(c, b.env.plus(c), label='psi plus')
    ax[0].set(xlabel='clearance [m]', ylabel='weight', title='Article envelopes')
    ax[0].legend()
    for key in ['f_minus', 'f_plus']:
        ax[1].plot(q, [v[key] for v in vals], label=key)
    ax[1].plot(q, .3+b.env.minus(.3*np.sin(q)-.05), '--', label='Phi minus')
    ax[1].plot(q, 1/3+b.env.plus(.3*np.sin(q)-.05), '--', label='Phi plus')
    ax[1].set(xlabel='q [rad]', ylabel='index', title='Article budget A')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'paper_example.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--margin', type=float)
    parser.add_argument('--max-splits', type=int)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.margin is not None:
        config['epsilon_g'] = args.margin
    if args.max_splits is not None:
        config['max_splits'] = args.max_splits
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print('Verifying article budgets A/B...', flush=True)
    write_json(out/'paper_validation.json', paper_validation())
    print('Loading pinned G1 model and moving collision solids...', flush=True)
    model_manifest, model_verified = model_provenance()
    g = G1Geometry(config)
    b = Bounds(g, config['budget'], Envelope(**config['envelope']))
    rng = np.random.default_rng(config['seed'])
    lo, hi = np.array(config['Q_lower']), np.array(config['Q_upper'])
    def candidates():
        return [lo, hi, (lo+hi)/2]+list(rng.uniform(lo, hi, (config['candidate_count'], len(lo))))
    print('Candidate search and branch-and-bound...', flush=True)
    result = search(b, lo, hi, config['q0'], candidates=candidates, margin=config['epsilon_g'],
                    tolerance=config['epsilon_o'], max_splits=config['max_splits'],
                    max_seconds=config.get('max_seconds'),
                    progress=lambda h: print('splits={splits} evals={evaluations} LB={LB:.8f} UB={UB:.8f} t={elapsed_s:.1f}s'.format(**h), flush=True))
    result.update(config=config, geometry=dict(p_m=g.p.tolist(), anchors_m=g.anchors.tolist(),
                  d_m=b.d.tolist(), h=b.h.tolist(), sigma=b.sigma.tolist(), f_cap=b.cap, **g.constants),
                  assumptions=dict(Q='kinematic fixture box only; balance/self-collision not asserted',
                    envelopes='synthetic; not radio calibration',
                    errors='supplied synthetic budgets; e_phi includes FK/SDF but is not verified',
                    Lq='analytic chain-length bound in real arithmetic',
                    geometry='MuJoCo convex collision hulls and cylinders; not anatomical/radio geometry'),
                  provenance=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    mujoco=mujoco.__version__, matplotlib=matplotlib.__version__,
                    model_manifest=model_manifest,
                    model_verified=model_verified,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    result['hardware'] = host_hardware()
    M = b.h.T@b.h
    mu = float(np.linalg.eigvalsh(M)[0])
    result['saturation_width'] = dict(mu_min=mu,
        calibration=(b.env.upper_cap-b.env.lower_cap)*mu,
        corrections=(b.env.upper_cap+b.env.lower_cap)*float(sum(b.sigma))+2*b.b['e_lambda'],
        applies_to_saturated_easy_example=bool(np.allclose(result['selected']['w_minus'], b.env.lower_cap)))
    cover = result['remaining_boxes']+result['pruned_boxes']
    if cover:
        wk = np.array([b.env.plus(np.array(v['center']['c_plus'])+b.geometry.Lq*np.linalg.norm((np.array(v['hi'])-v['lo'])/2)) for v in cover])
        result['cover_weight_upper_max'] = wk.max(axis=0).tolist()
        result['cover_excludes_simultaneous_upper_saturation'] = bool(np.all(np.any(wk < b.env.upper_cap, axis=1)))
    result['witnesses'] = {key:g.clearance(result[key]['q'], int(b.b['N']), witnesses=True)[1] for key in ['reference', 'selected']}
    write_json(out/'result.json', result)
    with (out/'search.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([*g.joints, 'f_minus', 'f_plus', *['c_hat_'+str(i+1) for i in range(4)]])
        for v in result['evaluations']:
            w.writerow([*v['q'], v['f_minus'], v['f_plus'], *v['c_hat']])
    with (out/'bnb.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['splits', 'LB', 'UB', 'active_boxes', 'evaluations', 'elapsed_s'])
        w.writeheader()
        w.writerows(result['history'])
    print(result['theorem_outcome'], 'conditional:', result['LB'], result['UB'], flush=True)
    if not args.no_plots:
        print('Rendering robot postures and bound plots...', flush=True)
        figures(g, b, result, out)
        paper_plot(out)
    result['total_elapsed_s'] = time.perf_counter()-started
    write_json(out/'result.json', result)
    print(f'Done in {result["total_elapsed_s"]:.1f}s: {out}', flush=True)


if __name__ == '__main__':
    main()
