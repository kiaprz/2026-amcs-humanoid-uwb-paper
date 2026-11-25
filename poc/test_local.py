"""Independent continuous sphere check, pruning agreement and GN recovery."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from bounds import Envelope
from diffraction_model import DiffractionEnvelope,torso_intersections
from g1_geometry import G1Geometry,ROOT
from local_bounds import LocalBounds,enclose_cells
from estimator_experiments import gauss_newton


def test_local_cells_enclose_exact_continuous_sphere_clearance():
    # Exact continuous minimum of segment-to-sphere distance, not a denser grid.
    anchor=np.array([2.,0,0]);center=np.array([.3,.03,.02]);radius=.05
    N=32;s=np.linspace(0,1,N+1)
    sl,sr=np.maximum(0,s-.5/N),np.minimum(1,s+.5/N)
    values=np.linalg.norm(s[:,None]*anchor-center,axis=1)-radius
    for axis in np.eye(3):
        for angle in [-.06,.02,.06]:
            R=Rotation.from_rotvec(axis*angle).as_matrix()
            for t in [np.array([.01,0,.005]),np.array([0,-.02,0])]:
                B=np.linalg.norm(t)+np.linalg.norm(R-np.eye(3),2)*np.linalg.norm(anchor)
                cm,cp=enclose_cells(values,s*B,np.maximum(sl*B,sr*B),2.,N,0)
                v=anchor-t;actual_center=R@center
                fraction=np.clip(actual_center@v/(v@v),0,1)
                truth=np.linalg.norm(actual_center-fraction*v)-radius
                assert cm-1e-14<=truth<=cp+1e-14


def test_pruned_cells_match_full_evaluation_with_tracking():
    cfg=json.loads((ROOT/'poc/config_hard.json').read_text())
    r=json.loads((ROOT/'poc/results/reviewer/local/summary.json').read_text())
    g=G1Geometry(dict(cfg,budget=r['budget']))
    env=DiffractionEnvelope(np.linalg.norm(g.anchors-g.p,axis=1),torso_intersections(g))
    local=LocalBounds(g,r['budget'],env,tracking_rad=.0081)
    full=LocalBounds(g,r['budget'],env,tracking_rad=.0081,prune=False)
    for q in [cfg['q0'],cfg['Q_upper']]:
        a,z=local.at(q),full.at(q)
        assert min(a['pruned_cells'])>0
        for field in ['c_minus','c_plus','f_minus','f_plus']:
            assert np.allclose(a[field],z[field],rtol=0,atol=1e-12)


def test_gauss_newton_recovers_noiseless_positions():
    anchors=np.array([[1.8,1.2,-.4],[1.8,-.3,-.4],[-1.8,1.2,-.4],[0,0,1.8]])
    truth=np.array([[0,0,0],[.04,-.03,.02],[-.02,.06,-.01]])
    y=np.linalg.norm(truth[:,None,:]-anchors,axis=2)
    x,status=gauss_newton(y,anchors,np.array([.2,.7,.4,.8]),np.array([.02,-.02,.02]))
    assert status['nonconverged']==0
    assert np.allclose(x,truth,rtol=0,atol=1e-10)
