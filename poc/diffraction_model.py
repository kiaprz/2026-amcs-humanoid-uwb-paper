"""Physical sensitivity model; not measured radio calibration.

ITU-R P.526 single knife-edge loss. Gaussian time/range noise with constant
bandwidth gives information proportional to SNR. Frequency, edge distance,
and torso attenuation are explicit assumed parameter intervals.
"""
import numpy as np


class DiffractionEnvelope:
    def __init__(self, lengths, torso_length, frequency_GHz=(6.,8.), edge_distance_m=(.15,.60), torso_loss_dB_per_m=(50.,100.)):
        self.lengths=np.asarray(lengths)
        self.torso_length=np.asarray(torso_length)
        self.frequency_GHz=frequency_GHz
        self.edge_distance_m=edge_distance_m
        self.torso_loss_dB_per_m=torso_loss_dB_per_m
        # Smooth parameter extrema: a=d1, d2=d-a; inverse Fresnel radius.
        scales=[]
        for d in self.lengths:
            a0,a1=edge_distance_m
            assert 0<a0<=a1<d
            product_min=min(a0*(d-a0),a1*(d-a1))
            middle=np.clip(d/2,a0,a1)
            product_max=middle*(d-middle)
            scales.append([np.sqrt(2*d/(.299792458/frequency_GHz[0]*product_max)),
                           np.sqrt(2*d/(.299792458/frequency_GHz[1]*product_min))])
        self.scales=np.array(scales)
        self.lower_cap=10**(-self.torso_length*torso_loss_dB_per_m[1]/10)
        self.upper_cap=10**(-self.torso_length*torso_loss_dB_per_m[0]/10)
        # |d attenuation/dc| <= ln(10)/10 * (20/ln(10)) * scale = 2*scale.
        self.upper_slope=2*self.scales[:,1]*self.upper_cap

    @staticmethod
    def loss(v):
        # Continuous positive-part extension of the ITU engineering formula.
        # Avoid a tiny discontinuity from its rounded -0.78 cut-off, so the
        # stated global Lipschitz bound really applies to this model.
        return np.maximum(0,6.9+20/np.log(10)*np.arcsinh(v-.1))

    def _bounds(self,c):
        c=np.asarray(c)
        vals=np.stack([10**(-self.loss(-c*s)/10) for s in self.scales.T])
        return vals.min(axis=0)*self.lower_cap, vals.max(axis=0)*self.upper_cap

    def minus(self,c):return self._bounds(c)[0]
    def plus(self,c):return self._bounds(c)[1]


def torso_intersections(geometry):
    """Line/convex collision-polytope intersections, without a sampled ray.

    Union intervals prevents double counting overlapping torso solids.
    Only torso_link collision meshes are classified as static torso here.
    """
    from scipy.spatial import ConvexHull
    import mujoco
    m,d=geometry.robot.model,geometry.robot.data
    body=geometry.robot._bid['torso_link']
    intervals=[[] for _ in geometry.anchors]
    for gid in range(m.ngeom):
        if m.geom_group[gid]!=3 or m.geom_bodyid[gid]!=body:
            continue
        assert m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH
        mid=m.geom_dataid[gid]; adr=m.mesh_vertadr[mid]; n=m.mesh_vertnum[mid]
        hull=ConvexHull(m.mesh_vert[adr:adr+n].astype(float))
        R=d.geom_xmat[gid].reshape(3,3)
        start=(geometry.p-d.geom_xpos[gid])@R
        for i,a in enumerate(geometry.anchors):
            ray=(a-geometry.p)@R; lo,hi=0.,1.
            for equation in hull.equations:
                offset=equation[:3]@start+equation[3]; slope=equation[:3]@ray
                if abs(slope)<1e-15:
                    if offset>0:lo,hi=1.,0.;break
                elif slope>0:hi=min(hi,-offset/slope)
                else:lo=max(lo,-offset/slope)
            if lo<hi:intervals[i].append((lo,hi))
    lengths=[]
    for i,parts in enumerate(intervals):
        merged=[]
        for lo,hi in sorted(parts):
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        lengths.append(sum(hi-lo for lo,hi in merged)*np.linalg.norm(geometry.anchors[i]-geometry.p))
    return np.array(lengths)
