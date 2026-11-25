"""Cell-wise clearance bounds for rigid torso motion and joint tracking.

No posture-dependent torso equilibrium is introduced. Inputs are assumed
uniform magnitudes, informed by finite standing trajectories. All arithmetic
is ordinary float64; analytic formulas alone do not verify rounding errors.
"""
import mujoco
import numpy as np
from bounds import Bounds
from g1_geometry import cylinder_sdf, hull_sdf


def enclose_cells(values, node_loss, cell_loss, distance, intervals, error):
    """1-Lipschitz SDF on Voronoi cells; uncertainty maxima are cell-wise."""
    lower = float(np.min(values-error-distance/(2*intervals)-cell_loss))
    upper = float(np.min(values+error+node_loss))
    return lower, upper


class LocalBounds(Bounds):
    def __init__(self, geometry, budget, envelope, tracking_rad=(0., 0.), prune=True):
        super().__init__(geometry, budget, envelope)
        self.tau = np.broadcast_to(tracking_rad, (len(geometry.joints),)).copy()
        if not np.all(np.isfinite(self.tau)) or np.any(self.tau < 0):
            raise ValueError('Invalid tracking bound')
        self.prune = prune
        m = geometry.robot.model
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in geometry.joints]
        self.reaches = {j:geometry.constants['joint_reach_m'][name]
                        for j,name in zip(self.jids, geometry.joints)}
        self.origin_errors = {}
        for j in self.jids:
            ancestors = set()
            body = int(m.body_parentid[int(m.jnt_bodyid[j])])
            while body > 0:
                ancestors.add(body); body = int(m.body_parentid[body])
            self.origin_errors[j] = sum(t*(self.reaches[j]+self.reaches[k])
                                       for k,t in zip(self.jids,self.tau)
                                       if int(m.jnt_bodyid[k]) in ancestors)

    def profile(self, s, active):
        g = self.geometry
        ray = g.p[None,None,:]+s[None,:,None]*(g.anchors-g.p)[:,None,:]
        loss = np.broadcast_to(s*self.b['relative_geometry_m'], (4,len(s))).copy()
        for j,tau in zip(self.jids,self.tau):
            if j in active:
                origin = g.robot.data.xanchor[j]
                loss += tau*(np.linalg.norm(ray-origin,axis=2)+self.origin_errors[j])
        return loss

    def clearance_interval(self, q):
        g = self.geometry; g.pose(q)
        m,d = g.robot.model,g.robot.data
        N = int(self.b['N']); s = np.linspace(0,1,N+1)
        sl,sr = np.maximum(0,s-.5/N),np.minimum(1,s+.5/N)
        ray = g.p[None,None,:]+s[None,:,None]*(g.anchors-g.p)[:,None,:]
        lower,upper = np.full(4,np.inf),np.full(4,np.inf)
        retained_s = np.zeros(4); pruned = np.zeros(4,dtype=int)
        # Any evaluated node gives a uniform upper bound on the global minimum.
        for gid,hull,active in g.geoms:
            local = (g.p-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3,3)
            value = (hull_sdf(local[None,:],hull)[0] if hull is not None
                     else cylinder_sdf(local[None,:],*m.geom_size[gid,:2])[0])
            upper = np.minimum(upper,value+self.b['e_phi']+self.profile(np.array([0.]),active)[:,0])
        for gid,hull,active in g.geoms:
            local = (ray-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3,3)
            node_loss = self.profile(s,active)
            cell_loss = np.maximum(self.profile(sl,active),self.profile(sr,active))
            radius = g.geom_radii[gid]
            coarse = np.linalg.norm(local,axis=2)-radius-self.d[:,None]/(2*N)-cell_loss-self.b['e_phi']
            for i in range(4):
                indices = np.flatnonzero(coarse[i] <= upper[i]+1e-12) if self.prune else np.arange(N+1)
                pruned[i] += N+1-len(indices)
                if not len(indices):
                    continue
                points = local[i,indices]
                values = hull_sdf(points,hull) if hull is not None else cylinder_sdf(points,*m.geom_size[gid,:2])
                cm,cp = enclose_cells(values,node_loss[i,indices],cell_loss[i,indices],self.d[i],N,self.b['e_phi'])
                lower[i],upper[i] = min(lower[i],cm),min(upper[i],cp)
                retained_s[i] = max(retained_s[i],float(s[indices].max()))
        return dict(c_minus=lower.tolist(),c_plus=upper.tolist(),
                    retained_parameter_max=retained_s.tolist(),pruned_cells=pruned.tolist(),
                    total_cells_per_link=len(g.geoms)*(N+1))

    def at(self, q):
        interval = self.clearance_interval(q)
        wm,wp = self.env.minus(np.array(interval['c_minus'])),self.env.plus(np.array(interval['c_plus']))
        return dict(q=np.asarray(q).tolist(),**interval,w_minus=wm.tolist(),w_plus=wp.tolist(),
                    f_minus=self.eigen(wm)-self.b['e_lambda']-wm@self.sigma,
                    f_plus=self.eigen(wp)+self.b['e_lambda']+wp@self.sigma)
