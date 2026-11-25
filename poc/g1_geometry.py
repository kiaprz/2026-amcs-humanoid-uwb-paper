"""Signed distances to G1 collision solids, driven by existing MuJoCo FK.

Mesh solids use the convex hull (MuJoCo collision semantics), cylinders use
their analytic SDF. Each geom is a body segment in the article's L set.
"""
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import mujoco

ROOT = Path(__file__).resolve().parents[1]
# Geometry adapter and pinned Menagerie model bundled with this repository;
# G1_XML may point to another copy of the same model.
os.environ.setdefault('G1_XML', str(ROOT/'g1_model/g1.xml'))
sys.path.insert(0, str(ROOT/'poc/menagerie_shim'))
from humanoid_uwb_sim import G1Model, TagMount


def cylinder_sdf(x, radius, half_length):
    d = np.column_stack((np.linalg.norm(x[:, :2], axis=1)-radius,
                         np.abs(x[:, 2])-half_length))
    return np.linalg.norm(np.maximum(d, 0), axis=1)+np.minimum(np.max(d, axis=1), 0)


def hull_sdf(points, hull):
    """Exact real-arithmetic SDF to a convex polytope; vectorized triangles.

    Inside: negative nearest supporting-plane distance. Outside: closest
    triangle, considering plane projections and all three triangle edges.
    """
    planes = points@hull.equations[:, :3].T+hull.equations[:, 3]
    signed = planes.max(axis=1)
    outside = np.flatnonzero(signed > 0)
    tris = hull.points[hull.simplices]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b-a, c-a
    normals = np.cross(ab, ac)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    aa, bb, cc = np.sum(ab*ab, axis=1), np.sum(ab*ac, axis=1), np.sum(ac*ac, axis=1)
    denominator = aa*cc-bb*bb
    for start in range(0, len(outside), 64):
        ids = outside[start:start+64]
        delta = points[ids, None, :]-a
        height = np.einsum('ptk,tk->pt', delta, normals)
        u = np.einsum('ptk,tk->pt', delta, ab)
        v = np.einsum('ptk,tk->pt', delta, ac)
        s = (cc*u-bb*v)/denominator
        t = (aa*v-bb*u)/denominator
        dist2 = np.where((s >= 0) & (t >= 0) & (s+t <= 1), height**2, np.inf)
        for x, y in [(a, b), (b, c), (c, a)]:
            edge = y-x
            dx = points[ids, None, :]-x
            fraction = np.clip(np.einsum('ptk,tk->pt', dx, edge)/np.sum(edge*edge, axis=1), 0, 1)
            residual = dx-fraction[:, :, None]*edge
            dist2 = np.minimum(dist2, np.sum(residual*residual, axis=2))
        signed[ids] = np.sqrt(dist2.min(axis=1))
    return signed


class G1Geometry:
    def __init__(self, config):
        self.robot = G1Model(tag=TagMount('torso_link', np.array(config['tag_offset_m'])))
        self.joints = config['joints']
        if len(self.joints) not in (2, 3) or len(set(self.joints)) != len(self.joints):
            raise ValueError('Choose two or three distinct left arm joints')
        arm_names = {'left_'+part+'_joint' for part in ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']}
        if any(j not in arm_names for j in self.joints):
            raise ValueError('Expected left arm joints')
        m = self.robot.model
        self.robot.set_base(np.array(config['base_position_m']), 0.0)
        self.fixed_qpos = self.robot.data.qpos.copy()
        self.pose(config['q0'])
        self.p = self.robot.tag_position().copy()
        self.anchors = self.p+np.array(config['anchor_offsets_m'])
        if self.anchors.shape != (4, 3):
            raise ValueError('Exactly four anchors required')
        self.geoms = []
        self.geom_radii = {}
        # Exactly those collision geoms with an ancestor controlled joint.
        selected_jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        for gid in range(m.ngeom):
            if m.geom_group[gid] != 3:
                continue
            body = int(m.geom_bodyid[gid])
            ancestry = []
            while body > 0:
                ancestry.append(body)
                body = int(m.body_parentid[body])
            active = [j for j in selected_jids if int(m.jnt_bodyid[j]) in ancestry]
            if not active:
                continue
            typ = int(m.geom_type[gid])
            hull = None
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gid]
                begin, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                hull = ConvexHull(m.mesh_vert[begin:begin+count].astype(float))
            elif typ != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError('Unsupported moving collision geom')
            self.geoms.append((gid, hull, active))
            self.geom_radii[gid] = (float(np.max(np.linalg.norm(hull.points, axis=1)))
                                   if hull is not None else float(np.linalg.norm(m.geom_size[gid, :2])))
        if not self.geoms:
            raise ValueError('No moving geometry')
        # Rigorous structural bound in exact arithmetic, NOT sampled Jacobian.
        # For joint j: |dr_local/dq_j| <= |r_world - joint_origin_j|.
        # Bound joint origins by torso-to-joint sum of body translation lengths
        # plus |joint local offset|. Then Frobenius bounds spectral norm.
        torso = self.robot._bid['torso_link']
        torso_p = self.robot.data.xpos[torso].copy()
        reaches = {}
        for j in selected_jids:
            body = int(m.jnt_bodyid[j])
            reach = float(np.linalg.norm(m.jnt_pos[j]))
            while body != torso:
                if body == 0:
                    raise ValueError('Joint is not a torso descendant')
                reach += float(np.linalg.norm(m.body_pos[body]))
                body = int(m.body_parentid[body])
            reaches[j] = reach
        self.Lq = np.zeros(4)
        for _, _, active in self.geoms:
            for i, anchor in enumerate(self.anchors):
                D = max(np.linalg.norm(self.p-torso_p), np.linalg.norm(anchor-torso_p)) + config['budget']['relative_geometry_m']
                self.Lq[i] = max(self.Lq[i], np.linalg.norm([D+reaches[j] for j in active]))
        self.constants = dict(Lq_m_per_rad=self.Lq.tolist(), joint_reach_m={self.joints[k]: reaches[j] for k,j in enumerate(selected_jids)},
                              moving_geom_ids=[g[0] for g in self.geoms],
                              moving_bodies=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g[0]])) for g in self.geoms])
        for j, l, h in zip(self.joints, config['Q_lower'], config['Q_upper']):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if l > h or l < m.jnt_range[jid, 0] or h > m.jnt_range[jid, 1]:
                raise ValueError('Q exceeds joint limits')

    def pose(self, q):
        if len(q) != len(self.joints):
            raise ValueError('Wrong posture dimension')
        self.robot.data.qpos[:] = self.fixed_qpos
        self.robot.set_joints(dict(zip(self.joints, q)))
        self.robot.forward()

    def clearance(self, q, N, prune=True, witnesses=False):
        self.pose(q)
        m, d = self.robot.model, self.robot.data
        segment = self.p[None, None, :]+np.linspace(0, 1, N+1)[None, :, None]*(self.anchors-self.p)[:, None, :]
        out = np.full(4, np.inf)
        hits = [None]*4
        for gid, hull, _ in self.geoms:
            rotation = d.geom_xmat[gid].reshape(3, 3)
            local = (segment-d.geom_xpos[gid])@rotation
            for i in range(4):
                # B is contained in its enclosing ball, so phi_B(x) >=
                # ||x||-radius. This prunes nodes, NOT the continuous segment.
                # A small slack guards float comparisons; arithmetic is unverified.
                indices = np.flatnonzero(np.linalg.norm(local[i], axis=1)-self.geom_radii[gid] <= out[i]+1e-12) if prune else np.arange(N+1)
                if len(indices) == 0:
                    continue
                points = local[i, indices]
                sdf = hull_sdf(points, hull) if hull is not None else cylinder_sdf(points, *m.geom_size[gid, :2])
                k = int(np.argmin(sdf))
                if sdf[k] < out[i]:
                    out[i] = sdf[k]
                    node = int(indices[k])
                    hits[i] = dict(geom_id=gid, node=node, s=node/N, point_m=segment[i,node].tolist(), signed_distance_m=float(sdf[k]))
        return (out, hits) if witnesses else out

    def triangles(self, q):
        self.pose(q)
        m, d = self.robot.model, self.robot.data
        output = []
        for gid in range(m.ngeom):
            if m.geom_group[gid] != 2 or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[gid]
            start, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[start:start+n]@d.geom_xmat[gid].reshape(3, 3).T+d.geom_xpos[gid]
            start, n = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            faces = m.mesh_face[start:start+n]
            output.append((verts[faces], int(m.geom_bodyid[gid])))
        return output
