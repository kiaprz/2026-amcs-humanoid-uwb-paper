"""Geometry adapter: the G1Model/TagMount calls used by poc/g1_geometry.py.

It replaces the authors' simulation engine humanoid-uwb-sim, which is not
distributed, for the kinematic computations of the article; standing.py adds
the free-base model of the standing simulations. Model: MuJoCo Menagerie
unitree_g1/g1.xml at commit 71f066ad0be9cd271f7ed58c030243ef157af9f4
(SHA-256 checked against poc/model_manifest.json).
"""
import os
from dataclasses import dataclass
import numpy as np
import mujoco

MODEL = os.environ.get('G1_XML')


@dataclass
class TagMount:
    body: str
    offset: np.ndarray


class G1Model:
    def __init__(self, tag, keyframe=os.environ.get('G1_KEYFRAME')):
        self.model = mujoco.MjModel.from_xml_path(MODEL)
        self.data = mujoco.MjData(self.model)
        if keyframe:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        self.tag = tag
        self._bid = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b): b
                     for b in range(self.model.nbody)}
        self.forward()

    def set_base(self, position, yaw):
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.forward()

    def set_joints(self, values):
        for name, v in values.items():
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[j]] = v

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def body_position(self, name):
        return self.data.xpos[self._bid[name]].copy()

    def tag_position(self):
        b = self._bid[self.tag.body]
        return self.data.xpos[b]+self.data.xmat[b].reshape(3, 3)@self.tag.offset
