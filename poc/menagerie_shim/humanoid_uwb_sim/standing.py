"""Free-base standing model for poc/dynamic_standing.py, plain MuJoCo only.

The archived standing runs used the g1.xml cached by humanoid-uwb-sim. This
module builds the same model from the pinned Menagerie file named by G1_XML
(same SHA-256 as recorded in the stored results): the floating base is left
free, a floor plane is added, and the position actuators of the model
(kp=500) are kept unchanged. No balance controller is added.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

FLOOR = dict(name='experiment_floor', type='plane', size='3 3 .1',
             friction='.6 .005 .0001')


def model_path():
    path = os.environ.get('G1_XML')
    if not path:
        raise RuntimeError('Set G1_XML to the pinned g1_model/g1.xml')
    return Path(path).resolve()


def standing_model(timestep):
    """Return (MjModel with floor and free base, path of the source XML)."""
    path = model_path()
    tree = ET.parse(path); root = tree.getroot()
    root.find('compiler').set('meshdir', str(path.parent/'assets'))
    ET.SubElement(root.find('worldbody'), 'geom', **FLOOR)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.opt.timestep = timestep
    return model, path
