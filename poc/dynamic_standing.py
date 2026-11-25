"""Free-base, contact-dynamics sensitivity study, NOT a bound on all motions.

No base pose reset or weld is used. The existing position actuators hold
the joints. We add a floor and test a slow arm transition after settling.
Outputs describe finite trajectories, including any failure to stand.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from g1_geometry import ROOT, G1Geometry
from humanoid_uwb_sim.standing import standing_model


def rigid_frame_motion_bound(t, R, t0, R0, anchors):
    """Triangle bound for torso-local fixed-anchor coordinates.

    A torso-mounted tag has constant torso-local coordinates. Therefore
    interpolation from that tag to each fixed anchor has at most this
    displacement, even though the tag moves in world coordinates.
    """
    radius = max(np.linalg.norm(a-t0) for a in anchors)
    return float(np.linalg.norm(t-t0)+np.linalg.norm(R-R0, 2)*radius)


def run(seed=20261011, trials=5, timestep=.002):
    # Plain MuJoCo with the pinned Menagerie model (no humanoid-uwb-sim):
    # free base, added floor, the model's own position actuators.
    model, path = standing_model(timestep)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ['left_shoulder_pitch_joint','left_elbow_joint']]
    aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ['left_shoulder_pitch_joint','left_elbow_joint']]
    config = json.loads((ROOT/'poc/config_hard.json').read_text())
    arm0, arm1 = np.array(config['q0']), np.array(config['Q_upper'])
    offset = np.array(config['tag_offset_m'])
    nominal = G1Geometry(config)
    nominal_origin = nominal.robot.data.xpos[nominal.robot._bid['torso_link']].copy()
    nominal_R = nominal.robot.data.xmat[nominal.robot._bid['torso_link']].reshape(3, 3).copy()
    nominal_tag = nominal.p.copy()
    rng = np.random.default_rng(seed)
    rows, reports = [], []
    for trial in range(trials):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        # Match kinematic PoC joint targets outside the two moving coordinates.
        data.qpos[7:] = 0
        data.ctrl[:] = 0
        data.qpos[[model.jnt_qposadr[j] for j in jids]] = arm0
        data.ctrl[aids] = arm0
        data.qvel[:6] = 0 if trial == 0 else rng.uniform(-.002, .002, 6)
        mujoco.mj_forward(model, data)
        # Five-second standing settlement is reported separately.
        for _ in range(round(5/timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        origin = data.xpos[torso].copy()
        R0 = data.xmat[torso].reshape(3,3).copy()
        tag0 = origin+R0@offset
        anchors = tag0+np.array(config['anchor_offsets_m'])
        maxima = np.zeros(3)
        minheight = float(data.qpos[2]); max_tilt = 0.
        settled_height = minheight
        phases = {name: [] for name in ['reference_hold', 'arm_transition', 'final_relaxation', 'candidate_hold']}
        phase_origins = {}
        for step in range(round(6/timestep)):
            t = step*timestep
            s = np.clip((t-1)/2, 0, 1)
            blend = 3*s*s-2*s*s*s
            data.ctrl[aids] = arm0+blend*(arm1-arm0)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            R = data.xmat[torso].reshape(3,3)
            translation = np.linalg.norm(data.xpos[torso]-origin)
            rotation = np.linalg.norm(R-R0, 2)
            tag = np.linalg.norm(data.xpos[torso]+R@offset-tag0)
            # For nominal arm q, inverse torso transform changes any point
            # by at most |dt|+||R-R0||*radius. This covers base motion only.
            relative = rigid_frame_motion_bound(data.xpos[torso], R, origin, R0, anchors)
            tilt = np.arccos(np.clip(R[2,2], -1, 1))
            maxima = np.maximum(maxima, [tag, relative, translation])
            minheight = min(minheight, float(data.qpos[2]))
            max_tilt = max(max_tilt, float(tilt))
            phase = ('reference_hold' if t < 1 else 'arm_transition' if t < 3
                     else 'final_relaxation' if t < 5 else 'candidate_hold')
            tag_world = data.xpos[torso]+R@offset
            if phase not in phase_origins:
                phase_origins[phase] = (data.xpos[torso].copy(), R.copy(), tag_world.copy())
            phase_p, phase_R, phase_tag = phase_origins[phase]
            values = dict(
                settled_reference_tag_displacement_m=float(tag),
                settled_reference_frame_displacement_bound_m=rigid_frame_motion_bound(data.xpos[torso], R, origin, R0, nominal.anchors),
                nominal_tag_offset_m=float(np.linalg.norm(tag_world-nominal_tag)),
                nominal_segment_frame_bound_m=rigid_frame_motion_bound(data.xpos[torso], R, nominal_origin, nominal_R, nominal.anchors),
                within_phase_tag_drift_m=float(np.linalg.norm(tag_world-phase_tag)),
                within_phase_frame_drift_bound_m=rigid_frame_motion_bound(data.xpos[torso], R, phase_p, phase_R, nominal.anchors),
                controlled_joint_target_error_rad=float(np.max(np.abs(
                    data.qpos[[model.jnt_qposadr[j] for j in jids]]-data.ctrl[aids]))),
                controlled_joint_speed_rad_s=float(np.max(np.abs(
                    data.qvel[[model.jnt_dofadr[j] for j in jids]]))))
            phases[phase].append(values)
            if step % 10 == 0:
                rows.append(dict(phase=phase, **values, trial=trial, time_after_settle_s=t, tag_displacement_m=float(tag),
                    torso_translation_m=float(translation), rotation_spectral=float(rotation),
                    segment_frame_displacement_bound_m=float(relative), base_height_m=float(data.qpos[2])))
        phase_reports = {}
        for name, samples in phases.items():
            phase_reports[name] = dict(start=samples[0], end=samples[-1],
                maxima={k:max(v[k] for v in samples) for k in samples[0]}, samples=len(samples))
        reports.append(dict(phases=phase_reports, trial=trial, settled_base_height_m=settled_height,
            max_tag_displacement_m=float(maxima[0]), max_segment_frame_displacement_bound_m=float(maxima[1]),
            max_torso_translation_m=float(maxima[2]), min_base_height_m=minheight,
            max_torso_tilt_rad=max_tilt, standing_criterion_met=minheight>.6 and max_tilt<.25))
    return dict(seed=seed, trials=reports, timestep_s=timestep, settling_s=5, observation_s=6,
        arm_ramp_s=[1,3], joint_targets_other_than_controlled='zero',
        controller='Menagerie position actuators: kp=500, dampratio=1, original force limits; no balance feedback',
        contact='added plane, friction=[0.6,0.005,0.0001]; free base, gravity, no weld',
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        empirical_eta_tag_m=max(r['max_tag_displacement_m'] for r in reports),
        empirical_relative_geometry_m=max(r['max_segment_frame_displacement_bound_m'] for r in reports),
        static_windows_s=dict(reference_hold=[0,1], candidate_hold=[5,6]),
        phase_time_origin='after the initial 5s settling; final target fixed from t=3s, observation window starts at t=5s',
        nominal_base_position_m=config['base_position_m'],
        static_nominal_tag_budget_m=max(r['phases'][phase]['maxima']['nominal_tag_offset_m']
            for r in reports for phase in ['reference_hold','candidate_hold']),
        static_nominal_frame_budget_m=max(r['phases'][phase]['maxima']['nominal_segment_frame_bound_m']
            for r in reports for phase in ['reference_hold','candidate_hold']),
        interpretation='finite-trajectory empirical maxima; not global uncertainty bounds; excludes joint-tracking/model/anchor errors'), rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default='poc/results/reviewer/dynamics'); p.add_argument('--timestep',type=float,default=.002)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); report,rows=run(timestep=args.timestep)
    report['elapsed_s']=time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'trace.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report,indent=2))
