"""IK, joint hold and legacy ONNX locomotion for the standalone box experiment."""
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


class LegacyPolicy:
    """Original arm_moving_clock observation/action ABI, using simulation state.

    The original arm reference is zero (not the new standard PiPER HOME_ARM).
    Joints here are in actor order: FL/FR/RL/RR hips, thighs, then calves.
    Application to standard PiPER is an empirical transfer experiment.
    """
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(ROOT / 'models/policies/arm_moving_clock/exported/policy.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape != [1, 300] or self.session.get_outputs()[0].shape != [1, 12]:
            raise ValueError('Expected the legacy 300-input, 12-output locomotion policy')
        self.input = self.session.get_inputs()[0].name
        self.reset()

    def reset(self):
        self.history = np.zeros((5, 60), dtype=np.float32)
        self.previous = np.zeros(12)
        self.phase = np.array([0., .5, .5, 0.])

    def target(self, scene, command):
        d, m = scene.data, scene.model
        rotation = d.xmat[m.body('base').id].reshape(3, 3)
        self.phase = (self.phase + 1.4 * .02) % 1.
        clocks = self.phase if np.linalg.norm(command) >= .01 else np.full(4, -1.)
        home = np.array([0.] * 4 + [.9] * 4 + [-1.8] * 4)
        frame = np.concatenate((rotation.T @ d.qvel[:3], d.qvel[3:6], rotation.T @ [0, 0, -1],
                                command, [0., 0.], d.qpos[scene.leg_q] - home,
                                d.qvel[scene.leg_v], self.previous, clocks, d.qpos[scene.arm_q]))
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        action = self.session.run(None, {self.input: self.history.reshape(1, 300)})[0][0]
        if not np.isfinite(action).all():
            raise RuntimeError('Policy returned nonfinite actions')
        action = np.clip(action, -6, 6)
        target = home + .5 * (.8 * action + .2 * self.previous)
        self.previous = action.copy()
        return target


def solve_ik(scene, target_base, tolerance=.002):
    """Damped least-squares position + forward tool-axis IK, on a scratch state."""
    m = scene.model
    d = mujoco.MjData(m)
    d.qpos[:] = scene.data.qpos
    mujoco.mj_forward(m, d)
    base = m.body('base').id
    rotation = d.xmat[base].reshape(3, 3).copy()
    target = d.xpos[base] + rotation @ np.asarray(target_base)
    forward = rotation[:, 0]
    site = m.site('tcp').id
    dofs = [m.joint(f'joint{i}').dofadr[0] for i in range(1, 7)]
    limits = np.array([m.joint(f'joint{i}').range for i in range(1, 7)])
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for iteration in range(250):
        mujoco.mj_forward(m, d)
        axis = d.site_xmat[site].reshape(3, 3)[:, 2]
        pos_error = target - d.site_xpos[site]
        axis_error = np.cross(axis, forward)
        if np.linalg.norm(pos_error) < tolerance and np.dot(axis, forward) > .995:
            return d.qpos[scene.arm_q].copy(), float(np.linalg.norm(pos_error))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        jac = np.vstack((jp[:, dofs], .3 * jr[:, dofs]))
        error = np.concatenate((pos_error, .3 * axis_error))
        delta = jac.T @ np.linalg.solve(jac @ jac.T + .002 ** 2 * np.eye(6), error)
        delta *= min(1., .12 / max(np.max(np.abs(delta)), 1e-9))
        d.qpos[scene.arm_q] = np.clip(d.qpos[scene.arm_q] + delta, limits[:, 0] + .01, limits[:, 1] - .01)
    raise ValueError(f'IK unreachable: position error {np.linalg.norm(pos_error):.4f} m, axis dot {np.dot(axis, forward):.4f}')


class PushController:
    """SETTLE -> ARM_IK -> LOCK -> APPROACH -> PUSH -> STOP -> DONE.

    Lock means constant position-servo targets, not a weld or qpos overwrite.
    Only joint actuators are commanded. Box pose is read for task feedback.
    """
    def __init__(self, scene, distance=.5, speed=.12):
        self.scene, self.distance, self.speed = scene, distance, speed
        self.policy = LegacyPolicy()
        self.base_id = scene.model.body('base').id
        self.box_id = scene.model.body('push_box').id
        self.tcp_id = scene.model.site('tcp').id
        self.feet = {scene.model.body(n + '_foot').id for n in ('FL', 'FR', 'RL', 'RR')}
        self.fingers = {scene.model.body(n).id for n in ('link7', 'link8')}
        self.reset()

    def reset(self):
        self.scene.reset()
        self.policy.reset()
        self.home = self.scene.data.qpos[self.scene.arm_q].copy()
        self.goal, self.ik_error = solve_ik(self.scene, [.63, 0, .04])
        # Quintic blend: max derivative 1.875, max second derivative 5.774.
        delta = np.max(np.abs(self.goal - self.home))
        self.arm_duration = max(9., 1.875 * delta / np.deg2rad(25),
                                np.sqrt(5.774 * delta / np.deg2rad(50)))
        self._check_arm_path()
        self.arm_target = self.home.copy()
        self.box_start = self.scene.data.xpos[self.box_id].copy()
        self.command = np.zeros(3)
        self.state = 'SETTLE'
        self.transitions = [{'time': 0., 'state': self.state}]
        self.failure = None
        self.contact_seconds = 0.
        self.last_contact = None
        self.max_force = 0.
        self.max_tilt = 0.
        self.max_box_tilt = 0.
        self.min_height = .32
        self.max_arm_error = 0.
        self.arm_error_time = 0.
        self.bad_pairs = set()
        self.contact_pairs = set()
        self.stop_time = None
        self.rows = []

    def _check_arm_path(self):
        m = self.scene.model
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = self.scene.data.qpos
        # This samples nominal geometry; runtime force checks handle actual motion.
        for u in np.linspace(0., 1., 81):
            scratch.qpos[self.scene.arm_q] = self.home + u * (self.goal - self.home)
            mujoco.mj_forward(m, scratch)
            for contact in scratch.contact:
                a, b = (int(m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                if a and b and contact.dist < -.002:
                    raise ValueError(f'Arm path collision: {m.body(a).name}/{m.body(b).name}')

    def _transition(self, state):
        if state != self.state:
            self.state = state
            self.transitions.append({'time': round(self.scene.data.time, 3), 'state': state})
            print(f'[{self.scene.data.time:6.2f}s] {state}', flush=True)

    def _fail(self, reason):
        if self.failure is None:
            self.failure = reason
            self.command[:] = 0
            self._transition('FAILED')

    def _monitor(self):
        s = self.scene
        m, d = s.model, s.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self._fail('nonfinite_state')
            return
        tilt = float(np.arccos(np.clip(d.xmat[self.base_id, 8], -1, 1)))
        box_tilt = float(np.arccos(np.clip(d.xmat[self.box_id, 8], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        self.max_box_tilt = max(self.max_box_tilt, box_tilt)
        self.min_height = min(self.min_height, float(d.qpos[2]))
        error = float(np.max(np.abs(d.qpos[s.arm_q] - self.arm_target)))
        self.max_arm_error = max(self.max_arm_error, error)
        self.arm_error_time = self.arm_error_time + m.opt.timestep if error > .15 else 0.
        touching = False
        total_force = 0.
        for i, contact in enumerate(d.contact):
            a, b = (int(m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            magnitude = float(np.linalg.norm(force[:3]))
            if self.box_id in (a, b) and (b if a == self.box_id else a) in self.fingers:
                total_force += magnitude
                if magnitude > 1.:
                    touching = True
                    self.contact_pairs.add((m.body(a).name, m.body(b).name))
            elif 0 in (a, b) and (b if a == 0 else a) in self.feet | {self.box_id}:
                continue
            elif magnitude > 5.:
                self.bad_pairs.add((m.body(a).name, m.body(b).name))
        self.max_force = max(self.max_force, total_force)
        if touching:
            self.contact_seconds += m.opt.timestep
            self.last_contact = d.time
        if d.qpos[2] < .18 or tilt > .5:
            self._fail('robot_fall')
        elif box_tilt > .4:
            self._fail('box_tipped')
        elif self.bad_pairs:
            self._fail('unexpected_contact')
        elif self.arm_error_time > .5:
            self._fail('arm_tracking_error')
        elif total_force > 100:
            self._fail('excessive_push_force')
        elif any(w.number for w in d.warning):
            self._fail('mujoco_warning')

    def step(self):
        """One 50 Hz control tick; four 200 Hz physics steps."""
        if self.state in ('DONE', 'FAILED'):
            return
        s = self.scene
        m, d = s.model, s.data
        t = d.time
        end_arm = 2. + self.arm_duration
        displacement = d.xpos[self.box_id] - self.box_start
        if t < 2.:
            state = 'SETTLE'
        elif t < end_arm:
            state = 'ARM_IK'
            u = np.clip((t - 2.) / self.arm_duration, 0, 1)
            self.arm_target = self.home + (10*u**3 - 15*u**4 + 6*u**5) * (self.goal - self.home)
        elif t < end_arm + 1.:
            state = 'LOCK'
            self.arm_target = self.goal.copy()
        elif self.stop_time is not None:
            state = 'STOP'
            if t - self.stop_time >= 2.:
                if displacement[0] >= self.distance - .02 and self.contact_seconds >= .2:
                    self._transition('DONE')
                else:
                    self._fail('insufficient_displacement_or_contact')
                return
        elif displacement[0] >= self.distance:
            self.stop_time = t
            state = 'STOP'
        else:
            if np.max(abs(d.qpos[s.arm_q] - self.goal)) > .10:
                self._fail('arm_not_locked')
                return
            state = 'PUSH' if self.last_contact is not None else 'APPROACH'
        self._transition(state)
        if t > 45.:
            self._fail('task_timeout')
            return
        if abs(displacement[1]) > .15:
            self._fail('box_lateral_drift')
            return
        if state == 'PUSH' and t - self.last_contact > 2.:
            self._fail('contact_lost')
            return
        requested = np.zeros(3)
        if state in ('APPROACH', 'PUSH'):
            rotation = d.xmat[self.base_id].reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            # Keep the fixed arm centred on the box; yaw target remains world +X.
            error_y = d.xpos[self.box_id, 1] - d.site_xpos[self.tcp_id, 1]
            world_v = np.array([self.speed, np.clip(1.5 * error_y, -.12, .12)])
            requested[:2] = [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]] @ world_v
            requested[2] = np.clip(-2 * yaw, -.3, .3)
        self.command += np.clip(requested - self.command, -.01, .01)
        leg_target = self.policy.target(s, self.command)
        limits = m.actuator_ctrlrange[s.leg_a]
        for _ in range(4):
            d.ctrl[s.leg_a] = np.clip(20*(leg_target-d.qpos[s.leg_q])-1.5*d.qvel[s.leg_v], limits[:, 0], limits[:, 1])
            d.ctrl[s.arm_a] = self.arm_target
            mujoco.mj_step(m, d)
            self._monitor()
            if self.failure:
                break
        self.rows.append([float(d.time), self.state, *d.qpos[:3].tolist(),
                          *d.xpos[self.box_id].tolist(), *d.site_xpos[self.tcp_id].tolist(),
                          *self.command.tolist(), float(np.max(abs(d.qpos[s.arm_q]-self.arm_target))),
                          *self.arm_target.tolist(), *d.qpos[s.arm_q].tolist()])

    def report(self):
        s = self.scene
        return dict(kind='ik_joint_hold_onnx_box_push', passed=self.state == 'DONE' and not self.failure,
                    state=self.state, failure=self.failure, simulated_seconds=float(s.data.time),
                    robot='Go2 + standard PiPER', policy='arm_moving_clock (legacy Piper L transfer)',
                    target_distance_m=self.distance, command_speed_m_s=self.speed,
                    box_mass_kg=float(s.model.body_mass[self.box_id]),
                    box_displacement_m=(s.data.xpos[self.box_id]-self.box_start).tolist(),
                    contact_seconds=self.contact_seconds, contact_pairs=sorted(self.contact_pairs),
                    maximum_push_force_n=self.max_force, minimum_base_height_m=self.min_height,
                    maximum_base_tilt_rad=self.max_tilt, maximum_box_tilt_rad=self.max_box_tilt,
                    maximum_arm_tracking_error_rad=self.max_arm_error,
                    ik_target_base_m=[.63, 0, .04], ik_position_error_m=self.ik_error,
                    locked_arm_target_rad=self.goal.tolist(),
                    unexpected_contact_pairs_over_5N=sorted(self.bad_pairs),
                    mujoco_warnings=int(sum(w.number for w in s.data.warning)), transitions=self.transitions)
