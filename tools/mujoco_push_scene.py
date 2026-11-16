"""Go2 + standard PiPER box scene: standing PD, or --push for IK + ONNX walking.

The push mode holds IK arm joint targets and drives only the leg actuators.
On macOS launch the viewer through tools/launch_mujoco.sh (mjpython).
"""
import argparse
import csv
import json
from pathlib import Path
import queue
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "third_party/locomotion_policy"
sys.path.insert(0, str(POLICY_ROOT))
import policy_contract as spec

SCENE = spec.MODEL.parent / "scene_push_box.xml"


class Scene:
    def __init__(self, box_distance=1.0, box_mass=1.0):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.leg_q = [self.model.joint(n).qposadr[0] for n in spec.LEGS]
        self.leg_v = [self.model.joint(n).dofadr[0] for n in spec.LEGS]
        self.leg_a = [self.model.actuator(n).id for n in spec.LEGS]
        self.arm_q = [self.model.joint(n).qposadr[0] for n in spec.ARMS]
        self.arm_a = [self.model.actuator(n).id for n in spec.ARMS]
        self.box_q = self.model.joint("box_free").qposadr[0]
        self.box_distance = box_distance
        box = self.model.body("push_box").id
        self.model.body_inertia[box] *= box_mass / self.model.body_mass[box]
        self.model.body_mass[box] = box_mass
        mujoco.mj_setConst(self.model, self.data)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [0, 0, 0.32, 1, 0, 0, 0]
        self.data.qpos[self.leg_q] = spec.HOME_LEGS
        self.data.qpos[self.arm_q] = spec.HOME_ARM
        self.data.qpos[self.box_q:self.box_q + 7] = [self.box_distance + .2, 0, .2, 1, 0, 0, 0]
        self.data.ctrl[self.arm_a] = spec.HOME_ARM
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        torque = 60 * (np.asarray(spec.HOME_LEGS) - self.data.qpos[self.leg_q]) - 5 * self.data.qvel[self.leg_v]
        limits = self.model.actuator_ctrlrange[self.leg_a]
        self.data.ctrl[self.leg_a] = np.clip(torque, limits[:, 0], limits[:, 1])
        self.data.ctrl[self.arm_a] = spec.HOME_ARM
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")


def check_policy():
    """Check the existing actor's interface only; do not apply it to standard PiPER."""
    import onnxruntime as ort
    path = ROOT / "models/policies/arm_moving_clock/exported/policy.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    entry = session.get_inputs()[0]
    if entry.shape != [1, 300]:
        raise ValueError(f"Unexpected policy input: {entry.shape}")
    result = session.run(None, {entry.name: np.zeros((1, 300), dtype=np.float32)})[0]
    if result.shape != (1, 12) or not np.isfinite(result).all():
        raise ValueError("Invalid policy output")
    return {"path": str(path), "input": entry.shape, "output": list(result.shape),
            "finite": True, "scope": "load/inference only; legacy Piper L actor, not used for standing"}


def headless(scene, seconds):
    min_height, max_tilt = float("inf"), 0.0
    initial_box = scene.data.qpos[scene.box_q:scene.box_q + 3].copy()
    unexpected = set()
    feet = {scene.model.body(n + "_foot").id for n in ("FL", "FR", "RL", "RR")}
    box_body = scene.model.body("push_box").id
    for _ in range(round(seconds / scene.model.opt.timestep)):
        scene.step()
        d, m = scene.data, scene.model
        min_height = min(min_height, float(d.qpos[2]))
        # Base up-axis tilt against world vertical, independent of yaw.
        max_tilt = max(max_tilt, float(np.arccos(np.clip(d.xmat[m.body("base").id, 8], -1, 1))))
        for i, contact in enumerate(d.contact):
            a, b = (int(m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if 0 in (a, b) and (b if a == 0 else a) in feet | {box_body}:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if np.linalg.norm(force[:3]) > 5:
                unexpected.add((m.body(a).name, m.body(b).name))
    box_drift = float(np.linalg.norm(scene.data.qpos[scene.box_q:scene.box_q + 3] - initial_box))
    warnings = int(sum(w.number for w in scene.data.warning))
    passed = min_height > .20 and max_tilt < .30 and box_drift < .01 and warnings == 0 and not unexpected
    return {"kind": "standard_piper_box_standing_pd", "passed": passed,
            "simulated_seconds": float(scene.data.time), "minimum_base_height_m": min_height,
            "maximum_base_tilt_rad": max_tilt, "box_drift_m": box_drift,
            "warnings": warnings, "unexpected_contact_pairs_over_5N": sorted(unexpected),
            "base_position_m": scene.data.qpos[:3].tolist(),
            "box_position_m": scene.data.qpos[scene.box_q:scene.box_q + 3].tolist()}


def save_push_report(controller, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(controller.report(), indent=2, allow_nan=False) + '\n')
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'state', 'base_x', 'base_y', 'base_z', 'box_x', 'box_y', 'box_z',
                         'tcp_x', 'tcp_y', 'tcp_z', 'command_vx', 'command_vy', 'command_wz', 'arm_error']
                        + [f'arm_target_{i}' for i in range(1, 7)] + [f'arm_actual_{i}' for i in range(1, 7)])
        writer.writerows(controller.rows)


def show(scene, seconds, paused, controller=None, report_path=None):
    from mujoco import viewer
    events = queue.SimpleQueue()
    with viewer.launch_passive(scene.model, scene.data, key_callback=events.put) as window:
        window.cam.lookat[:] = [.5, 0, .3]
        window.cam.distance = 2.7
        window.cam.azimuth = 110
        window.cam.elevation = -22
        window.opt.geomgroup[3] = 0  # Render detailed meshes; collision geometry remains active.
        print("Viewer ready. SPACE: pause/resume; R: reset; ESC: close. Forward is +X.", flush=True)
        start = time.monotonic()
        reported = False
        while window.is_running() and (seconds == 0 or time.monotonic() - start < seconds):
            tick = time.monotonic()
            while not events.empty():
                key = events.get_nowait()
                if key == 32:
                    paused = not paused
                elif key in (ord("r"), ord("R")):
                    with window.lock():
                        scene.reset()
                        if controller:
                            controller.reset()
                            reported = False
                            paused = False
            if not paused:
                if controller:
                    controller.step()
                else:
                    for _ in range(4):
                        scene.step()
            if controller and controller.state in ('DONE', 'FAILED') and not reported:
                save_push_report(controller, report_path)
                print(json.dumps(controller.report(), indent=2), flush=True)
                print('Simulation frozen. R: repeat the task; ESC: close.', flush=True)
                reported = True
            if controller:
                window.cam.lookat[0] = (scene.data.qpos[0] + scene.data.xpos[controller.box_id, 0]) / 2
            window.sync()
            time.sleep(max(0, 4 * scene.model.opt.timestep - (time.monotonic() - tick)))
        if controller and not reported:
            save_push_report(controller, report_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--push", action="store_true", help="IK arm, hold joints, then use the ONNX leg policy to push")
    p.add_argument("--push-distance", type=float, default=.5, help="Required box displacement along world +X, metres")
    p.add_argument("--speed", type=float, default=.12, help="Forward command, m/s (0.03 to 0.2)")
    p.add_argument("--seconds", type=float, default=0, help="0: unlimited viewer / 20-second headless check")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--box-distance", type=float, default=1., help="Initial base origin to box near face along +X, metres (>=0.8)")
    p.add_argument("--box-mass", type=float, default=1., help="Box mass in kg")
    p.add_argument("--check-policy", action="store_true", help="Load and run the legacy ONNX actor once, without controlling the robot")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.report is None:
        args.report = ROOT / 'runtime/reports' / ('box_push.json' if args.push else 'mujoco_push_scene.json')
    if not np.isfinite([args.push_distance, args.speed]).all() or not .05 <= args.push_distance <= 1.5 or not .03 <= args.speed <= .2:
        p.error('push-distance must be 0.05 to 1.5 m and speed 0.03 to 0.2 m/s')
    if not np.isfinite([args.seconds, args.box_distance, args.box_mass]).all() or args.seconds < 0 or args.box_distance < .8 or args.box_mass <= 0:
        p.error("seconds >= 0, box-distance >= 0.8 and box-mass > 0 must be finite")
    if args.headless and args.seconds and args.seconds < 2:
        p.error("Use at least 2 seconds for the headless settling check")
    policy = check_policy() if args.check_policy else None
    if policy:
        print(json.dumps(policy, indent=2), flush=True)
    scene = Scene(args.box_distance, args.box_mass)
    print(f"Loaded {SCENE}\nStandard PiPER / {'IK + ONNX push' if args.push else 'standing PD'} / 0.4 m box / {args.box_mass} kg", flush=True)
    if args.push:
        from push_controller import PushController
        controller = PushController(scene, args.push_distance, args.speed)
        if args.headless:
            deadline = args.seconds or 46.
            while scene.data.time < deadline and controller.state not in ('DONE', 'FAILED'):
                controller.step()
            if controller.state not in ('DONE', 'FAILED'):
                controller._fail('run_duration_exceeded')
            save_push_report(controller, args.report)
            print(json.dumps(controller.report(), indent=2))
            return 0 if controller.report()['passed'] else 1
        show(scene, args.seconds, args.paused, controller, args.report)
        return 1 if controller.failure else 0
    if args.headless:
        report = headless(scene, args.seconds or 20)
        report["policy_check"] = policy
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    show(scene, args.seconds, args.paused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
