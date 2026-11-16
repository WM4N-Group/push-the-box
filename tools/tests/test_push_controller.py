"""Physical task regression, IK failure, joint hold, and reset checks (no GUI)."""
import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujoco_push_scene import Scene
from push_controller import PushController, solve_ik


class PushTests(unittest.TestCase):
    def test_ik_reachable_and_unreachable_without_mutating_scene(self):
        scene = Scene()
        before = scene.data.qpos.copy()
        target, error = solve_ik(scene, [.63, 0, .04])
        self.assertLess(error, .002)
        self.assertEqual(target.shape, (6,))
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            solve_ik(scene, [3., 0, 0])
        np.testing.assert_array_equal(scene.data.qpos, before)

    def test_full_push_and_reset(self):
        scene = Scene()
        controller = PushController(scene)
        initial = scene.data.qpos.copy()
        locked = None
        while controller.state not in ('DONE', 'FAILED'):
            controller.step()
            if controller.state in ('LOCK', 'APPROACH', 'PUSH', 'STOP'):
                if locked is None:
                    locked = controller.arm_target.copy()
                np.testing.assert_array_equal(controller.arm_target, locked)
            self.assertFalse(scene.data.xfrc_applied.any())
            self.assertFalse(scene.data.qfrc_applied.any())
        report = controller.report()
        self.assertTrue(report['passed'], report)
        self.assertGreater(report['box_displacement_m'][0], .48)
        self.assertLess(abs(report['box_displacement_m'][1]), .05)
        self.assertGreater(report['contact_seconds'], 1)
        self.assertEqual(report['unexpected_contact_pairs_over_5N'], [])
        self.assertEqual({n for pair in report['contact_pairs'] for n in pair}, {'push_box', 'link7', 'link8'})
        self.assertEqual(report['mujoco_warnings'], 0)
        controller.reset()
        np.testing.assert_array_equal(scene.data.qpos, initial)
        self.assertEqual(controller.state, 'SETTLE')
        self.assertFalse(controller.rows)
        self.assertFalse(controller.policy.history.any())
        self.assertIsNone(controller.last_contact)

    def test_fall_latches_failure_and_freezes(self):
        scene = Scene()
        controller = PushController(scene)
        scene.data.qpos[2] = .12
        mujoco.mj_forward(scene.model, scene.data)
        controller.step()
        self.assertEqual(controller.state, 'FAILED')
        self.assertEqual(controller.failure, 'robot_fall')
        snapshot = scene.data.qpos.copy()
        stopped_time = scene.data.time
        controller.step()
        self.assertEqual(scene.data.time, stopped_time)
        np.testing.assert_array_equal(scene.data.qpos, snapshot)
        self.assertFalse(controller.report()['passed'])


if __name__ == '__main__':
    unittest.main()
