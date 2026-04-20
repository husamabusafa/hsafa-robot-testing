"""
02_head_motion.py — Move the head (roll / pitch / yaw).

The head is a 6-motor Stewart platform. The SDK accepts the target pose as a
4x4 homogeneous transform (numpy). We build it from roll/pitch/yaw here.
"""
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini


def head_pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
              xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a 4x4 head pose from RPY (radians) and optional translation."""
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    M[:3, 3] = xyz
    return M


def main() -> None:
    with ReachyMini() as reachy:
        print("Centering head ...")
        reachy.goto_target(head=head_pose(), duration=1.0, body_yaw=None)
        time.sleep(0.2)

        print("Nodding yes ...")
        for _ in range(2):
            reachy.goto_target(head=head_pose(pitch=math.radians(15)),
                               duration=0.35, body_yaw=None)
            reachy.goto_target(head=head_pose(pitch=math.radians(-10)),
                               duration=0.35, body_yaw=None)
        reachy.goto_target(head=head_pose(), duration=0.4, body_yaw=None)

        print("Shaking no ...")
        for _ in range(2):
            reachy.goto_target(head=head_pose(yaw=math.radians(25)),
                               duration=0.30, body_yaw=None)
            reachy.goto_target(head=head_pose(yaw=math.radians(-25)),
                               duration=0.30, body_yaw=None)
        reachy.goto_target(head=head_pose(), duration=0.4, body_yaw=None)

        print("Tilting (roll) ...")
        reachy.goto_target(head=head_pose(roll=math.radians(20)),
                           duration=0.5, body_yaw=None)
        reachy.goto_target(head=head_pose(roll=math.radians(-20)),
                           duration=0.8, body_yaw=None)
        reachy.goto_target(head=head_pose(), duration=0.5, body_yaw=None)

        print("Done.")


if __name__ == "__main__":
    main()
