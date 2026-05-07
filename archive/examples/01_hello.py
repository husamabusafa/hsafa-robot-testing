"""
01_hello.py — Connect to the local Reachy Mini daemon and print robot state.

Prereq: the `reachy-mini-daemon` must already be running. Start it with:
    reachy-mini-daemon --headless --no-media --localhost-only
The daemon talks to the motor board over USB serial; the SDK talks to the
daemon over HTTP/WebSocket on localhost:8000.
"""
import numpy as np
from reachy_mini import ReachyMini


def main() -> None:
    print("Connecting to Reachy Mini daemon on localhost:8000 ...")
    with ReachyMini() as reachy:
        print("Connected.\n")

        pose = reachy.get_current_head_pose()
        np.set_printoptions(precision=3, suppress=True)
        print("Current head pose (4x4 homogeneous transform):")
        print(pose)

        head_joints, antenna_joints = reachy.get_current_joint_positions()
        print("\nHead motor joint positions (rad):")
        for i, v in enumerate(head_joints, start=1):
            print(f"  stewart_{i} = {v: .3f}")
        print("\nAntenna joint positions (rad):")
        print(f"  left  = {antenna_joints[0]: .3f}")
        print(f"  right = {antenna_joints[1]: .3f}")


if __name__ == "__main__":
    main()
