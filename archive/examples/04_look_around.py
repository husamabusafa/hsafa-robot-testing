"""
04_look_around.py — Curious "look around" behaviour using look_at_world.

`look_at_world(x, y, z, duration)` aims the head at a 3D point in the
robot's world frame (meters):
  +X forward, +Y left, +Z up, origin at the base.
Press Ctrl-C to stop; the robot will return to rest and sleep.
"""
import math
import random
import time

from reachy_mini import ReachyMini


def main() -> None:
    with ReachyMini(automatic_body_yaw=True) as reachy:
        # Start looking straight ahead, 0.5 m in front.
        reachy.look_at_world(0.5, 0.0, 0.0, duration=0.8)
        time.sleep(0.2)

        try:
            for i in range(8):
                # Random point ~0.5 m in front, in a cone around the robot.
                x = random.uniform(0.30, 0.60)
                y = random.uniform(-0.35, 0.35)   # left/right
                z = random.uniform(-0.15, 0.20)   # down/up
                dur = random.uniform(0.5, 1.0)

                reachy.look_at_world(x, y, z, duration=dur)

                if i % 3 == 0:
                    reachy.goto_target(
                        antennas=[
                            math.radians(random.uniform(-40, 40)),
                            math.radians(random.uniform(-40, 40)),
                        ],
                        duration=dur,
                        body_yaw=None,
                    )
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted, resting ...")

        print("Resting ...")
        reachy.look_at_world(0.5, 0.0, 0.0, duration=0.8)
        reachy.goto_target(antennas=[0.0, 0.0], duration=0.4, body_yaw=0.0)


if __name__ == "__main__":
    main()
