"""
03_antennas_and_body.py — Rotate the body and wiggle the antennas.

Antennas are given as `[left_rad, right_rad]`.
Body yaw is a single scalar in radians.
"""
import math
import time

from reachy_mini import ReachyMini


def main() -> None:
    # automatic_body_yaw=False so the body doesn't move when we only move the head.
    with ReachyMini(automatic_body_yaw=False) as reachy:

        print("Body: rotate left / right / center ...")
        reachy.goto_target(body_yaw=math.radians(45), duration=0.8)
        reachy.goto_target(body_yaw=math.radians(-45), duration=1.2)
        reachy.goto_target(body_yaw=0.0, duration=0.8)

        print("Antennas: expressive flap ...")
        flap_up = [math.radians(60), math.radians(-60)]
        flap_down = [math.radians(-60), math.radians(60)]
        rest = [0.0, 0.0]
        for _ in range(3):
            reachy.goto_target(antennas=flap_up, duration=0.2, body_yaw=None)
            reachy.goto_target(antennas=flap_down, duration=0.2, body_yaw=None)
        reachy.goto_target(antennas=rest, duration=0.3, body_yaw=None)

        print("Combined: 'looking over shoulder' with antenna ears-up ...")
        reachy.goto_target(
            body_yaw=math.radians(60),
            antennas=[math.radians(45), math.radians(45)],
            duration=1.0,
        )
        time.sleep(0.3)
        reachy.goto_target(body_yaw=0.0, antennas=rest, duration=1.0)

        print("Done.")


if __name__ == "__main__":
    main()
