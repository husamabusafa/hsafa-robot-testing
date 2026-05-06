"""Wake up Reachy Mini - enable torque and move to default pose."""

import time
from reachy_mini import ReachyMini
from hsafa_robot.robot_control import head_pose


def main():
    print("Waking up Reachy...")
    
    with ReachyMini(automatic_body_yaw=False, host="localhost") as reachy:
        print("✅ Connected. Moving to default pose...")
        
        # Move to neutral pose smoothly
        reachy.goto_target(
            head=head_pose(roll=0.0, pitch=0.0, yaw=0.0),
            duration=1.0,
            body_yaw=0.0,
            antennas=[0.0, 0.0],
        )
        
        time.sleep(1.2)
        print("✅ Reachy is awake and at neutral pose")


if __name__ == "__main__":
    main()
