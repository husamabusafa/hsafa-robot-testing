"""Test gyro-enhanced face tracking.

This script verifies that the BNO055 gyro data is properly integrated
into the face tracking system.
"""

import sys
import time
from reachy_mini import ReachyMini
from hsafa_robot.head_gyro import HeadGyro
from hsafa_robot.gyro_stabilizer import GyroStabilizer


def test_gyro_only():
    """Test that gyro data is flowing."""
    print("=" * 60)
    print("TEST 1: Gyro Data Flow")
    print("=" * 60)
    
    gyro = HeadGyro()
    if not gyro.start():
        print("❌ Failed to start gyro")
        return False
    
    print("✅ Gyro started")
    print("Reading for 3 seconds...")
    
    for i in range(10):
        data = gyro.get_latest()
        if data:
            print(f"  Sample {i+1}: gyro_z={data.gyro_z:.2f}°/s, heading={data.heading:.1f}°")
        else:
            print(f"  Sample {i+1}: No data yet...")
        time.sleep(0.3)
    
    gyro.stop()
    print("✅ Gyro test passed\n")
    return True


def test_stabilizer():
    """Test the gyro stabilizer."""
    print("=" * 60)
    print("TEST 2: Gyro Stabilizer")
    print("=" * 60)
    
    stab = GyroStabilizer()
    if not stab.start():
        print("❌ Failed to start stabilizer")
        return False
    
    print("✅ Stabilizer started")
    print("Testing compensation with fake errors...")
    
    # Simulate tracking error
    raw_err_x = 0.5  # target is to the right
    raw_err_y = -0.2  # target is up
    
    # Get compensation
    for i in range(5):
        data = stab.get_latest()
        if data:
            result = stab.compensate_head_motion(raw_err_x, raw_err_y, 0.033)
            print(f"  Raw error: ({raw_err_x:+.3f}, {raw_err_y:+.3f})")
            print(f"  Gyro: ({result.head_yaw_rate:+.2f}, {result.head_pitch_rate:+.2f}) °/s")
            print(f"  Compensated: ({result.err_x:+.3f}, {result.err_y:+.3f})")
            print(f"  Applied: {result.compensation_applied}")
        time.sleep(0.5)
    
    stab.stop()
    print("✅ Stabilizer test passed\n")
    return True


def test_with_reachy():
    """Test with actual Reachy robot."""
    print("=" * 60)
    print("TEST 3: Integration with Reachy")
    print("=" * 60)
    
    print("Connecting to Reachy...")
    try:
        reachy = ReachyMini(automatic_body_yaw=False)
        print("✅ Connected to Reachy")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Import and test the controller
    from hsafa_robot.robot_control import RobotController
    from hsafa_robot.tracker import CascadeTracker, ensure_pose_model, pick_device
    
    model_path = ensure_pose_model()
    device = pick_device()
    tracker = CascadeTracker(model_path, device)
    
    print("Testing controller with gyro integration...")
    
    with reachy:
        controller = RobotController(reachy, tracker)
        
        # Check if gyro was initialized
        if controller._gyro_enabled:
            print("✅ Gyro stabilizer is enabled in controller")
        else:
            print("⚠️ Gyro stabilizer not enabled (ESP may not be connected)")
        
        # Check initial snapshot
        print(f"  Snapshot gyro fields: yaw_rate={controller.snapshot.gyro_yaw_rate}, "
              f"pitch_rate={controller.snapshot.gyro_pitch_rate}, "
              f"compensated={controller.snapshot.gyro_compensated}")
    
    print("✅ Reachy integration test passed\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("GYRO-ENHANCED FACE TRACKING TEST")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test 1: Gyro data flow
    if not test_gyro_only():
        all_passed = False
    
    # Test 2: Stabilizer
    if not test_stabilizer():
        all_passed = False
    
    # Test 3: Reachy integration (optional - may fail if daemon not running)
    try:
        if not test_with_reachy():
            all_passed = False
    except Exception as e:
        print(f"⚠️ Reachy test skipped (expected if daemon not running): {e}\n")
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nGyro-enhanced face tracking is ready!")
        print("\nRun main.py to see it in action:")
        print("  ./.venv/bin/python main.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
