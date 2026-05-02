"""Test script to integrate ESP gyro with Reachy Mini.

This shows how to read gyro data while controlling the robot.
"""

import time
import sys
from reachy_mini import ReachyMini
from hsafa_robot.esp_gyro_bridge import ESPGyroBridge


def main():
    print("=" * 60)
    print("ESP GYRO + REACHY MINI INTEGRATION TEST")
    print("=" * 60)
    
    # Start ESP gyro reader
    print("\n1. Starting ESP gyro reader...")
    gyro_bridge = ESPGyroBridge()
    if not gyro_bridge.start():
        print("   ❌ Failed to start ESP reader")
        print("   Make sure ESP is connected to USB")
        return 1
    print("   ✅ ESP reader started")
    
    # Give ESP time to initialize
    time.sleep(1)
    
    # Connect to Reachy
    print("\n2. Connecting to Reachy Mini...")
    try:
        reachy = ReachyMini()
        print("   ✅ Connected to Reachy")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        gyro_bridge.stop()
        return 1
    
    print("\n" + "=" * 60)
    print("LIVE GYRO DATA + ROBOT STATUS")
    print("=" * 60)
    print("\nPress Ctrl+C to exit\n")
    
    print(f"{'Time':>8} | {'Gyro X':>8} | {'Gyro Y':>8} | {'Gyro Z':>8} | {'Head Yaw':>10}")
    print("-" * 60)
    
    try:
        with reachy:
            while True:
                # Get gyro data from ESP
                gyro = gyro_bridge.get_latest()
                
                # Get robot head pose from Reachy
                try:
                    head_pose = reachy.get_current_head_pose()
                    # Extract yaw from pose matrix
                    import numpy as np
                    yaw = np.arctan2(head_pose[1, 0], head_pose[0, 0])
                    yaw_deg = yaw * 57.2958
                except:
                    yaw_deg = 0.0
                
                # Display
                t = time.strftime("%H:%M:%S")
                if gyro:
                    print(f"{t:>8} | {gyro.gyro_x:>8.3f} | {gyro.gyro_y:>8.3f} | {gyro.gyro_z:>8.3f} | {yaw_deg:>10.1f}°", end='\r')
                else:
                    print(f"{t:>8} | {'--':>8} | {'--':>8} | {'--':>8} | {yaw_deg:>10.1f}°", end='\r')
                
                time.sleep(0.05)  # 20Hz update
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        gyro_bridge.stop()
        print("\n✅ ESP reader stopped")
        print("✅ Reachy connection closed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
