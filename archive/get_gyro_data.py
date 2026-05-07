"""Get BNO055 gyroscope data from ESP32.

This is the working script to read gyro data from the Reachy head ESP.

Usage:
    ./.venv/bin/python get_gyro_data.py

Press Ctrl+C to stop.

If no data appears:
1. Check ESP LED is ON
2. Try unplugging and replugging the ESP USB
3. Or press the ESP reset button
4. Then run this script again
"""

import serial
import struct
import time
import sys


def read_gyro_loop(port="/dev/cu.usbserial-83430", baud=460800):
    """Read and display gyro data from ESP BNO055."""
    
    print("=" * 70)
    print("BNO055 GYROSCOPE READER")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Baud: {baud}")
    print()
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"❌ Failed to open port: {e}")
        print("\nIs the ESP connected? Check USB connection.")
        return False
    
    print("✅ Connected to ESP")
    print("Waiting for data... (move the head to see gyro changes)")
    print("Press Ctrl+C to stop\n")
    
    # BNO055 packet format
    HEADER = b"\xaa\x55"
    PAYLOAD_SIZE = 93
    PACKET_SIZE = PAYLOAD_SIZE + 4  # + header + len + checksum
    
    # Format: quat(4f) + euler(3f) + acc(3f) + lin(3f) + grav(3f) + gyro(3f) + mag(3f) + temp(b) + cal(4B)
    FMT = "<4f 3f 3f 3f 3f 3f 3f b 4B"
    
    buffer = b""
    packet_count = 0
    last_display = 0
    
    # Print header
    print(f"{'Time':>8} | {'Gyro X':>8} | {'Gyro Y':>8} | {'Gyro Z':>8} | {'Heading':>8} | {'Cal':>5}")
    print("-" * 70)
    
    try:
        while True:
            # Read available data
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting)
            
            # Process packets
            while len(buffer) >= PACKET_SIZE:
                idx = buffer.find(HEADER)
                if idx == -1:
                    # Keep last 2 bytes (might be start of header)
                    buffer = buffer[-2:] if len(buffer) >= 2 else b""
                    break
                
                if len(buffer) < idx + PACKET_SIZE:
                    break  # Need more data
                
                packet = buffer[idx:idx + PACKET_SIZE]
                buffer = buffer[idx + PACKET_SIZE:]
                
                # Verify length
                if packet[2] != PAYLOAD_SIZE:
                    continue
                
                # Verify checksum (XOR of payload)
                payload = packet[3:3 + PAYLOAD_SIZE]
                checksum = packet[3 + PAYLOAD_SIZE]
                xor_sum = 0
                for b in payload:
                    xor_sum ^= b
                
                if xor_sum != checksum:
                    continue  # Bad packet
                
                # Decode
                data = struct.unpack(FMT, payload)
                packet_count += 1
                
                # Extract gyro (indices 16, 17, 18) and euler heading (index 4)
                gyro_x, gyro_y, gyro_z = data[16], data[17], data[18]
                heading = data[4]  # Euler H (heading/yaw)
                
                # Calibration status
                cal_sys, cal_gyro, cal_acc, cal_mag = data[23], data[24], data[25], data[26]
                cal_str = f"{int(cal_sys)}{int(cal_gyro)}{int(cal_acc)}{int(cal_mag)}"
                
                # Display at 10Hz
                now = time.time()
                if now - last_display >= 0.1:
                    t = time.strftime("%H:%M:%S")
                    print(f"{t:>8} | {gyro_x:>8.2f} | {gyro_y:>8.2f} | {gyro_z:>8.2f} | {heading:>8.1f} | {cal_str:>5}", end="\r")
                    last_display = now
            
            time.sleep(0.001)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        ser.close()
        print(f"\nTotal packets: {packet_count}")
        if packet_count == 0:
            print("\n⚠️  NO DATA RECEIVED")
            print("\nTroubleshooting:")
            print("1. Check ESP LED is ON")
            print("2. Unplug and replug the ESP USB cable")
            print("3. Press the ESP reset button")
            print("4. Then run this script again")
        return packet_count > 0


if __name__ == "__main__":
    success = read_gyro_loop()
    sys.exit(0 if success else 1)
