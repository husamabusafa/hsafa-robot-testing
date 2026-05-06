"""Head Gyroscope integration for Hsafa Robot.

Reads BNO055 IMU data from ESP32 in Reachy Mini head.
Provides gyro, accelerometer, quaternion, and orientation data.

Usage:
    from hsafa_robot.head_gyro import HeadGyro
    
    gyro = HeadGyro()
    gyro.start()
    
    # In your control loop:
    data = gyro.get_latest()
    if data:
        print(f"Gyro: {data.gyro_x}, {data.gyro_y}, {data.gyro_z}")
"""

import serial
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GyroData:
    """Complete IMU data from head-mounted BNO055."""
    # Timestamp
    timestamp: float
    
    # Quaternion (w, x, y, z)
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    
    # Euler angles (heading, roll, pitch) in degrees
    heading: float
    roll: float
    pitch: float
    
    # Accelerometer (m/s², includes gravity)
    acc_x: float
    acc_y: float
    acc_z: float
    
    # Linear acceleration (m/s², gravity removed)
    lin_acc_x: float
    lin_acc_y: float
    lin_acc_z: float
    
    # Gravity vector (m/s²)
    grav_x: float
    grav_y: float
    grav_z: float
    
    # Gyroscope (deg/s) - PRIMARY DATA
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Gyroscope in rad/s
    gyro_x_rad: float
    gyro_y_rad: float
    gyro_z_rad: float
    
    # Magnetometer (uT)
    mag_x: float
    mag_y: float
    mag_z: float
    
    # Temperature (°C)
    temp: int
    
    # Calibration status (0-3, 3 is fully calibrated)
    cal_sys: int
    cal_gyro: int
    cal_acc: int
    cal_mag: int


class HeadGyro:
    """Reader for head-mounted BNO055 gyroscope via ESP32."""
    
    def __init__(self, port: str = "/dev/cu.usbserial-83430", baud: int = 460800):
        self.port = port
        self.baud = baud
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[GyroData] = None
        self._lock = threading.Lock()
        self._packet_count = 0
        
        # Protocol constants
        self._header = b"\xaa\x55"
        self._payload_size = 93
        self._packet_size = self._payload_size + 4
        self._fmt = "<4f 3f 3f 3f 3f 3f 3f b 4B"
    
    def start(self) -> bool:
        """Start reading gyro data from ESP."""
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[HeadGyro] Failed to start: {e}")
            return False
    
    def stop(self):
        """Stop reading."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial:
            self._serial.close()
            self._serial = None
    
    def get_latest(self) -> Optional[GyroData]:
        """Get the latest sensor reading."""
        with self._lock:
            return self._latest
    
    def get_gyro_degrees(self) -> Optional[tuple[float, float, float]]:
        """Get gyroscope values in degrees/sec."""
        data = self.get_latest()
        if data:
            return (data.gyro_x, data.gyro_y, data.gyro_z)
        return None
    
    def get_gyro_radians(self) -> Optional[tuple[float, float, float]]:
        """Get gyroscope values in radians/sec."""
        data = self.get_latest()
        if data:
            return (data.gyro_x_rad, data.gyro_y_rad, data.gyro_z_rad)
        return None
    
    def is_calibrated(self) -> bool:
        """Check if gyro is fully calibrated."""
        data = self.get_latest()
        if data:
            return data.cal_gyro == 3
        return False
    
    def _read_loop(self):
        """Background thread to read from ESP."""
        buffer = b""
        
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    buffer += self._serial.read(self._serial.in_waiting)
                
                # Process complete packets
                while len(buffer) >= self._packet_size:
                    idx = buffer.find(self._header)
                    if idx == -1:
                        buffer = buffer[-2:] if len(buffer) >= 2 else b""
                        break
                    
                    if len(buffer) < idx + self._packet_size:
                        break
                    
                    packet = buffer[idx:idx + self._packet_size]
                    buffer = buffer[idx + self._packet_size:]
                    
                    # Verify length
                    if packet[2] != self._payload_size:
                        continue
                    
                    # Verify checksum
                    payload = packet[3:3 + self._payload_size]
                    checksum = packet[3 + self._payload_size]
                    xor_sum = 0
                    for b in payload:
                        xor_sum ^= b
                    
                    if xor_sum != checksum:
                        continue
                    
                    # Decode
                    data = struct.unpack(self._fmt, payload)
                    self._packet_count += 1
                    
                    # Create GyroData object
                    gyro_data = GyroData(
                        timestamp=time.time(),
                        quat_w=data[0], quat_x=data[1], quat_y=data[2], quat_z=data[3],
                        heading=data[4], roll=data[5], pitch=data[6],
                        acc_x=data[7], acc_y=data[8], acc_z=data[9],
                        lin_acc_x=data[10], lin_acc_y=data[11], lin_acc_z=data[12],
                        grav_x=data[13], grav_y=data[14], grav_z=data[15],
                        gyro_x=data[16], gyro_y=data[17], gyro_z=data[18],
                        gyro_x_rad=data[16] * 0.0174533,
                        gyro_y_rad=data[17] * 0.0174533,
                        gyro_z_rad=data[18] * 0.0174533,
                        mag_x=data[19], mag_y=data[20], mag_z=data[21],
                        temp=data[22],
                        cal_sys=data[23], cal_gyro=data[24],
                        cal_acc=data[25], cal_mag=data[26],
                    )
                    
                    with self._lock:
                        self._latest = gyro_data
                
                # Prevent buffer overflow
                if len(buffer) > 4096:
                    buffer = buffer[-self._packet_size:]
                    
            except Exception as e:
                time.sleep(0.01)


def test_head_gyro():
    """Quick test - run this to verify gyro is working."""
    import sys
    
    print("=" * 70)
    print("HEAD GYROSCOPE TEST")
    print("=" * 70)
    print()
    
    gyro = HeadGyro()
    if not gyro.start():
        print("Failed to start. Is ESP connected?")
        sys.exit(1)
    
    print("Reading... Move the head to see gyro changes.")
    print("Press Ctrl+C to stop\n")
    print(f"{'Time':>8} | {'Gyro X':>8} | {'Gyro Y':>8} | {'Gyro Z':>8} | {'Heading':>8} | {'Cal':>5}")
    print("-" * 70)
    
    try:
        while True:
            data = gyro.get_latest()
            if data:
                t = time.strftime("%H:%M:%S")
                cal = f"{data.cal_sys}{data.cal_gyro}{data.cal_acc}{data.cal_mag}"
                print(f"{t:>8} | {data.gyro_x:>8.2f} | {data.gyro_y:>8.2f} | {data.gyro_z:>8.2f} | {data.heading:>8.1f} | {cal:>5}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        gyro.stop()


if __name__ == "__main__":
    test_head_gyro()
