"""Live ESP BNO055 Gyroscope Reader.

Decodes the binary protocol from ESP32 running the BNO055 IMU.

Protocol:
- Baud: 460800
- Packet: [0xAA] [0x55] [LEN] [93 bytes payload] [XOR checksum]
- Payload: quat(4f) + euler(3f) + acc(3f) + lin(3f) + grav(3f) + gyro(3f) + mag(3f) + temp(1b) + cal(4b)
"""

import serial
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BNO055Data:
    """Complete BNO055 sensor data."""
    # Quaternion (w, x, y, z)
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    
    # Euler angles (heading, roll, pitch) in degrees
    euler_h: float
    euler_r: float
    euler_p: float
    
    # Accelerometer (m/s², includes gravity)
    acc_x: float
    acc_y: float
    acc_z: float
    
    # Linear acceleration (m/s², gravity removed)
    lin_x: float
    lin_y: float
    lin_z: float
    
    # Gravity vector (m/s²)
    grav_x: float
    grav_y: float
    grav_z: float
    
    # Gyroscope (deg/s) - THIS IS WHAT YOU WANT
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Magnetometer (uT)
    mag_x: float
    mag_y: float
    mag_z: float
    
    # Temperature (°C)
    temp: int
    
    # Calibration status (0-3 for each)
    cal_sys: int
    cal_gyro: int
    cal_acc: int
    cal_mag: int
    
    timestamp: float


class BNO055Reader:
    """Reader for BNO055 binary protocol from ESP."""
    
    HEADER = bytes([0xAA, 0x55])
    PAYLOAD_SIZE = 93  # sizeof(Packet) from ESP code
    PACKET_SIZE = PAYLOAD_SIZE + 4  # + header(2) + len(1) + checksum(1)
    
    def __init__(self, port: str = '/dev/cu.usbserial-83430'):
        self.port = port
        self.serial: Optional[serial.Serial] = None
        self.buffer = b''
        
    def connect(self) -> bool:
        """Connect to ESP at 460800 baud."""
        try:
            self.serial = serial.Serial(self.port, 460800, timeout=1)
            print(f"✅ Connected to ESP at 460800 baud")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial:
            self.serial.close()
            self.serial = None
    
    def _find_packet(self) -> Optional[bytes]:
        """Find and extract next complete packet from buffer."""
        # Look for header 0xAA 0x55
        while len(self.buffer) >= self.PACKET_SIZE:
            idx = self.buffer.find(self.HEADER)
            if idx == -1:
                # No header found, discard buffer except last byte (might be start of header)
                self.buffer = self.buffer[-1:] if self.buffer else b''
                return None
            
            # Check if we have full packet
            if len(self.buffer) < idx + self.PACKET_SIZE:
                return None  # Wait for more data
            
            # Extract packet
            packet = self.buffer[idx:idx + self.PACKET_SIZE]
            self.buffer = self.buffer[idx + self.PACKET_SIZE:]
            
            # Verify length byte
            length = packet[2]
            if length != self.PAYLOAD_SIZE:
                continue  # Invalid length, skip
            
            # Verify checksum (XOR of payload)
            payload = packet[3:3 + self.PAYLOAD_SIZE]
            checksum = packet[3 + self.PAYLOAD_SIZE]
            xor_sum = 0
            for b in payload:
                xor_sum ^= b
            
            if xor_sum == checksum:
                return payload
            # Checksum failed, continue looking
        
        return None
    
    def _decode_payload(self, payload: bytes) -> BNO055Data:
        """Decode 93-byte payload into sensor data."""
        # Unpack all floats (little-endian)
        # Format: < means little-endian, f means float, b means signed char, B means unsigned char
        fmt = '<4f 3f 3f 3f 3f 3f 3f 3f b 4B'
        
        unpacked = struct.unpack(fmt, payload)
        
        return BNO055Data(
            # Quaternion (w, x, y, z)
            quat_w=unpacked[0],
            quat_x=unpacked[1],
            quat_y=unpacked[2],
            quat_z=unpacked[3],
            
            # Euler (h, r, p)
            euler_h=unpacked[4],
            euler_r=unpacked[5],
            euler_p=unpacked[6],
            
            # Accel
            acc_x=unpacked[7],
            acc_y=unpacked[8],
            acc_z=unpacked[9],
            
            # Linear accel
            lin_x=unpacked[10],
            lin_y=unpacked[11],
            lin_z=unpacked[12],
            
            # Gravity
            grav_x=unpacked[13],
            grav_y=unpacked[14],
            grav_z=unpacked[15],
            
            # Gyro (deg/s)
            gyro_x=unpacked[16],
            gyro_y=unpacked[17],
            gyro_z=unpacked[18],
            
            # Magnetometer
            mag_x=unpacked[19],
            mag_y=unpacked[20],
            mag_z=unpacked[21],
            
            # Temperature
            temp=unpacked[22],
            
            # Calibration
            cal_sys=unpacked[23],
            cal_gyro=unpacked[24],
            cal_acc=unpacked[25],
            cal_mag=unpacked[26],
            
            timestamp=time.time()
        )
    
    def read(self) -> Optional[BNO055Data]:
        """Read one complete sensor frame."""
        if not self.serial:
            return None
        
        # Fill buffer
        if self.serial.in_waiting:
            self.buffer += self.serial.read(self.serial.in_waiting)
        
        # Find and decode packet
        payload = self._find_packet()
        if payload:
            return self._decode_payload(payload)
        
        return None
    
    def read_blocking(self, timeout: float = 2.0) -> Optional[BNO055Data]:
        """Read with blocking until data received or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            data = self.read()
            if data:
                return data
            time.sleep(0.001)
        return None


def main():
    """Live display of gyro data."""
    print("=" * 70)
    print("ESP BNO055 GYROSCOPE LIVE READER")
    print("=" * 70)
    print()
    
    reader = BNO055Reader()
    if not reader.connect():
        return
    
    print("Reading data... Press Ctrl+C to stop\n")
    
    # Header
    print(f"{'Time':>8} | {'Gyro X':>8} | {'Gyro Y':>8} | {'Gyro Z':>8} | {'Cal':>8}")
    print("-" * 70)
    
    try:
        while True:
            data = reader.read()
            if data:
                t = time.strftime("%H:%M:%S")
                cal = f"{data.cal_sys}{data.cal_gyro}{data.cal_acc}{data.cal_mag}"
                print(f"{t:>8} | {data.gyro_x:>8.2f} | {data.gyro_y:>8.2f} | {data.gyro_z:>8.2f} | {cal:>8}", end='\r')
            else:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        reader.disconnect()
    
    print("Done.")


if __name__ == "__main__":
    main()
