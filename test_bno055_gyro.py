"""Test BNO055 gyroscope from ESP32.

Protocol (from ESP firmware):
- Serial: 460800 baud
- Packet: [0xAA] [0x55] [LEN=93] [payload] [XOR checksum]
- Payload: quat(4f) + euler(3f) + acc(3f) + lin(3f) + grav(3f) + gyro(3f) + mag(3f) + temp(1b) + cal(4B)
"""

import serial
import struct
import time
from dataclasses import dataclass


@dataclass
class BNO055Packet:
    """Decoded BNO055 sensor data."""
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    euler_h: float
    euler_r: float
    euler_p: float
    acc_x: float
    acc_y: float
    acc_z: float
    lin_x: float
    lin_y: float
    lin_z: float
    grav_x: float
    grav_y: float
    grav_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    temp: int
    cal_sys: int
    cal_gyro: int
    cal_acc: int
    cal_mag: int


def decode_packet(payload: bytes) -> BNO055Packet:
    """Decode 93-byte payload."""
    fmt = "<4f 3f 3f 3f 3f 3f 3f b 4B"
    u = struct.unpack(fmt, payload)
    return BNO055Packet(
        quat_w=u[0], quat_x=u[1], quat_y=u[2], quat_z=u[3],
        euler_h=u[4], euler_r=u[5], euler_p=u[6],
        acc_x=u[7], acc_y=u[8], acc_z=u[9],
        lin_x=u[10], lin_y=u[11], lin_z=u[12],
        grav_x=u[13], grav_y=u[14], grav_z=u[15],
        gyro_x=u[16], gyro_y=u[17], gyro_z=u[18],
        mag_x=u[19], mag_y=u[20], mag_z=u[21],
        temp=u[22],
        cal_sys=u[23], cal_gyro=u[24], cal_acc=u[25], cal_mag=u[26],
    )


def main():
    PORT = "/dev/cu.usbserial-83430"
    BAUD = 460800
    HEADER = b"\xaa\x55"
    PAYLOAD_SIZE = 93
    PACKET_SIZE = PAYLOAD_SIZE + 4

    print("=" * 70)
    print("BNO055 GYROSCOPE TEST")
    print("=" * 70)
    print(f"Port: {PORT}")
    print(f"Baud: {BAUD}")
    print()

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    print("Connected. Reading data...")
    print("Press Ctrl+C to stop\n")

    print(f"{'Time':>8} | {'Gyro X':>8} | {'Gyro Y':>8} | {'Gyro Z':>8} | {'Euler H':>8} | {'Cal':>6}")
    print("-" * 70)

    buffer = b""
    packets = 0

    try:
        while True:
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting)

            # Find packets in buffer
            while len(buffer) >= PACKET_SIZE:
                idx = buffer.find(HEADER)
                if idx == -1:
                    buffer = buffer[-2:] if len(buffer) >= 2 else b""
                    break

                if len(buffer) < idx + PACKET_SIZE:
                    break

                packet = buffer[idx : idx + PACKET_SIZE]
                buffer = buffer[idx + PACKET_SIZE :]

                # Verify length byte
                if packet[2] != PAYLOAD_SIZE:
                    continue

                # Verify XOR checksum
                payload = packet[3 : 3 + PAYLOAD_SIZE]
                checksum = packet[3 + PAYLOAD_SIZE]
                xor_sum = 0
                for b in payload:
                    xor_sum ^= b

                if xor_sum != checksum:
                    continue

                # Decode
                data = decode_packet(payload)
                packets += 1

                t = time.strftime("%H:%M:%S")
                cal = f"{data.cal_sys}{data.cal_gyro}{data.cal_acc}{data.cal_mag}"
                print(
                    f"{t:>8} | {data.gyro_x:>8.2f} | {data.gyro_y:>8.2f} | {data.gyro_z:>8.2f} | "
                    f"{data.euler_h:>8.1f} | {cal:>6}",
                    end="\r",
                )

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        ser.close()
        print(f"Total packets decoded: {packets}")


if __name__ == "__main__":
    main()
