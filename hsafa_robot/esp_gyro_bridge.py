"""ESP Gyroscope Bridge for Hsafa Robot.

Integrates ESP32-based gyroscope with the Reachy Mini SDK.
The ESP sends binary data over USB serial.
"""

import threading
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

# Optional serial import
try:
    import serial
    import struct
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

log = logging.getLogger(__name__)


@dataclass
class GyroSample:
    """Single gyroscope reading."""
    timestamp: float
    gyro_x: float  # rad/s
    gyro_y: float
    gyro_z: float
    accel_x: float  # m/s²
    accel_y: float
    accel_z: float


class ESPGyroBridge:
    """Bridge to read gyro data from ESP and make it available to Hsafa.
    
    Usage:
        from hsafa_robot.esp_gyro_bridge import ESPGyroBridge
        
        bridge = ESPGyroBridge()
        bridge.start()
        
        # Later, in your code:
        gyro = bridge.get_latest()
        if gyro:
            print(f"Gyro: {gyro.gyro_x}, {gyro.gyro_y}, {gyro.gyro_z}")
    """
    
    DEFAULT_PORT = '/dev/cu.usbserial-83430'
    DEFAULT_BAUD = 115200
    
    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD):
        self.port = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[GyroSample] = None
        self._lock = threading.Lock()
        
        if not HAS_SERIAL:
            log.error("pyserial not installed. Run: pip install pyserial")
    
    def start(self) -> bool:
        """Start the ESP reader thread."""
        if not HAS_SERIAL:
            return False
            
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            log.info(f"ESP gyro reader started on {self.port}")
            return True
        except Exception as e:
            log.error(f"Failed to start ESP reader: {e}")
            return False
    
    def stop(self):
        """Stop the reader."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial:
            self._serial.close()
            self._serial = None
        log.info("ESP gyro reader stopped")
    
    def get_latest(self) -> Optional[GyroSample]:
        """Get the latest gyro reading."""
        with self._lock:
            return self._latest
    
    def get_gyro_degrees(self) -> Optional[Tuple[float, float, float]]:
        """Get gyro values in degrees/sec."""
        sample = self.get_latest()
        if sample:
            # Convert rad/s to deg/s if needed
            return (
                sample.gyro_x * 57.2958,
                sample.gyro_y * 57.2958,
                sample.gyro_z * 57.2958
            )
        return None
    
    def _read_loop(self):
        """Background thread to read from ESP."""
        buffer = b''
        packet_size = 21  # Based on earlier analysis
        
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    data = self._serial.read(self._serial.in_waiting)
                    buffer += data
                    
                    # Process complete packets
                    while len(buffer) >= packet_size:
                        packet = buffer[:packet_size]
                        sample = self._decode_packet(packet)
                        
                        if sample:
                            with self._lock:
                                self._latest = sample
                        
                        buffer = buffer[packet_size:]
                
                # Prevent buffer overflow
                if len(buffer) > 1024:
                    buffer = buffer[-packet_size:]
                    
            except Exception as e:
                log.warning(f"ESP read error: {e}")
                time.sleep(0.1)
    
    def _decode_packet(self, packet: bytes) -> Optional[GyroSample]:
        """Decode ESP binary packet to gyro sample.
        
        NOTE: This is a placeholder decoder. The actual format needs
        to be determined by analyzing the ESP firmware output.
        
        Based on hex analysis, packets are 21 bytes with structure:
        - Bytes 0-3: Header (varies)
        - Bytes 4-15: 3 floats (gyro or accel data)
        - Bytes 16-20: Footer/timestamp
        """
        if len(packet) < 21:
            return None
        
        try:
            # Attempt to decode as 3 floats at offset 4
            # This is a guess - actual format may differ
            g1, g2, g3 = struct.unpack('<3f', packet[4:16])
            
            # Values might need scaling/calibration
            # For now, pass through raw
            return GyroSample(
                timestamp=time.time(),
                gyro_x=g1,
                gyro_y=g2,
                gyro_z=g3,
                accel_x=0.0,  # Unknown until format verified
                accel_y=0.0,
                accel_z=9.81  # Assume gravity aligned
            )
        except:
            return None


def test_esp_gyro():
    """Quick test function."""
    import sys
    
    print("ESP Gyro Bridge Test")
    print("=" * 50)
    
    bridge = ESPGyroBridge()
    if not bridge.start():
        print("Failed to start. Is ESP connected?")
        sys.exit(1)
    
    print(f"Reading from {bridge.port}...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for i in range(100):  # 10 seconds
            sample = bridge.get_latest()
            if sample:
                print(f"Gyro (rad/s): X={sample.gyro_x:8.3f} Y={sample.gyro_y:8.3f} Z={sample.gyro_z:8.3f}", end='\r')
            else:
                print(f"Waiting for data... ({i}/100)", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        print("\n\nStopped.")


if __name__ == "__main__":
    test_esp_gyro()
