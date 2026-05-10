#!/usr/bin/env python3
"""
Reads BNO085 yaw/pitch/roll from a Pi serial UART,
converts to a quaternion, and publishes sensor_msgs/Imu.

Serial protocol (framed with 0xAA 0xAA header, 17-byte payload):
  byte  0   : packet counter
  bytes 1-2 : yaw   (little-endian int16, units = degrees * 100)
  bytes 3-4 : roll  (little-endian int16, units = degrees * 100)
  bytes 5-6 : pitch (little-endian int16, units = degrees * 100)
  bytes 7-16: reserved
"""

import math
import serial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def _decode_angle(lsb: int, msb: int) -> float:
    """Return signed degrees from a little-endian int16 scaled by 100."""
    val = (msb << 8) | lsb
    if val > 32767:
        val -= 65536
    return val / 100.0


def _euler_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """
    Convert yaw / pitch / roll (in degrees, ZYX convention) to a unit quaternion
    (x, y, z, w).
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y_, z, w


class SerialImuPublisher(Node):
    def __init__(self):
        super().__init__('serial_imu')

        self.declare_parameter('serial_port', '/dev/ttyAMA3')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('orientation_variance', 0.0004)   # ~0.02 deg²

        self._port = str(self.get_parameter('serial_port').value)
        self._baud = int(self.get_parameter('baud_rate').value)
        self._topic = str(self.get_parameter('imu_topic').value)
        self._frame_id = str(self.get_parameter('imu_frame_id').value)
        self._ori_var = float(self.get_parameter('orientation_variance').value)

        self._pub = self.create_publisher(Imu, self._topic, 10)
        self._ser: serial.Serial | None = None

        # Timer drives serial polling at ~200 Hz to stay ahead of incoming packets
        self._timer = self.create_timer(0.005, self._tick)
        self.get_logger().info(
            f'Serial IMU publisher ready → {self._topic} (port={self._port})'
        )

    # ------------------------------------------------------------------
    # Serial lifecycle
    # ------------------------------------------------------------------

    def _open_serial(self) -> bool:
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0)
            self.get_logger().info(f'Opened serial port {self._port} at {self._baud} baud')
            return True
        except serial.SerialException as exc:
            self.get_logger().warn(f'Cannot open serial port {self._port}: {exc}')
            self._ser = None
            return False

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _tick(self):
        if self._ser is None or not self._ser.is_open:
            self._open_serial()
            return

        try:
            self._drain_packets()
        except serial.SerialException as exc:
            self.get_logger().warn(f'Serial read error, closing port: {exc}')
            self._ser.close()
            self._ser = None

    def _drain_packets(self):
        """Read and process all complete packets currently in the serial buffer."""
        while True:
            # Look for the 0xAA 0xAA frame header
            b = self._ser.read(1)
            if not b:
                return
            if b != b'\xaa':
                continue

            b = self._ser.read(1)
            if not b:
                return
            if b != b'\xaa':
                continue

            payload = self._ser.read(17)
            if len(payload) != 17:
                return

            self._publish(payload)

    # ------------------------------------------------------------------
    # Packet processing
    # ------------------------------------------------------------------

    def _publish(self, p: bytes):
        yaw   = _decode_angle(p[1], p[2])
        roll  = _decode_angle(p[3], p[4])
        pitch = _decode_angle(p[5], p[6])

        # Normalise yaw to 0–360
        if yaw < 0:
            yaw += 360.0

        qx, qy, qz, qw = _euler_to_quaternion(yaw, pitch, roll)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.orientation_covariance = [
            self._ori_var, 0.0, 0.0,
            0.0, self._ori_var, 0.0,
            0.0, 0.0, self._ori_var,
        ]

        # Gyro and linear-acceleration not available from this serial stream
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SerialImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
