#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

import board
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER
from adafruit_bno08x import BNO_REPORT_GYROSCOPE
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C


class BNO08XImuPublisher(Node):
    def __init__(self):
        super().__init__('bno08x_imu')

        self.declare_parameter('physical_imu_topic', '/imu/data')
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('imu_publish_hz', 50.0)
        self.declare_parameter('imu_i2c_address', 0x4A)
        self.declare_parameter('imu_retry_period_s', 2.0)

        self._topic = str(self.get_parameter('physical_imu_topic').value)
        self._frame_id = str(self.get_parameter('imu_frame_id').value)
        self._publish_hz = max(float(self.get_parameter('imu_publish_hz').value), 1.0)
        self._address = int(self.get_parameter('imu_i2c_address').value)
        self._retry_period_s = max(float(self.get_parameter('imu_retry_period_s').value), 0.5)

        self._imu_pub = self.create_publisher(Imu, self._topic, 10)
        self._sensor = None
        self._i2c = None
        self._last_init_attempt = 0.0
        self._init_error_logged = False

        self._orientation_variance = 0.02 * 0.02
        self._angular_velocity_variance = 0.03 * 0.03
        self._linear_acceleration_variance = 0.2 * 0.2

        self._timer = self.create_timer(1.0 / self._publish_hz, self._tick)
        self.get_logger().info(
            'BNO08x IMU publisher configured for '
            f'{self._topic} at 0x{self._address:02X} '
            '(direct Pi I2C mode only)'
        )

    def _covariance(self, variance: float):
        return [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance]

    def _connect(self):
        now = time.monotonic()
        if (now - self._last_init_attempt) < self._retry_period_s:
            return

        self._last_init_attempt = now
        try:
            self._i2c = board.I2C()
            self._sensor = BNO08X_I2C(self._i2c, address=self._address)
            self._sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self._sensor.enable_feature(BNO_REPORT_GYROSCOPE)
            self._sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
            self._init_error_logged = False
            self.get_logger().info(
                f'Connected to BNO08x on I2C address 0x{self._address:02X}'
            )
        except Exception as exc:
            self._sensor = None
            self._i2c = None
            if not self._init_error_logged:
                self.get_logger().warn(
                    f'BNO08x not available on I2C address 0x{self._address:02X}: {exc}'
                )
                self._init_error_logged = True

    def _tick(self):
        if self._sensor is None:
            self._connect()
            return

        try:
            quat_i, quat_j, quat_k, quat_real = self._sensor.quaternion
            gyro_x, gyro_y, gyro_z = self._sensor.gyro
            accel_x, accel_y, accel_z = self._sensor.acceleration
        except Exception as exc:
            self.get_logger().warn(f'BNO08x read failed, retrying connection: {exc}')
            self._sensor = None
            self._i2c = None
            return

        if any(math.isnan(value) for value in [quat_i, quat_j, quat_k, quat_real]):
            return

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.orientation.x = float(quat_i)
        msg.orientation.y = float(quat_j)
        msg.orientation.z = float(quat_k)
        msg.orientation.w = float(quat_real)
        msg.orientation_covariance = self._covariance(self._orientation_variance)
        msg.angular_velocity.x = float(gyro_x)
        msg.angular_velocity.y = float(gyro_y)
        msg.angular_velocity.z = float(gyro_z)
        msg.angular_velocity_covariance = self._covariance(self._angular_velocity_variance)
        msg.linear_acceleration.x = float(accel_x)
        msg.linear_acceleration.y = float(accel_y)
        msg.linear_acceleration.z = float(accel_z)
        msg.linear_acceleration_covariance = self._covariance(self._linear_acceleration_variance)
        self._imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BNO08XImuPublisher()
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