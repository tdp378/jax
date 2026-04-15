#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        self._odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self._cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self._odom_frame = self.declare_parameter('odom_frame_id', 'odom').value
        self._base_frame = self.declare_parameter('base_frame_id', 'base_link').value
        self._current_mode_topic = self.declare_parameter('current_mode_topic', '/jax/current_mode').value
        self._rate_hz = float(self.declare_parameter('publish_rate_hz', 30.0).value)
        self._cmd_vel_timeout_s = float(self.declare_parameter('cmd_vel_timeout_s', 0.25).value)

        if self._rate_hz <= 0.0:
            self._rate_hz = 30.0

        self._dt = 1.0 / self._rate_hz
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._mode = 'rest'
        self._last_cmd_time = None
        self._last_update_time = self.get_clock().now()

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 10)
        self._cmd_sub = self.create_subscription(Twist, self._cmd_vel_topic, self._cmd_cb, 10)
        self._mode_sub = self.create_subscription(String, self._current_mode_topic, self._mode_cb, 10)
        self._timer = self.create_timer(self._dt, self._publish)

        self.get_logger().info(
            f'Odom publisher active on {self._odom_topic} from {self._cmd_vel_topic} '
            f'(rate={self._rate_hz:.1f} Hz)'
        )

    def _cmd_cb(self, msg: Twist):
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._wz = float(msg.angular.z)
        self._last_cmd_time = self.get_clock().now()

    def _mode_cb(self, msg: String):
        self._mode = msg.data.strip().lower()

    def _publish(self):
        now = self.get_clock().now()
        dt = (now - self._last_update_time).nanoseconds * 1e-9
        self._last_update_time = now

        # Guard against paused/reset sim time or long scheduling gaps.
        if dt <= 0.0:
            dt = self._dt
        dt = min(dt, 0.1)

        age = None
        if self._last_cmd_time is not None:
            age = (now - self._last_cmd_time).nanoseconds * 1e-9

        cmd_fresh = age is not None and age <= self._cmd_vel_timeout_s
        moving_mode = self._mode in {'trot', 'walk'}

        vx = self._vx if moving_mode and cmd_fresh else 0.0
        vy = self._vy if moving_mode and cmd_fresh else 0.0
        wz = self._wz if moving_mode and cmd_fresh else 0.0

        # Integrate robot-frame velocity into world-frame pose.
        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        world_vx = c * vx - s * vy
        world_vy = s * vx + c * vy

        self._x += world_vx * dt
        self._y += world_vy * dt
        self._yaw += wz * dt

        half = 0.5 * self._yaw
        qz = math.sin(half)
        qw = math.cos(half)

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.angular.z = wz

        self._odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()