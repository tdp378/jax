#!/usr/bin/env python3
"""Map geometry_msgs/Twist (/cmd_vel) -> sensor_msgs/Joy on /joy for jax_driver.

teleop_twist_keyboard (and similar) publish Twist; the quadruped stack expects Joy axes
matching InputInterface (axes[1]=forward, axes[0]=strafe, axes[3]=yaw rate).

Do not run this together with Keyboard.py or joy_node — only one publisher should own /joy.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


def _clamp(v: float, lim: float = 1.0) -> float:
    return max(-lim, min(lim, v))


class CmdVelToJoyBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_joy_bridge')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('max_linear_x', 1.2)
        self.declare_parameter('max_linear_y', 0.5)
        self.declare_parameter('max_angular_z', 2.0)
        self.declare_parameter('publish_rate', 30.0)

        self._twist = Twist()
        self._joy = Joy()
        self._joy.axes = [0.0] * 8
        self._joy.buttons = [0] * 11

        in_topic = self.get_parameter('input_topic').value
        self.create_subscription(Twist, in_topic, self._cb, 10)
        self._pub = self.create_publisher(Joy, '/joy', qos_profile_sensor_data)

        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'Bridging Twist from {in_topic!r} -> /joy '
            f'(match jax_control.Config max lin x/y, yaw)'
        )

    def _cb(self, msg: Twist):
        self._twist = msg

    def _tick(self):
        mx = float(self.get_parameter('max_linear_x').value)
        my = float(self.get_parameter('max_linear_y').value)
        mz = float(self.get_parameter('max_angular_z').value)
        t = self._twist
        # InputInterface: x_vel = axes[1]*mx, y_vel = axes[0]*my, yaw = axes[3]*mz
        self._joy.axes[0] = _clamp(t.linear.y / my, 1.0) if my > 1e-6 else 0.0
        self._joy.axes[1] = _clamp(t.linear.x / mx, 1.0) if mx > 1e-6 else 0.0
        self._joy.axes[3] = _clamp(t.angular.z / mz, 1.0) if mz > 1e-6 else 0.0
        for i in (2, 4, 5, 6, 7):
            self._joy.axes[i] = 0.0
        self._joy.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._joy)


def main():
    rclpy.init()
    node = CmdVelToJoyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
