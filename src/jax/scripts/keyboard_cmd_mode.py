#!/usr/bin/env python3
"""Keyboard teleop for Jax mode + velocity control.

Publishes:
- /cmd_vel (geometry_msgs/Twist)
- /jax_mode (std_msgs/String)

This node is intended to be run separately from simulation launch.
"""

import signal
import sys

import rclpy
from geometry_msgs.msg import Twist
from pynput import keyboard as kb
from rclpy.node import Node
from std_msgs.msg import String

HELP = """\
=== Jax Keyboard Cmd/Mode ===
Movement (hold):
  W/S : forward / back
  A/D : strafe left / right
  J/L : yaw left / right
  Shift: 2x speed

Body attitude (rest mode):
  Arrow Up/Down    -> pitch +/-
  Arrow Left/Right -> roll +/-
  9 / 0            -> raise / lower body

Modes (tap):
  1: trot
  2: rest
  3: sit
  4: lay

Stop:
  Space: zero twist immediately
=============================="""


class KeyboardCmdMode:
    def __init__(self, node: Node):
        self._node = node
        self._twist_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self._mode_pub = node.create_publisher(String, '/jax_mode', 10)

        self._speed_scale = 1.0
        self._base_linear = float(node.declare_parameter('linear_step', 0.40).value)
        self._base_yaw = float(node.declare_parameter('yaw_step', 0.80).value)
        self._base_attitude = float(node.declare_parameter('attitude_step', 0.50).value)
        self._base_height = float(node.declare_parameter('height_step', 0.75).value)
        self._publish_rate = float(node.declare_parameter('publish_rate_hz', 30.0).value)
        self._dashboard_rate = float(node.declare_parameter('dashboard_rate_hz', 5.0).value)

        self._twist = Twist()
        self._current_mode = 'rest'

        self._listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        node.create_timer(1.0 / max(self._publish_rate, 1.0), self._publish)
        node.create_timer(1.0 / max(self._dashboard_rate, 1.0), self._render_dashboard)
        node.get_logger().info('Keyboard cmd/mode node started')

    def _publish_mode(self, mode: str):
        msg = String()
        msg.data = mode
        self._mode_pub.publish(msg)
        self._current_mode = mode
        self._node.get_logger().info(f'Mode -> {mode}')

    def _publish(self):
        self._twist_pub.publish(self._twist)

    def _render_dashboard(self):
        t = self._twist
        lines = [
            "=== Jax Keyboard Cmd/Mode ===",
            f"Mode: {self._current_mode:<6} | Speed scale: {self._speed_scale:.1f}x",
            "",
            "Active command values:",
            f"  linear.x={t.linear.x:+.2f}  linear.y={t.linear.y:+.2f}  linear.z={t.linear.z:+.2f}",
            f"  angular.x={t.angular.x:+.2f} angular.y={t.angular.y:+.2f} angular.z={t.angular.z:+.2f}",
            "",
            "Movement (hold):",
            "  W/S forward/back, A/D strafe, J/L yaw, Shift 2x",
            "Body (rest):",
            "  Arrows pitch/roll, 9/0 height",
            "Modes (tap):",
            "  1 trot, 2 rest, 3 sit, 4 lay",
            "Stop:",
            "  Space zero twist",
        ]
        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def _on_press(self, key):
        lin = self._base_linear * self._speed_scale
        yaw = self._base_yaw * self._speed_scale
        att = self._base_attitude
        hgt = self._base_height

        try:
            ch = key.char
        except AttributeError:
            ch = None

        if key in (kb.Key.shift, kb.Key.shift_r):
            self._speed_scale = 2.0
            return

        if ch in ('w', 'W'):
            self._twist.linear.x = lin
        elif ch in ('s', 'S'):
            self._twist.linear.x = -lin
        elif ch in ('a', 'A'):
            self._twist.linear.y = lin
        elif ch in ('d', 'D'):
            self._twist.linear.y = -lin
        elif ch in ('j', 'J'):
            self._twist.angular.z = yaw
        elif ch in ('l', 'L'):
            self._twist.angular.z = -yaw
        elif ch == '1':
            self._publish_mode('trot')
        elif ch == '2':
            self._publish_mode('rest')
        elif ch == '3':
            self._publish_mode('sit')
        elif ch == '4':
            self._publish_mode('lay')
        elif ch == '9':
            self._twist.linear.z = -hgt
        elif ch == '0':
            self._twist.linear.z = hgt
        elif key == kb.Key.up:
            self._twist.angular.y = att
        elif key == kb.Key.down:
            self._twist.angular.y = -att
        elif key == kb.Key.left:
            self._twist.angular.x = att
        elif key == kb.Key.right:
            self._twist.angular.x = -att
        elif key == kb.Key.space:
            self._zero_twist()

    def _on_release(self, key):
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if key in (kb.Key.shift, kb.Key.shift_r):
            self._speed_scale = 1.0
        elif ch in ('w', 'W', 's', 'S'):
            self._twist.linear.x = 0.0
        elif ch in ('a', 'A', 'd', 'D'):
            self._twist.linear.y = 0.0
        elif ch in ('j', 'J', 'l', 'L'):
            self._twist.angular.z = 0.0
        elif ch in ('9', '0'):
            self._twist.linear.z = 0.0
        elif key in (kb.Key.up, kb.Key.down):
            self._twist.angular.y = 0.0
        elif key in (kb.Key.left, kb.Key.right):
            self._twist.angular.x = 0.0

    def _zero_twist(self):
        self._twist = Twist()

    def stop(self):
        self._zero_twist()
        self._publish()
        self._listener.stop()
        sys.stdout.write("\x1b[0m\n")
        sys.stdout.flush()


def main():
    rclpy.init()
    node = rclpy.create_node('keyboard_cmd_mode')

    teleop = KeyboardCmdMode(node)

    def _shutdown(*_):
        teleop.stop()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        rclpy.spin(node)
    finally:
        teleop.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
