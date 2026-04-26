#!/usr/bin/env python3
"""Keyboard teleop for Jax mode + velocity control.

Publishes:
- /cmd_vel (geometry_msgs/Twist)
- /jax_mode (std_msgs/String)

This node is intended to be run separately from simulation launch.
"""

import signal
import os
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

HELP = """\
=== Jax Keyboard Cmd/Mode ===
Movement (hold):
  W/S : forward / back
  A/D : strafe left / right
  J/L : yaw left / right

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
        self._key_hold_s = max(float(node.declare_parameter('key_hold_s', 0.20).value), 0.05)
        self._publish_rate = float(node.declare_parameter('publish_rate_hz', 30.0).value)
        self._dashboard_rate = float(node.declare_parameter('dashboard_rate_hz', 5.0).value)

        self._twist = Twist()
        self._current_mode = 'rest'
        self._running = True
        self._stdin_fd = None
        self._stdin_old_attrs = None
        self._stdin_stream = None
        self._active_until = {
            'linear_x': 0.0,
            'linear_y': 0.0,
            'linear_z': 0.0,
            'angular_x': 0.0,
            'angular_y': 0.0,
            'angular_z': 0.0,
        }

        if sys.stdin.isatty():
            self._stdin_stream = sys.stdin
            self._stdin_fd = self._stdin_stream.fileno()
            self._stdin_old_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._stdin_thread.start()
            node.get_logger().info('Keyboard input backend: stdin TTY')
        else:
            try:
                self._stdin_stream = open('/dev/tty', 'rb', buffering=0)
                self._stdin_fd = self._stdin_stream.fileno()
                self._stdin_old_attrs = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
                self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
                self._stdin_thread.start()
                node.get_logger().info('Keyboard input backend: /dev/tty fallback')
            except OSError:
                self._stdin_thread = None
                node.get_logger().warn('No TTY available (stdin + /dev/tty failed); keyboard input disabled')

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
        self._expire_twist_components()
        self._twist_pub.publish(self._twist)

    def _set_twist_component(self, component: str, value: float):
        now = time.monotonic()
        if component == 'linear_x':
            self._twist.linear.x = value
        elif component == 'linear_y':
            self._twist.linear.y = value
        elif component == 'linear_z':
            self._twist.linear.z = value
        elif component == 'angular_x':
            self._twist.angular.x = value
        elif component == 'angular_y':
            self._twist.angular.y = value
        elif component == 'angular_z':
            self._twist.angular.z = value
        self._active_until[component] = now + self._key_hold_s

    def _expire_twist_components(self):
        now = time.monotonic()
        if now > self._active_until['linear_x']:
            self._twist.linear.x = 0.0
        if now > self._active_until['linear_y']:
            self._twist.linear.y = 0.0
        if now > self._active_until['linear_z']:
            self._twist.linear.z = 0.0
        if now > self._active_until['angular_x']:
            self._twist.angular.x = 0.0
        if now > self._active_until['angular_y']:
            self._twist.angular.y = 0.0
        if now > self._active_until['angular_z']:
            self._twist.angular.z = 0.0

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
            "  W/S forward/back, A/D strafe, J/L yaw",
            "Body (rest):",
            "  Arrows pitch/roll, 9/0 height",
            "Modes (tap):",
            "  1 trot, 2 rest, 3 sit, 4 lay",
            "Stop:",
            "  Space zero twist",
        ]
        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def _on_key(self, key: str):
        lin = self._base_linear * self._speed_scale
        yaw = self._base_yaw * self._speed_scale
        att = self._base_attitude
        hgt = self._base_height

        if key in ('w', 'W'):
            self._set_twist_component('linear_x', lin)
        elif key in ('s', 'S'):
            self._set_twist_component('linear_x', -lin)
        elif key in ('a', 'A'):
            self._set_twist_component('linear_y', lin)
        elif key in ('d', 'D'):
            self._set_twist_component('linear_y', -lin)
        elif key in ('j', 'J'):
            self._set_twist_component('angular_z', yaw)
        elif key in ('l', 'L'):
            self._set_twist_component('angular_z', -yaw)
        elif key == '1':
            self._publish_mode('trot')
        elif key == '2':
            self._publish_mode('rest')
        elif key == '3':
            self._publish_mode('sit')
        elif key == '4':
            self._publish_mode('lay')
        elif key == '9':
            self._set_twist_component('linear_z', -hgt)
        elif key == '0':
            self._set_twist_component('linear_z', hgt)
        elif key == '\x1b[A':
            self._set_twist_component('angular_y', att)
        elif key == '\x1b[B':
            self._set_twist_component('angular_y', -att)
        elif key == '\x1b[D':
            self._set_twist_component('angular_x', att)
        elif key == '\x1b[C':
            self._set_twist_component('angular_x', -att)
        elif key == ' ':
            self._zero_twist()

    def _stdin_loop(self):
        while self._running:
            if self._stdin_fd is None:
                break

            readable, _, _ = select.select([self._stdin_fd], [], [], 0.05)
            if not readable:
                continue

            raw = os.read(self._stdin_fd, 1)
            if not raw:
                continue

            key = raw.decode(errors='ignore')
            if key == '\x03':
                self._running = False
                break

            if key == '\x1b':
                if select.select([self._stdin_fd], [], [], 0.01)[0]:
                    key += os.read(self._stdin_fd, 1).decode(errors='ignore')
                if select.select([self._stdin_fd], [], [], 0.01)[0]:
                    key += os.read(self._stdin_fd, 1).decode(errors='ignore')

            self._on_key(key)

    def _zero_twist(self):
        self._twist = Twist()

    def stop(self):
        self._running = False
        self._zero_twist()
        self._publish()
        if self._stdin_fd is not None and self._stdin_old_attrs is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_attrs)
        if self._stdin_stream not in (None, sys.stdin):
            self._stdin_stream.close()
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
