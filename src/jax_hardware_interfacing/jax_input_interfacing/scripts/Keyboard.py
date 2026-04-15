#!/usr/bin/env python3
"""Keyboard → /joy publisher using pynput (requires DISPLAY + python3-pynput).

Simple, no termios tricks. Yes, typed characters echo in the terminal — that's harmless.
Focus the launch terminal so pynput receives key events.
"""
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy

from pynput import keyboard as kb

HELP = """\
=== Jax Keyboard ===
 W/S     forward / back
 A/D     strafe left / right
 Arrows  pitch (up/down) / yaw (left/right)
 9/0     raise / lower body
 8/7     roll
 1       toggle TROT gait  ← PRESS THIS FIRST to walk!
 2       hop
 Bksp    toggle manual / external control
 Shift   hold for 2× speed
======================="""


class KeyboardJoy:
    def __init__(self, node: Node):
        self._node = node
        self._pub = node.create_publisher(Joy, '/joy', qos_profile_sensor_data)
        self._msg = Joy()
        self._msg.axes = [0.0] * 8
        self._msg.buttons = [0] * 11
        self._speed = 1

        self._listener = kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        node.create_timer(1.0 / 30.0, self._publish)
        node.get_logger().info('Keyboard node started — publishing /joy at 30 Hz')

    def _publish(self):
        self._msg.header.stamp = self._node.get_clock().now().to_msg()
        self._pub.publish(self._msg)

    def _on_press(self, key):
        s = 0.5 * self._speed
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if key == kb.Key.shift or key == kb.Key.shift_r:
            self._speed = 2
        elif ch in ('w', 'W'):
            self._msg.axes[1] = s
        elif ch in ('s', 'S'):
            self._msg.axes[1] = -s
        elif ch in ('a', 'A'):
            self._msg.axes[0] = s
        elif ch in ('d', 'D'):
            self._msg.axes[0] = -s
        elif ch == '1':
            self._msg.buttons[5] = 1
        elif ch == '2':
            self._msg.buttons[0] = 1
        elif key == kb.Key.backspace:
            self._msg.buttons[4] = 1
        elif key == kb.Key.up:
            self._msg.axes[4] = s
        elif key == kb.Key.down:
            self._msg.axes[4] = -s
        elif key == kb.Key.left:
            self._msg.axes[3] = s
        elif key == kb.Key.right:
            self._msg.axes[3] = -s
        elif ch == '0':
            self._msg.axes[7] = 1.0
        elif ch == '9':
            self._msg.axes[7] = -1.0
        elif ch == '8':
            self._msg.axes[6] = 1.0
        elif ch == '7':
            self._msg.axes[6] = -1.0

    def _on_release(self, key):
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if key == kb.Key.shift or key == kb.Key.shift_r:
            self._speed = 1
        elif ch in ('w', 'W', 's', 'S'):
            self._msg.axes[1] = 0.0
        elif ch in ('a', 'A', 'd', 'D'):
            self._msg.axes[0] = 0.0
        elif ch == '1':
            self._msg.buttons[5] = 0
        elif ch == '2':
            self._msg.buttons[0] = 0
        elif key == kb.Key.backspace:
            self._msg.buttons[4] = 0
        elif key in (kb.Key.up, kb.Key.down):
            self._msg.axes[4] = 0.0
        elif key in (kb.Key.left, kb.Key.right):
            self._msg.axes[3] = 0.0
        elif ch in ('0', '9'):
            self._msg.axes[7] = 0.0
        elif ch in ('8', '7'):
            self._msg.axes[6] = 0.0

    def stop(self):
        self._listener.stop()


def main():
    rclpy.init()
    node = rclpy.create_node('keyboard_input_listener')

    for line in HELP.strip().split('\n'):
        node.get_logger().info(line)

    joy = KeyboardJoy(node)

    def _shutdown(*_):
        joy.stop()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        rclpy.spin(node)
    finally:
        joy.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
