#!/usr/bin/env python3

import math
import os
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState

# ===== SIM TOPICS (HARDCODED) =====
MODE_TOPIC = '/jax_mode'
CURRENT_MODE_TOPIC = '/jax/current_mode'
RAW_TROT_TOPIC = '/jax/trot_joint_commands'
OUTPUT_TOPIC = '/leg_joint_position_controller/commands'
JOINT_STATES_TOPIC = '/joint_states'
TRANSITION_STEPS = 50


class JaxModeManager(Node):
    def __init__(self):
        super().__init__('mode_manager')

        pkg_share = get_package_share_directory('jax_behaviors')
        poses_file = os.path.join(pkg_share, 'config', 'poses.yaml')

        self.joint_names, self.poses = self.load_poses(poses_file)

        self.static_modes = { 'sit', 'lay', 'stand' }
        self.dynamic_modes = {'trot', 'rest'}
        self.valid_modes = self.static_modes | self.dynamic_modes

        if 'rest' not in self.poses:
            raise RuntimeError("poses.yaml must define a 'rest' pose")

        self.transition_steps = TRANSITION_STEPS
        self.current_mode = 'rest'
        self.current_pose = list(self.poses['rest'])
        self.target_pose = list(self.poses['rest'])
        self.step_count = self.transition_steps
        self.pending_dynamic_mode = None
        self.latest_driver_mode = 'rest'

        self.latest_joint_positions = None
        self.latest_walk_command = None

        self.mode_sub = self.create_subscription(
            String,
            MODE_TOPIC,
            self.mode_callback,
            10
        )

        self.walk_sub = self.create_subscription(
            Float64MultiArray,
            RAW_TROT_TOPIC,
            self.walk_callback,
            10
        )

        self.current_mode_sub = self.create_subscription(
            String,
            CURRENT_MODE_TOPIC,
            self.current_mode_callback,
            10
        )

        self.js_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_state_callback,
            50
        )

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            OUTPUT_TOPIC,
            10
        )

        self.timer = self.create_timer(0.02, self.interpolation_loop)

        self.get_logger().info(f"Loaded poses from: {poses_file}")
        self.get_logger().info(f"Mode topic: {MODE_TOPIC}")
        self.get_logger().info(f"Current mode topic: {CURRENT_MODE_TOPIC}")
        self.get_logger().info(f"Raw trot topic: {RAW_TROT_TOPIC}")
        self.get_logger().info(f"Output topic: {OUTPUT_TOPIC}")
        self.get_logger().info("Jax Mode Manager ready.")

    def load_poses(self, poses_file):
        with open(poses_file, 'r') as f:
            data = yaml.safe_load(f)

        joint_names = data.get('joint_names', [])
        poses = data.get('poses', {})

        if not joint_names:
            raise RuntimeError("poses.yaml missing 'joint_names'")
        if not poses:
            raise RuntimeError("poses.yaml missing 'poses'")

        joint_count = len(joint_names)
        for pose_name, angles in poses.items():
            if len(angles) != joint_count:
                raise RuntimeError(
                    f"Pose '{pose_name}' length {len(angles)} does not match joint_names length {joint_count}"
                )

        return joint_names, poses

    def mode_callback(self, msg):
        mode = msg.data.lower().strip()

        if mode not in self.valid_modes:
            self.get_logger().warn(
                f"Mode '{mode}' is not valid. Valid modes: {sorted(self.valid_modes)}"
            )
            return

        if mode == self.current_mode:
            return

        if mode == 'trot' and self.current_mode not in {'stand', 'rest'}:
            self.get_logger().warn(
                f"Blocked transition: {self.current_mode} -> trot. Enter 'stand' or 'rest' before 'trot'."
            )
            return

        if mode in self.dynamic_modes and self.current_mode in self.static_modes:
            snap = self.get_best_current_pose()
            if snap is not None:
                self.current_pose = snap
                self.get_logger().info("Using live pose snapshot for smooth transition.")
            else:
                self.get_logger().warn("No valid pose snapshot available; using last known pose.")

            old_mode = self.current_mode
            self.pending_dynamic_mode = mode
            self.target_pose = None
            self.step_count = 0
            self.get_logger().info(
                f"Transitioning smoothly to dynamic mode: {old_mode} -> {mode}"
            )
            return

        old_mode = self.current_mode
        self.current_mode = mode
        self.get_logger().info(f"Mode change: {old_mode} -> {mode}")

        if mode in self.dynamic_modes:
            self.step_count = self.transition_steps
            return

        if mode in self.static_modes:
            snap = self.get_best_current_pose()
            if snap is not None:
                self.current_pose = snap
                self.get_logger().info("Using live pose snapshot for smooth transition.")
            else:
                self.get_logger().warn("No valid pose snapshot available; using last known pose.")

            if mode not in self.poses:
                self.get_logger().warn(f"No pose configured for static mode '{mode}'")
                return

            self.target_pose = list(self.poses[mode])
            self.step_count = 0
            self.get_logger().info(f"Transitioning smoothly to: {mode}")

    def joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return

        name_to_pos = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                name_to_pos[name] = msg.position[i]

        if all(name in name_to_pos for name in self.joint_names):
            ordered = [name_to_pos[name] for name in self.joint_names]
            if not any(math.isnan(x) or math.isinf(x) for x in ordered):
                self.latest_joint_positions = ordered

    def current_mode_callback(self, msg):
        self.latest_driver_mode = msg.data.lower().strip()

    def walk_callback(self, msg):
        if len(msg.data) == len(self.joint_names):
            self.latest_walk_command = list(msg.data)

            if (
                self.pending_dynamic_mode is not None
                and self.target_pose is None
                and self.latest_driver_mode == self.pending_dynamic_mode
            ):
                self.target_pose = list(msg.data)
                self.get_logger().info(
                    f"Captured live target for transition to: {self.pending_dynamic_mode}"
                )

        if self.current_mode in self.dynamic_modes and self.pending_dynamic_mode is None:
            self.cmd_pub.publish(msg)

    def get_best_current_pose(self):
        if self.latest_joint_positions is not None:
            return list(self.latest_joint_positions)
        if self.current_mode in self.static_modes and self.current_pose is not None:
            return list(self.current_pose)
        if self.latest_walk_command is not None:
            return list(self.latest_walk_command)
        if self.current_pose is not None:
            return list(self.current_pose)
        return None

    def interpolation_loop(self):
        if self.current_mode not in self.static_modes and self.pending_dynamic_mode is None:
            return

        if self.target_pose is None:
            return

        if self.step_count < self.transition_steps:
            self.step_count += 1
            alpha = self.step_count / float(self.transition_steps)

            new_angles = [
                self.current_pose[i] + (self.target_pose[i] - self.current_pose[i]) * alpha
                for i in range(len(self.joint_names))
            ]

            self.publish_angles(new_angles)

            if self.step_count == self.transition_steps:
                self.current_pose = list(self.target_pose)
                if self.pending_dynamic_mode is not None:
                    old_mode = self.current_mode
                    self.current_mode = self.pending_dynamic_mode
                    self.pending_dynamic_mode = None
                    self.get_logger().info(
                        f"Mode change: {old_mode} -> {self.current_mode}"
                    )

    def publish_angles(self, angles):
        if len(angles) != len(self.joint_names):
            self.get_logger().error("Angle vector length does not match joint_names length.")
            return

        msg = Float64MultiArray()
        msg.data = list(angles)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = JaxModeManager()
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