#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64MultiArray, String

from jax_control.Command import Command
from jax_control.Config import Configuration, Leg_linkage
from jax_control.Controller import Controller
from jax_control.Kinematics import four_legs_inverse_kinematics
from jax_control.State import BehaviorState, RobotMode, State
from jax_msgs.msg import JointSpace, TaskSpace


def parse_driver_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('is_sim', nargs='?', type=int, default=0)
    parser.add_argument('is_physical', nargs='?', type=int, default=1)
    parser.add_argument('use_imu', nargs='?', type=int, default=0)
    return parser.parse_args(remove_ros_args(argv)[1:])


class JaxDriver:
    def __init__(self, is_sim, is_physical, use_imu, node: Node):
        self.node = node
        self.message_rate = 50
        self._loop_period = 1.0 / self.message_rate

        self.is_sim = is_sim
        self.is_physical = is_physical
        self.use_imu = use_imu

        # ✅ MODE ALIAS MAP (THIS WAS YOUR ISSUE AREA)
        self._mode_map = {
            'rest': RobotMode.REST,
            'trot': RobotMode.TROT,
            'walk': RobotMode.TROT,    # compatibility
            'sit': RobotMode.SIT,
            'lay': RobotMode.LAY,
        }

        self.latest_cmd_vel = Twist()
        self.latest_mode = RobotMode.REST
        self.rest_recenter_pending = False
        self._imu_sub = None

        self.desired_mode_topic = '/jax_mode'
        self.filtered_cmd_vel_topic = '/cmd_vel'

        self.current_mode_topic = node.declare_parameter(
            'current_mode_topic', '/jax/current_mode'
        ).value

        self._current_mode_pub = node.create_publisher(String, self.current_mode_topic, 10)

        self._declare_behavior_pose_parameters()

        # ---------------- SUBS ----------------
        self.mode_sub = node.create_subscription(
            String, self.desired_mode_topic, self.update_robot_mode, 10
        )

        self.cmd_vel_sub = node.create_subscription(
            Twist, self.filtered_cmd_vel_topic, self.update_cmd_vel, 10
        )

        self.estop_status_sub = node.create_subscription(
            Bool, '/emergency_stop_status', self.update_emergency_stop_status, 10
        )

        # ---------------- SIM OUTPUT ----------------
        self._sim_leg_cmds_pub = None
        if self.is_sim:
            # 👉 SEND TROT OUTPUT TO RAW TOPIC (mode manager will take over final output)
            gz_leg_topic = '/jax/trot_joint_commands'

            self._sim_leg_cmds_pub = node.create_publisher(
                Float64MultiArray, gz_leg_topic, 10
            )

        # ---------------- CONFIG ----------------
        self.config = Configuration()
        self._apply_behavior_pose_parameters()

                # REST height slider mapping
        # Slider input is expected to be:
        #   -1.0 = bottom
        #    0.0 = center
        #   +1.0 = top
        self.rest_height_center = node.declare_parameter(
            'rest_height_center',
            -0.17531
        ).value

        self.rest_height_min = node.declare_parameter(
            'rest_height_min',
            self.rest_height_center - 0.06
        ).value

        self.rest_height_max = node.declare_parameter(
            'rest_height_max',
            self.rest_height_center + 0.06
        ).value

        self.rest_max_roll = float(node.declare_parameter('rest_max_roll', 0.20).value)
        self.rest_max_pitch = float(node.declare_parameter('rest_max_pitch', 0.20).value)

        # Optional speed slider for TROT.
        # Axis options: linear.x, linear.y, linear.z, angular.x, angular.y, angular.z, none
        self.trot_speed_slider_axis = str(
            node.declare_parameter('trot_speed_slider_axis', 'angular.x').value
        ).strip().lower()
        self.trot_speed_min_scale = float(
            node.declare_parameter('trot_speed_min_scale', 0.20).value
        )
        self.trot_speed_max_scale = float(
            node.declare_parameter('trot_speed_max_scale', 1.00).value
        )
        self.trot_speed_slider_deadband = float(
            node.declare_parameter('trot_speed_slider_deadband', 0.03).value
        )

        if self.trot_speed_min_scale > self.trot_speed_max_scale:
            self.trot_speed_min_scale, self.trot_speed_max_scale = (
                self.trot_speed_max_scale,
                self.trot_speed_min_scale,
            )

        # ---------------- CONTROLLER ----------------
        self.controller = Controller(self.config, four_legs_inverse_kinematics, node)
        self.state = State()

        self.state.robot_mode = RobotMode.REST
        self.state.behavior_state = BehaviorState.REST

        if self.use_imu and self.is_sim:
            imu_topic = node.declare_parameter('sim_imu_topic', '/jax/imu').value
            self._imu_sub = node.create_subscription(Imu, imu_topic, self.update_imu, 10)
            self.node.get_logger().info(f'IMU enabled: subscribing to {imu_topic}')

        self.node.get_logger().info(
            f"TROT speed slider axis={self.trot_speed_slider_axis}, "
            f"scale=[{self.trot_speed_min_scale:.2f}, {self.trot_speed_max_scale:.2f}]"
        )

        self.node.get_logger().info("✅ Jax mode driver ready")

        self.publish_current_mode()

    # ---------------- PARAMETERS ----------------

    def _declare_behavior_pose_parameters(self):
        defaults = {
            'sit_x_offsets': [-0.03, -0.03, 0.09, 0.09],
            'sit_y_offsets': [0.0, 0.0, 0.0, 0.0],
            'sit_z_offsets': [-0.18, -0.18, -0.18, -0.18],
            'lay_x_offsets': [-0.015, -0.015, 0.035, 0.035],
            'lay_y_offsets': [0.0, 0.0, 0.0, 0.0],
            'lay_z_offsets': [-0.24, -0.24, -0.24, -0.24],
            'rest_x_offsets': [-0.010774, -0.010774, 0.0, 0.0],
            'rest_y_offsets': [0.0, 0.0, 0.0, 0.0],
        }

        for name, default in defaults.items():
            self.node.declare_parameter(name, default)

    def _param_vec(self, name):
        return np.array(self.node.get_parameter(name).value, dtype=float)

    def _apply_behavior_pose_parameters(self):
        self.config.set_behavior_pose_offsets(
            sit_x=self._param_vec('sit_x_offsets'),
            sit_y=self._param_vec('sit_y_offsets'),
            sit_z=self._param_vec('sit_z_offsets'),
            lay_x=self._param_vec('lay_x_offsets'),
            lay_y=self._param_vec('lay_y_offsets'),
            lay_z=self._param_vec('lay_z_offsets'),
            rest_x=self._param_vec('rest_x_offsets'),
            rest_y=self._param_vec('rest_y_offsets'),
        )

    # ---------------- CALLBACKS ----------------

    def update_robot_mode(self, msg: String):
        mode = msg.data.strip().lower()

        if mode == 'stand':
            # Stand is handled by jax_mode_manager static pose publishing.
            # Keep driver mode unchanged so stand remains distinct from REST.
            self.node.get_logger().info("Mode -> stand (handled by mode_manager)")
            return

        if mode not in self._mode_map:
            self.node.get_logger().warn(f"Unknown mode: {mode}")
            return

        self.latest_mode = self._mode_map[mode]

        if self.latest_mode == RobotMode.REST:
            # Recenter height each time REST is explicitly requested.
            self.rest_recenter_pending = True
            # Avoid stale non-zero slider commands immediately undoing the recenter.
            self.latest_cmd_vel.linear.z = 0.0

        self.node.get_logger().info(f"Mode -> {self.latest_mode.value}")
        self.publish_current_mode()

    def publish_current_mode(self):
        msg = String()
        msg.data = self.latest_mode.value
        self._current_mode_pub.publish(msg)

    def update_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel = msg

    def update_emergency_stop_status(self, msg):
        self.state.currently_estopped = 1 if msg.data else 0

    def update_imu(self, msg: Imu):
        q = msg.orientation

        # Ignore empty/invalid quaternions often sent before sensor initialization.
        if q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0:
            return

        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.state.euler_orientation = [yaw, pitch, roll]

    def _get_twist_axis_value(self, axis_name: str) -> float:
        if axis_name == 'linear.x':
            return float(self.latest_cmd_vel.linear.x)
        if axis_name == 'linear.y':
            return float(self.latest_cmd_vel.linear.y)
        if axis_name == 'linear.z':
            return float(self.latest_cmd_vel.linear.z)
        if axis_name == 'angular.x':
            return float(self.latest_cmd_vel.angular.x)
        if axis_name == 'angular.y':
            return float(self.latest_cmd_vel.angular.y)
        if axis_name == 'angular.z':
            return float(self.latest_cmd_vel.angular.z)
        return 0.0

    def _get_trot_speed_scale(self) -> float:
        if self.trot_speed_slider_axis == 'none':
            return 1.0

        raw = float(np.clip(self._get_twist_axis_value(self.trot_speed_slider_axis), -1.0, 1.0))
        if abs(raw) < self.trot_speed_slider_deadband:
            raw = 0.0

        # Map [-1, 1] -> [0, 1], then to [min_scale, max_scale]
        normalized = 0.5 * (raw + 1.0)
        scale = self.trot_speed_min_scale + normalized * (
            self.trot_speed_max_scale - self.trot_speed_min_scale
        )
        return float(np.clip(scale, self.trot_speed_min_scale, self.trot_speed_max_scale))

    # ---------------- CORE ----------------

    def build_command(self):
        command = Command()

        command.horizontal_velocity = np.array([
            self.latest_cmd_vel.linear.x,
            self.latest_cmd_vel.linear.y
        ])
        command.yaw_rate = self.latest_cmd_vel.angular.z

        if self.latest_mode == RobotMode.TROT:
            speed_scale = self._get_trot_speed_scale()
            command.horizontal_velocity *= speed_scale
            command.yaw_rate *= speed_scale
            self.state.speed_factor = speed_scale
        else:
            self.state.speed_factor = 1.0

        # Preserve current state by default
        command.height = self.state.height
        command.pitch = self.state.pitch
        command.roll = self.state.roll

        # Height slider is normalized:
        #   -1.0 = bottom
        #    0.0 = center
        #   +1.0 = top
        #
        # Use it in BOTH REST and TROT so gait can adapt like old Jax did.
        if self.latest_mode in (RobotMode.REST, RobotMode.TROT):
            slider = float(np.clip(self.latest_cmd_vel.linear.z, -1.0, 1.0))

            if slider >= 0.0:
                command.height = self.rest_height_center + slider * (
                    self.rest_height_max - self.rest_height_center
                )
            else:
                command.height = self.rest_height_center + (-slider) * (
                    self.rest_height_min - self.rest_height_center
                )

        if self.latest_mode == RobotMode.REST and self.rest_recenter_pending:
            command.height = float(self.rest_height_center)
            self.rest_recenter_pending = False

        # In REST, map cmd_vel inputs to body attitude while feet remain planted.
        # This supports both angular-axis and linear-axis teleop layouts.
        if self.latest_mode == RobotMode.REST:
            roll_input = np.clip(
                self.latest_cmd_vel.angular.x + self.latest_cmd_vel.linear.y,
                -1.0,
                1.0,
            )
            pitch_input = np.clip(
                self.latest_cmd_vel.angular.y + self.latest_cmd_vel.linear.x,
                -1.0,
                1.0,
            )
            command.roll = roll_input * self.rest_max_roll
            command.pitch = pitch_input * self.rest_max_pitch

        return command

    def apply_mode(self):
        if self.latest_mode == RobotMode.TROT:
            # Transitioning from static poses directly into gait can leave the
            # planner in a poor foot-state seed. Re-enter REST for one cycle,
            # then switch to TROT on the next loop.
            if self.state.behavior_state in (BehaviorState.SIT, BehaviorState.LAY):
                self.state.behavior_state = BehaviorState.REST
                self.rest_recenter_pending = True
            else:
                self.state.behavior_state = BehaviorState.TROT
        elif self.latest_mode == RobotMode.REST:
            self.state.behavior_state = BehaviorState.REST
        elif self.latest_mode == RobotMode.SIT:
            self.state.behavior_state = BehaviorState.SIT
        elif self.latest_mode == RobotMode.LAY:
            self.state.behavior_state = BehaviorState.LAY

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)

            self.apply_mode()

            command = self.build_command()

            self.controller.run(self.state, command)

            if self.is_sim:
                self.publish_joints(self.state.joint_angles)

            time.sleep(self._loop_period)

    def publish_joints(self, joint_angles):
        if not self._sim_leg_cmds_pub:
            return

        msg = Float64MultiArray()

        msg.data = [
            float(joint_angles[0, 0]), float(joint_angles[1, 0]), float(joint_angles[2, 0]),
            float(joint_angles[0, 1]), float(joint_angles[1, 1]), float(joint_angles[2, 1]),
            float(joint_angles[0, 2]), float(joint_angles[1, 2]), float(joint_angles[2, 2]),
            float(joint_angles[0, 3]), float(joint_angles[1, 3]), float(joint_angles[2, 3]),
        ]

        self._sim_leg_cmds_pub.publish(msg)


def main(args=None):
    import sys
    argv = sys.argv if args is None else args
    parsed = parse_driver_args(argv)

    rclpy.init(args=args)
    node = rclpy.create_node('jax_driver')

    try:
        driver = JaxDriver(parsed.is_sim, parsed.is_physical, parsed.use_imu, node)
        driver.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()