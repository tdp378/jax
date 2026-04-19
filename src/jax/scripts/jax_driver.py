#!/usr/bin/env python3

import argparse
import math
import os
import time

import numpy as np
import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import BatteryState, Imu, Temperature
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

        self._sent_first_safe_pose = False
        self._latest_mode_manager_joint_angles = None
        self._last_battery_query_time = 0.0

        self._servo_direction_defaults = [
            1, -1, -1,
            1, -1, -1,
            1, -1, -1,
            1, -1, -1,
        ]
        self._servo_order_defaults = list(range(12))
        self._servo_offset_deg_defaults = [0.0] * 12

        self._declare_servo_calibration_parameters()
        self._apply_servo_calibration_parameters()

        self._mode_map = {
            'rest': RobotMode.REST,
            'trot': RobotMode.TROT,
            'walk': RobotMode.TROT,
            'sit': RobotMode.SIT,
            'lay': RobotMode.LAY,
        }

        self.latest_cmd_vel = Twist()
        self._last_cmd_vel_time = None
        self.latest_mode = RobotMode.REST
        self.rest_recenter_pending = False
        self._imu_sub = None

        self.desired_mode_topic = '/jax_mode'
        self.filtered_cmd_vel_topic = '/cmd_vel'
        self.raw_joint_topic = '/jax/trot_joint_commands'
        self.final_joint_topic = '/leg_joint_position_controller/commands'

        self.current_mode_topic = node.declare_parameter(
            'current_mode_topic', '/jax/current_mode'
        ).value
        self._battery_query_period_s = float(
            node.declare_parameter('battery_query_period_s', 0.10).value
        )
        self._cmd_vel_timeout_s = max(
            float(node.declare_parameter('cmd_vel_timeout_s', 0.25).value),
            self._loop_period,
        )
        self._rest_tilt_deadband = max(
            float(node.declare_parameter('rest_tilt_deadband', 0.08).value),
            0.0,
        )
        self._height_slider_deadband = max(
            float(node.declare_parameter('height_slider_deadband', 0.05).value),
            0.0,
        )
        self._battery_voltage_scale = float(
            node.declare_parameter('battery_voltage_scale', 0.78).value
        )
        self._battery_voltage_offset = float(
            node.declare_parameter('battery_voltage_offset', 0.0).value
        )
        self._cpu_temp_enabled = bool(
            node.declare_parameter('cpu_temp_enabled', bool(self.is_physical)).value
        )
        self._cpu_temp_topic = str(
            node.declare_parameter('cpu_temp_topic', '/cpu_temperature').value
        )
        self._cpu_temp_frame_id = str(
            node.declare_parameter('cpu_temp_frame_id', 'cpu').value
        )
        self._cpu_temp_source_path = str(
            node.declare_parameter(
                'cpu_temp_source_path',
                '/sys/class/thermal/thermal_zone0/temp',
            ).value
        )
        self._cpu_temp_publish_period_s = max(
            float(node.declare_parameter('cpu_temp_publish_period_s', 1.0).value),
            self._loop_period,
        )
        self._cpu_temp_variance_c = max(
            float(node.declare_parameter('cpu_temp_variance_c', 0.0).value),
            0.0,
        )
        self._last_cpu_temp_publish_time = 0.0
        self._cpu_temp_read_error_logged = False

        self._current_mode_pub = node.create_publisher(String, self.current_mode_topic, 10)
        self._battery_pub = node.create_publisher(BatteryState, '/jax/battery', 10)
        self._cpu_temp_pub = None
        if self._cpu_temp_enabled:
            self._cpu_temp_pub = node.create_publisher(Temperature, self._cpu_temp_topic, 10)

        self.config = Configuration()
        self._declare_behavior_pose_parameters()
        self._declare_global_stance_parameters()

        self.mode_sub = node.create_subscription(
            String, self.desired_mode_topic, self.update_robot_mode, 10
        )
        self.cmd_vel_sub = node.create_subscription(
            Twist, self.filtered_cmd_vel_topic, self.update_cmd_vel, 10
        )
        self.estop_status_sub = node.create_subscription(
            Bool, '/emergency_stop_status', self.update_emergency_stop_status, 10
        )

        if self.is_physical:
            self.final_joint_cmd_sub = node.create_subscription(
                Float64MultiArray,
                self.final_joint_topic,
                self.update_mode_manager_joint_command,
                10,
            )
        else:
            self.final_joint_cmd_sub = None

        self._raw_leg_cmds_pub = node.create_publisher(
            Float64MultiArray, self.raw_joint_topic, 10
        )

        self._apply_global_stance_parameters()
        self._apply_behavior_pose_parameters()

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

        self.controller = Controller(self.config, four_legs_inverse_kinematics, node)
        self.state = State()
        self.state.robot_mode = RobotMode.REST
        self.state.behavior_state = BehaviorState.REST

        self.serial_port = None
        if self.is_physical:
            self.serial_port = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.02)
            self.node.get_logger().info('Serial connection to Arduino established.')

        if self.use_imu and self.is_sim:
            imu_topic = node.declare_parameter('sim_imu_topic', '/jax/imu').value
            self._imu_sub = node.create_subscription(Imu, imu_topic, self.update_imu, 10)
            self.node.get_logger().info(f'IMU enabled: subscribing to {imu_topic}')

        self.node.get_logger().info(
            f'TROT speed slider axis={self.trot_speed_slider_axis}, '
            f'scale=[{self.trot_speed_min_scale:.2f}, {self.trot_speed_max_scale:.2f}]'
        )
        if self._cpu_temp_enabled:
            self.node.get_logger().info(
                f'CPU temp telemetry active on {self._cpu_temp_topic} '
                f'from {self._cpu_temp_source_path}'
            )
        self.node.get_logger().info('Jax Arduino mode driver ready')
        self.publish_current_mode()

    def _declare_servo_calibration_parameters(self):
        self.node.declare_parameter('servo_direction', self._servo_direction_defaults)
        self.node.declare_parameter('servo_order', self._servo_order_defaults)
        self.node.declare_parameter('servo_offset_deg', self._servo_offset_deg_defaults)

    def _get_fixed_length_param(self, name, expected_len, cast, default):
        values = list(self.node.get_parameter(name).value)
        if len(values) != expected_len:
            self.node.get_logger().warn(
                f"Parameter '{name}' expected {expected_len} values, got {len(values)}. Using defaults."
            )
            return list(default)
        return [cast(value) for value in values]

    def _apply_servo_calibration_parameters(self):
        self.servo_direction = self._get_fixed_length_param(
            'servo_direction', 12, int, self._servo_direction_defaults
        )
        self.servo_order = self._get_fixed_length_param(
            'servo_order', 12, int, self._servo_order_defaults
        )
        self.servo_offset_deg = self._get_fixed_length_param(
            'servo_offset_deg', 12, float, self._servo_offset_deg_defaults
        )

    def _declare_global_stance_parameters(self):
        defaults = {
            'default_stance_delta_x': float(self.config.delta_x),
            'default_stance_delta_y': float(self.config.delta_y),
            'front_leg_x_shift': float(self.config.front_leg_x_shift),
            'rear_leg_x_shift': float(self.config.rear_leg_x_shift),
            'default_z_ref': float(self.config.default_z_ref),
        }
        for name, default in defaults.items():
            self.node.declare_parameter(name, default)

    def _apply_global_stance_parameters(self):
        self.config.delta_x = float(self.node.get_parameter('default_stance_delta_x').value)
        self.config.delta_y = float(self.node.get_parameter('default_stance_delta_y').value)
        self.config.front_leg_x_shift = float(self.node.get_parameter('front_leg_x_shift').value)
        self.config.rear_leg_x_shift = float(self.node.get_parameter('rear_leg_x_shift').value)
        self.config.default_z_ref = float(self.node.get_parameter('default_z_ref').value)

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

    def update_robot_mode(self, msg: String):
        mode = msg.data.strip().lower()
        if mode == 'stand':
            self.node.get_logger().info('Mode -> stand (handled by mode_manager)')
            return
        if mode not in self._mode_map:
            self.node.get_logger().warn(f'Unknown mode: {mode}')
            return

        self.latest_mode = self._mode_map[mode]
        if self.latest_mode == RobotMode.REST:
            self.rest_recenter_pending = True
            self.latest_cmd_vel.linear.z = 0.0

        self.node.get_logger().info(f'Mode -> {self.latest_mode.value}')
        self.publish_current_mode()

    def publish_current_mode(self):
        msg = String()
        msg.data = self.latest_mode.value
        self._current_mode_pub.publish(msg)

    def _zero_twist_command(self):
        self.latest_cmd_vel = Twist()

    def _apply_deadband(self, value: float, deadband: float) -> float:
        if abs(value) < deadband:
            return 0.0
        return value

    def _get_fresh_cmd_vel(self) -> Twist:
        if self._last_cmd_vel_time is None:
            return Twist()

        age = time.monotonic() - self._last_cmd_vel_time
        if age > self._cmd_vel_timeout_s:
            self._zero_twist_command()
            self._last_cmd_vel_time = None
        return self.latest_cmd_vel

    def update_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel = msg
        self._last_cmd_vel_time = time.monotonic()

    def update_emergency_stop_status(self, msg):
        self.state.currently_estopped = 1 if msg.data else 0

    def update_mode_manager_joint_command(self, msg: Float64MultiArray):
        if len(msg.data) != 12:
            self.node.get_logger().warn(
                f'Ignoring final joint command: expected 12 values, got {len(msg.data)}'
            )
            return
        self._latest_mode_manager_joint_angles = np.array(msg.data, dtype=float).reshape((4, 3)).T

    def update_imu(self, msg: Imu):
        q = msg.orientation
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

        normalized = 0.5 * (raw + 1.0)
        scale = self.trot_speed_min_scale + normalized * (
            self.trot_speed_max_scale - self.trot_speed_min_scale
        )
        return float(np.clip(scale, self.trot_speed_min_scale, self.trot_speed_max_scale))

    def build_command(self):
        cmd_vel = self._get_fresh_cmd_vel()

        command = Command()
        command.horizontal_velocity = np.array([
            cmd_vel.linear.x,
            cmd_vel.linear.y,
        ])
        command.yaw_rate = cmd_vel.angular.z

        if self.latest_mode == RobotMode.TROT:
            speed_scale = self._get_trot_speed_scale()
            command.horizontal_velocity *= speed_scale
            command.yaw_rate *= speed_scale
            self.state.speed_factor = speed_scale
        else:
            self.state.speed_factor = 1.0

        command.height = self.state.height
        command.pitch = self.state.pitch
        command.roll = self.state.roll

        if self.latest_mode in (RobotMode.REST, RobotMode.TROT):
            slider = float(np.clip(cmd_vel.linear.z, -1.0, 1.0))
            slider = self._apply_deadband(slider, self._height_slider_deadband)
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

        if self.latest_mode == RobotMode.REST:
            roll_input = float(np.clip(
                cmd_vel.angular.x + cmd_vel.linear.y,
                -1.0,
                1.0,
            ))
            pitch_input = float(np.clip(
                cmd_vel.angular.y + cmd_vel.linear.x,
                -1.0,
                1.0,
            ))

            roll_input = self._apply_deadband(roll_input, self._rest_tilt_deadband)
            pitch_input = self._apply_deadband(pitch_input, self._rest_tilt_deadband)

            command.roll = roll_input * self.rest_max_roll
            command.pitch = pitch_input * self.rest_max_pitch

        return command

    def apply_mode(self):
        if self.latest_mode == RobotMode.TROT:
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

    def _request_battery_voltage(self):
        if not self.serial_port or not self.serial_port.is_open:
            return
        now = time.monotonic()
        if (now - self._last_battery_query_time) < self._battery_query_period_s:
            return
        self._last_battery_query_time = now
        self.serial_port.write(b'BAT?\n')

    def _publish_battery_voltage(self, raw_voltage: float):
        batt = BatteryState()
        batt.header.stamp = self.node.get_clock().now().to_msg()
        batt.voltage = (raw_voltage * self._battery_voltage_scale) + self._battery_voltage_offset
        batt.present = True
        batt.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        batt.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        batt.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        batt.percentage = float(np.clip((batt.voltage - 13.6) / (16.8 - 13.6), 0.0, 1.0))
        self._battery_pub.publish(batt)

    def _read_cpu_temperature_c(self):
        if not os.path.exists(self._cpu_temp_source_path):
            raise FileNotFoundError(self._cpu_temp_source_path)

        with open(self._cpu_temp_source_path, 'r', encoding='utf-8') as temp_file:
            raw_value = temp_file.read().strip()

        temp_c = float(raw_value)
        if temp_c > 1000.0:
            temp_c /= 1000.0
        return temp_c

    def _publish_cpu_temperature(self):
        if not self._cpu_temp_pub:
            return

        now = time.monotonic()
        if (now - self._last_cpu_temp_publish_time) < self._cpu_temp_publish_period_s:
            return
        self._last_cpu_temp_publish_time = now

        try:
            temp_c = self._read_cpu_temperature_c()
            self._cpu_temp_read_error_logged = False
        except (OSError, ValueError) as exc:
            if not self._cpu_temp_read_error_logged:
                self.node.get_logger().warn(f'CPU temp read failed: {exc}')
                self._cpu_temp_read_error_logged = True
            return

        msg = Temperature()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._cpu_temp_frame_id
        msg.temperature = temp_c
        msg.variance = self._cpu_temp_variance_c * self._cpu_temp_variance_c
        self._cpu_temp_pub.publish(msg)

    def _drain_serial_feedback(self):
        if not self.serial_port or not self.serial_port.is_open:
            return
        try:
            while self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if not line or line in {'OK', 'ERR', 'JAX_SERVO_READY'}:
                    continue
                raw_voltage = None
                if line.startswith('VOLT:'):
                    payload = line.split(':', 1)[1]
                    try:
                        raw_voltage = float(payload)
                    except ValueError:
                        self.node.get_logger().warn(f'Invalid battery response: {line}')
                        continue
                else:
                    try:
                        raw_voltage = float(line)
                    except ValueError:
                        continue

                self._publish_battery_voltage(raw_voltage)
        except serial.SerialException as exc:
            self.node.get_logger().warn(f'Serial feedback error: {exc}')

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self.apply_mode()

            command = self.build_command()
            self.controller.run(self.state, command)
            raw_joint_angles = self.state.joint_angles
            self.publish_joints(raw_joint_angles)

            if self.is_physical:
                joint_angles = raw_joint_angles
                if self._latest_mode_manager_joint_angles is not None:
                    joint_angles = self._latest_mode_manager_joint_angles

                if not self._sent_first_safe_pose:
                    self.send_joint_angles_to_arduino(np.zeros((3, 4)))
                    self._sent_first_safe_pose = True
                else:
                    self.send_joint_angles_to_arduino(joint_angles)

                self._request_battery_voltage()
                self._drain_serial_feedback()
                self._publish_cpu_temperature()

            time.sleep(self._loop_period)

    def send_joint_angles_to_arduino(self, joint_angles):
        if not self.serial_port or not self.serial_port.is_open:
            return
        flat_angles = [float(joint_angles[i, j]) for j in range(4) for i in range(3)]
        ordered_angles = [flat_angles[i] * self.servo_direction[i] for i in self.servo_order]
        deg_angles = [
            max(0, min(180, math.degrees(angle) + 90 + self.servo_offset_deg[index]))
            for index, angle in enumerate(ordered_angles)
        ]
        cmd_str = ','.join(f'{a:.2f}' for a in deg_angles) + '\n'
        self.serial_port.write(cmd_str.encode('utf-8'))

    def publish_joints(self, joint_angles):
        if not self._raw_leg_cmds_pub:
            return

        msg = Float64MultiArray()
        msg.data = [
            float(joint_angles[0, 0]), float(joint_angles[1, 0]), float(joint_angles[2, 0]),
            float(joint_angles[0, 1]), float(joint_angles[1, 1]), float(joint_angles[2, 1]),
            float(joint_angles[0, 2]), float(joint_angles[1, 2]), float(joint_angles[2, 2]),
            float(joint_angles[0, 3]), float(joint_angles[1, 3]), float(joint_angles[2, 3]),
        ]
        self._raw_leg_cmds_pub.publish(msg)


def main(args=None):
    import sys

    argv = sys.argv if args is None else args
    parsed = parse_driver_args(argv)

    rclpy.init(args=args)
    node = rclpy.create_node('jax_driver')
    driver = None

    try:
        driver = JaxDriver(parsed.is_sim, parsed.is_physical, parsed.use_imu, node)
        driver.run()
    except KeyboardInterrupt:
        pass
    finally:
        if driver and driver.serial_port and driver.serial_port.is_open:
            driver.serial_port.close()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
