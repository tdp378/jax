#!/usr/bin/env python3

import argparse
import math
import os
import time

from typing import Optional

import numpy as np
import rclpy
try:
    import serial
except ImportError:
    serial = None

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    SMBus = None
    i2c_msg = None

try:
    import smbus
except ImportError:
    smbus = None
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

        self._latest_mode_manager_joint_angles = None
        self._last_battery_query_time = 0.0
        self._last_imu_query_time = 0.0
        self._transport = 'serial'
        self._i2c_bus = None
        self._i2c_address = 0x08
        self._i2c_read_len = 64
        self._i2c_protocol = 'binary_v1'
        self._warned_i2c_imu_unsupported = False

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
        self.config = Configuration()

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
            float(node.declare_parameter('cmd_vel_timeout_s', self.config.cmd_vel_timeout_s).value),
            self._loop_period,
        )
        self._rest_tilt_deadband = max(
            float(node.declare_parameter('rest_tilt_deadband', self.config.rest_tilt_deadband).value),
            0.0,
        )
        self._height_slider_deadband = max(
            float(node.declare_parameter('height_slider_deadband', self.config.height_slider_deadband).value),
            0.0,
        )
        self._trot_linear_deadband = max(
            float(node.declare_parameter('trot_linear_deadband', self.config.trot_linear_deadband).value),
            0.0,
        )
        self._trot_yaw_deadband = max(
            float(node.declare_parameter('trot_yaw_deadband', self.config.trot_yaw_deadband).value),
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
        legacy_physical_imu_via_serial = bool(
            node.declare_parameter('physical_imu_via_serial', bool(self.is_physical)).value
        )
        self._physical_imu_via_arduino = bool(
            node.declare_parameter('physical_imu_via_arduino', legacy_physical_imu_via_serial).value
        )
        self._transport = str(
            node.declare_parameter('arduino_transport', 'serial').value
        ).strip().lower()
        if self._transport not in {'serial', 'i2c'}:
            self.node.get_logger().warn(
                f"Unsupported arduino_transport '{self._transport}', defaulting to serial"
            )
            self._transport = 'serial'

        self._serial_port_name = str(
            node.declare_parameter('arduino_serial_port', '/dev/ttyAMA0').value
        )
        self._serial_baud = int(
            node.declare_parameter('arduino_serial_baud', 115200).value
        )
        self._serial_timeout = float(
            node.declare_parameter('arduino_serial_timeout_s', 0.02).value
        )

        self._i2c_bus_id = int(
            node.declare_parameter('arduino_i2c_bus', 1).value
        )
        self._i2c_address = int(
            node.declare_parameter('arduino_i2c_address', 8).value
        )
        self._i2c_read_len = max(
            8,
            int(node.declare_parameter('arduino_i2c_read_len', 64).value),
        )
        self._i2c_protocol = str(
            node.declare_parameter('arduino_i2c_protocol', 'binary_v1').value
        ).strip().lower()

        self._imu_query_period_s = max(
            float(node.declare_parameter('imu_query_period_s', 0.05).value),
            self._loop_period,
        )
        self._imu_frame_id = str(
            node.declare_parameter('imu_frame_id', 'imu_link').value
        )
        self._imu_orientation_variance = max(
            float(node.declare_parameter('imu_orientation_variance', 0.02).value),
            0.0,
        )
        self._imu_angular_velocity_variance = max(
            float(node.declare_parameter('imu_angular_velocity_variance', 0.03).value),
            0.0,
        )
        self._imu_linear_acceleration_variance = max(
            float(node.declare_parameter('imu_linear_acceleration_variance', 0.2).value),
            0.0,
        )

        self._current_mode_pub = node.create_publisher(String, self.current_mode_topic, 10)
        self._battery_pub = node.create_publisher(BatteryState, '/jax/battery', 10)
        self._cpu_temp_pub = None
        if self._cpu_temp_enabled:
            self._cpu_temp_pub = node.create_publisher(Temperature, self._cpu_temp_topic, 10)
        self._imu_pub = None
        if self.use_imu and self.is_physical and self._physical_imu_via_arduino:
            physical_imu_topic = str(
                node.declare_parameter('physical_imu_topic', '/imu/data').value
            )
            self._imu_pub = node.create_publisher(Imu, physical_imu_topic, 10)

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

        locomotion_defaults = self.config.locomotion_parameter_defaults()

        self.rest_height_center = node.declare_parameter(
            'rest_height_center',
            locomotion_defaults['rest_height_center']
        ).value
        self.rest_height_min = node.declare_parameter(
            'rest_height_min',
            locomotion_defaults['rest_height_min']
        ).value
        self.rest_height_max = node.declare_parameter(
            'rest_height_max',
            locomotion_defaults['rest_height_max']
        ).value

        self.rest_max_roll = float(node.declare_parameter(
            'rest_max_roll', locomotion_defaults['rest_max_roll']
        ).value)
        self.rest_max_pitch = float(node.declare_parameter(
            'rest_max_pitch', locomotion_defaults['rest_max_pitch']
        ).value)

        self.trot_speed_slider_axis = str(
            node.declare_parameter(
                'trot_speed_slider_axis', locomotion_defaults['trot_speed_slider_axis']
            ).value
        ).strip().lower()
        self.trot_speed_min_scale = float(
            node.declare_parameter(
                'trot_speed_min_scale', locomotion_defaults['trot_speed_min_scale']
            ).value
        )
        self.trot_speed_max_scale = float(
            node.declare_parameter(
                'trot_speed_max_scale', locomotion_defaults['trot_speed_max_scale']
            ).value
        )
        self.trot_speed_slider_deadband = float(
            node.declare_parameter(
                'trot_speed_slider_deadband', locomotion_defaults['trot_speed_slider_deadband']
            ).value
        )
        self.trot_start_linear_threshold = float(
            node.declare_parameter(
                'trot_start_linear_threshold', locomotion_defaults['trot_start_linear_threshold']
            ).value
        )
        self.trot_start_yaw_threshold = float(
            node.declare_parameter(
                'trot_start_yaw_threshold', locomotion_defaults['trot_start_yaw_threshold']
            ).value
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
            if self._transport == 'serial':
                if serial is None:
                    raise RuntimeError(
                        'pyserial is required for arduino_transport=serial but is not installed'
                    )
                self.serial_port = serial.Serial(
                    self._serial_port_name,
                    self._serial_baud,
                    timeout=self._serial_timeout,
                )
                self.node.get_logger().info(
                    f'Serial connection to Arduino established on {self._serial_port_name}.'
                )
            elif self._transport == 'i2c':
                if SMBus is not None:
                    self._i2c_bus = SMBus(self._i2c_bus_id)
                elif smbus is not None:
                    self._i2c_bus = smbus.SMBus(self._i2c_bus_id)
                else:
                    raise RuntimeError(
                        'I2C transport requires python smbus2 or smbus module, but neither is installed'
                    )
                self.node.get_logger().info(
                    f'I2C connection to Arduino established on bus {self._i2c_bus_id} '
                    f'address 0x{self._i2c_address:02X}.'
                )

        if self.use_imu and not (self.is_physical and self._physical_imu_via_arduino):
            imu_topic = node.declare_parameter(
                'sim_imu_topic' if self.is_sim else 'physical_imu_topic',
                '/jax/imu' if self.is_sim else '/imu/data',
            ).value
            self._imu_sub = node.create_subscription(Imu, imu_topic, self.update_imu, 10)
            self.node.get_logger().info(f'IMU enabled: subscribing to {imu_topic}')
        elif self.use_imu and self.is_physical and self._physical_imu_via_arduino:
            self.node.get_logger().info('IMU enabled: expecting IMU data from Arduino transport feedback')

        self.node.get_logger().info(
            f'TROT speed slider axis={self.trot_speed_slider_axis}, '
            f'scale=[{self.trot_speed_min_scale:.2f}, {self.trot_speed_max_scale:.2f}]'
        )
        if self._cpu_temp_enabled:
            self.node.get_logger().info(
                f'CPU temp telemetry active on {self._cpu_temp_topic} '
                f'from {self._cpu_temp_source_path}'
            )
        self.node.get_logger().info(f'Jax Arduino mode driver ready (transport={self._transport})')
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
        defaults = self.config.global_stance_parameter_defaults()
        for name, default in defaults.items():
            self.node.declare_parameter(name, default)

    def _apply_global_stance_parameters(self):
        self.config.default_stance_delta_x = float(self.node.get_parameter('default_stance_delta_x').value)
        self.config.default_stance_delta_y = float(self.node.get_parameter('default_stance_delta_y').value)
        self.config.front_leg_x_shift = float(self.node.get_parameter('front_leg_x_shift').value)
        self.config.rear_leg_x_shift = float(self.node.get_parameter('rear_leg_x_shift').value)
        self.config.default_z_ref = float(self.node.get_parameter('default_z_ref').value)
        self.config.reverse_step_scale = float(self.node.get_parameter('reverse_step_scale').value)

    def _declare_behavior_pose_parameters(self):
        defaults = self.config.behavior_pose_parameter_defaults()
        for name, default in defaults.items():
            self.node.declare_parameter(name, default)

    def _param_vec(self, name):
        return np.array(self.node.get_parameter(name).value, dtype=float)

    def _apply_behavior_pose_parameters(self):
        self.config.set_behavior_pose_offsets(
            rest_x=self._param_vec('rest_x_offsets'),
            rest_y=self._param_vec('rest_y_offsets'),
        )

    def update_robot_mode(self, msg: String):
        mode = msg.data.strip().lower()
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

    def _get_twist_axis_value(self, axis_name: str, cmd_vel: Optional[Twist] = None) -> float:
        twist = cmd_vel if cmd_vel is not None else self.latest_cmd_vel
        if axis_name == 'linear.x':
            return float(twist.linear.x)
        if axis_name == 'linear.y':
            return float(twist.linear.y)
        if axis_name == 'linear.z':
            return float(twist.linear.z)
        if axis_name == 'angular.x':
            return float(twist.angular.x)
        if axis_name == 'angular.y':
            return float(twist.angular.y)
        if axis_name == 'angular.z':
            return float(twist.angular.z)
        return 0.0

    def _get_trot_speed_scale(self, cmd_vel: Optional[Twist] = None) -> float:
        if self.trot_speed_slider_axis == 'none':
            return 1.0
        raw = float(np.clip(self._get_twist_axis_value(self.trot_speed_slider_axis, cmd_vel), -1.0, 1.0))
        if abs(raw) < self.trot_speed_slider_deadband:
            raw = 0.0

        # Centered slider should correspond to minimum speed; both directions
        # increase speed magnitude.
        normalized = abs(raw)
        scale = self.trot_speed_min_scale + normalized * (
            self.trot_speed_max_scale - self.trot_speed_min_scale
        )
        return float(np.clip(scale, self.trot_speed_min_scale, self.trot_speed_max_scale))

    def _has_trot_motion_intent(self, cmd_vel: Twist) -> bool:
        vx = self._apply_deadband(float(cmd_vel.linear.x), self._trot_linear_deadband)
        vy = self._apply_deadband(float(cmd_vel.linear.y), self._trot_linear_deadband)
        yaw = self._apply_deadband(float(cmd_vel.angular.z), self._trot_yaw_deadband)

        speed_scale = self._get_trot_speed_scale(cmd_vel)
        vx *= speed_scale
        vy *= speed_scale
        yaw *= speed_scale

        linear_mag = float(np.hypot(vx, vy))
        return (
            linear_mag > self.trot_start_linear_threshold
            or abs(yaw) > self.trot_start_yaw_threshold
        )

    def build_command(self):
        cmd_vel = self._get_fresh_cmd_vel()

        command = Command()
        command.horizontal_velocity = np.array([
            self._apply_deadband(float(cmd_vel.linear.x), self._trot_linear_deadband),
            self._apply_deadband(float(cmd_vel.linear.y), self._trot_linear_deadband),
        ])
        command.yaw_rate = self._apply_deadband(float(cmd_vel.angular.z), self._trot_yaw_deadband)

        if self.latest_mode == RobotMode.TROT:
            speed_scale = self._get_trot_speed_scale(cmd_vel)
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
            cmd_vel = self._get_fresh_cmd_vel()
            if self._has_trot_motion_intent(cmd_vel):
                self.state.behavior_state = BehaviorState.TROT
            else:
                self.state.behavior_state = BehaviorState.REST
        elif self.latest_mode == RobotMode.REST:
            self.state.behavior_state = BehaviorState.REST
        elif self.latest_mode in (RobotMode.SIT, RobotMode.LAY):
            self.state.behavior_state = BehaviorState.REST

    def _request_battery_voltage(self):
        if not self._transport_ready():
            return
        now = time.monotonic()
        if (now - self._last_battery_query_time) < self._battery_query_period_s:
            return
        self._last_battery_query_time = now
        if self._transport == 'i2c' and self._i2c_protocol == 'binary_v1':
            raw_voltage = self._read_i2c_battery_voltage()
            if raw_voltage is not None:
                self._publish_battery_voltage(raw_voltage)
            return

        self._write_transport_line('BAT?')

    def _request_imu_sample(self):
        if not self._transport_ready():
            return
        if not (self.use_imu and self.is_physical and self._physical_imu_via_arduino):
            return
        now = time.monotonic()
        if (now - self._last_imu_query_time) < self._imu_query_period_s:
            return
        self._last_imu_query_time = now

        if self._transport == 'i2c' and self._i2c_protocol == 'binary_v1':
            if not self._warned_i2c_imu_unsupported:
                self.node.get_logger().warn(
                    'IMU over Arduino I2C binary_v1 is not implemented in the current Arduino sketch. '
                    'Set physical_imu_via_arduino:=false and use direct IMU topic publishing instead.'
                )
                self._warned_i2c_imu_unsupported = True
            return

        self._write_transport_line('IMU?')

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

    def _covariance(self, variance: float):
        return [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance]

    def _build_imu_msg(self, values) -> Optional[Imu]:
        if len(values) != 10:
            return None

        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._imu_frame_id
        msg.orientation.x = float(values[0])
        msg.orientation.y = float(values[1])
        msg.orientation.z = float(values[2])
        msg.orientation.w = float(values[3])
        msg.orientation_covariance = self._covariance(
            self._imu_orientation_variance * self._imu_orientation_variance
        )
        msg.angular_velocity.x = float(values[4])
        msg.angular_velocity.y = float(values[5])
        msg.angular_velocity.z = float(values[6])
        msg.angular_velocity_covariance = self._covariance(
            self._imu_angular_velocity_variance * self._imu_angular_velocity_variance
        )
        msg.linear_acceleration.x = float(values[7])
        msg.linear_acceleration.y = float(values[8])
        msg.linear_acceleration.z = float(values[9])
        msg.linear_acceleration_covariance = self._covariance(
            self._imu_linear_acceleration_variance * self._imu_linear_acceleration_variance
        )
        return msg

    def _handle_serial_imu(self, line: str) -> bool:
        if not line.startswith('IMU:'):
            return False
        payload = line.split(':', 1)[1]
        try:
            values = [float(part.strip()) for part in payload.split(',')]
        except ValueError:
            self.node.get_logger().warn(f'Invalid IMU response: {line}')
            return True

        msg = self._build_imu_msg(values)
        if msg is None:
            self.node.get_logger().warn(
                'Invalid IMU response: expected 10 comma-separated values '
                '(qx,qy,qz,qw,gx,gy,gz,ax,ay,az)'
            )
            return True

        self.update_imu(msg)
        if self._imu_pub is not None:
            self._imu_pub.publish(msg)
        return True

    def _drain_serial_feedback(self):
        if not self._transport_ready():
            return

        if self._transport == 'i2c':
            if self._i2c_protocol == 'binary_v1':
                return
            # Arduino I2C request/response returns one payload per read.
            for _ in range(2):
                payload = self._read_i2c_line()
                if not payload:
                    continue
                for line in payload.splitlines():
                    self._handle_feedback_line(line.strip())
            return

        try:
            while self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                self._handle_feedback_line(line)
        except Exception as exc:
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

                self.send_joint_angles_to_arduino(joint_angles)

                self._request_battery_voltage()
                self._request_imu_sample()
                self._drain_serial_feedback()
                self._publish_cpu_temperature()

            time.sleep(self._loop_period)

    def send_joint_angles_to_arduino(self, joint_angles):
        if not self._transport_ready():
            return

        deg_angles = self._joint_angles_to_servo_degrees(joint_angles)
        self._send_servo_degrees_to_arduino(deg_angles)

    def _send_servo_degrees_to_arduino(self, deg_angles):
        if not self._transport_ready():
            return

        if self._transport == 'i2c' and self._i2c_protocol == 'binary_v1':
            self._write_i2c_servo_angles_binary_v1(deg_angles)
            return

        cmd_str = ','.join(f'{a:.2f}' for a in deg_angles) + '\n'
        self._write_transport_line(cmd_str)

    def _transport_ready(self):
        if self._transport == 'serial':
            return self.serial_port is not None and self.serial_port.is_open
        if self._transport == 'i2c':
            return self._i2c_bus is not None
        return False

    def _write_transport_line(self, text: str):
        if self._transport == 'serial':
            payload = text if text.endswith('\n') else f'{text}\n'
            self.serial_port.write(payload.encode('utf-8'))
            return

        if self._transport == 'i2c':
            if self._i2c_protocol == 'binary_v1':
                # binary_v1 path should use typed command helpers.
                return
            payload = text if text.endswith('\n') else f'{text}\n'
            data = payload.encode('ascii', errors='ignore')
            max_chunk = 28
            for idx in range(0, len(data), max_chunk):
                self._i2c_write_bytes(data[idx:idx + max_chunk])
            return

    def _i2c_write_bytes(self, data: bytes):
        if self._i2c_bus is None or not data:
            return

        if i2c_msg is not None:
            self._i2c_bus.i2c_rdwr(i2c_msg.write(self._i2c_address, list(data)))
            return

        if len(data) == 1:
            self._i2c_bus.write_byte(self._i2c_address, int(data[0]))
            return

        self._i2c_bus.write_i2c_block_data(
            self._i2c_address,
            int(data[0]),
            [int(b) for b in data[1:]],
        )

    def _i2c_read_bytes(self, count: int):
        if self._i2c_bus is None or count <= 0:
            return b''

        if i2c_msg is not None:
            msg = i2c_msg.read(self._i2c_address, count)
            self._i2c_bus.i2c_rdwr(msg)
            return bytes(msg)

        return bytes(self._i2c_bus.read_byte(self._i2c_address) for _ in range(count))

    def _write_i2c_servo_angles_binary_v1(self, deg_angles):
        if self._i2c_bus is None:
            return

        packed = [1]
        for angle in deg_angles:
            packed.append(int(max(0, min(180, round(angle)))))

        self._i2c_write_bytes(bytes(packed))

    def _read_i2c_battery_voltage(self):
        if self._i2c_bus is None:
            return None

        raw = self._i2c_read_bytes(2)
        if len(raw) != 2:
            return None

        centivolts = int.from_bytes(raw, byteorder='little', signed=False)
        return centivolts / 100.0

    def _read_i2c_line(self):
        if self._i2c_bus is None:
            return ''
        raw = self._i2c_read_bytes(self._i2c_read_len)
        text = raw.decode('utf-8', errors='ignore')
        return text.replace('\x00', '').strip()

    def _handle_feedback_line(self, line: str):
        if not line or line in {'OK', 'ERR', 'JAX_SERVO_READY'}:
            return
        if self._handle_serial_imu(line):
            return

        raw_voltage = None
        if line.startswith('VOLT:'):
            payload = line.split(':', 1)[1]
            try:
                raw_voltage = float(payload)
            except ValueError:
                self.node.get_logger().warn(f'Invalid battery response: {line}')
                return
        else:
            try:
                raw_voltage = float(line)
            except ValueError:
                return

        self._publish_battery_voltage(raw_voltage)

    def close_transport(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.serial_port = None
        if self._i2c_bus is not None:
            try:
                self._i2c_bus.close()
            except Exception:
                pass
            self._i2c_bus = None

    def _joint_angles_to_servo_degrees(self, joint_angles):
        flat_angles = [float(joint_angles[i, j]) for j in range(4) for i in range(3)]
        ordered_angles = [flat_angles[i] * self.servo_direction[i] for i in self.servo_order]
        return [
            max(0, min(180, math.degrees(angle) + 90 + self.servo_offset_deg[index]))
            for index, angle in enumerate(ordered_angles)
        ]

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
        if driver:
            driver.close_transport()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
