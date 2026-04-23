from jax_control.Gaits import GaitController
from jax_control.SwingLegController import SwingController
from jax_control.StanceController import StanceController
from jax_utilities.Utilities import clipped_first_order_filter
from jax_control.State import BehaviorState, State
from jax_msgs.msg import Angle, JointSpace, TaskSpace

import numpy as np
from transforms3d.euler import euler2mat
from geometry_msgs.msg import Point
from math import degrees


class Controller:
    """Controller and planner object"""

    def __init__(self, config, inverse_kinematics, node):
        self.config = config
        self._node = node

        self.task_space_pub = node.create_publisher(TaskSpace, 'task_space_goals', 10)
        self.joint_space_pub = node.create_publisher(JointSpace, 'joint_space_goals', 10)

        self.smoothed_yaw = 0.0
        self.inverse_kinematics = inverse_kinematics

        self.contact_modes = np.zeros(4)
        self.gait_controller = GaitController(self.config)
        self.swing_controller = SwingController(self.config)
        self.stance_controller = StanceController(self.config)
        # Attitude stabilization controller (uses IMU estimate, but is separate from sensing)
        self.stabilize_roll_kp = getattr(self.config, 'stabilize_roll_kp', 0.35)
        self.stabilize_roll_ki = getattr(self.config, 'stabilize_roll_ki', 0.03)
        self.stabilize_roll_kd = getattr(self.config, 'stabilize_roll_kd', 0.02)
        self.stabilize_pitch_kp = getattr(self.config, 'stabilize_pitch_kp', 0.75)
        self.stabilize_pitch_ki = getattr(self.config, 'stabilize_pitch_ki', 0.05)
        self.stabilize_pitch_kd = getattr(self.config, 'stabilize_pitch_kd', 0.03)
        self.stabilize_max_tilt = getattr(self.config, 'stabilize_max_tilt', 0.4)
        self.stabilize_deadband = getattr(self.config, 'stabilize_deadband', 0.04)
        self.stabilize_integral_limit = getattr(self.config, 'stabilize_integral_limit', 0.2)
        self.stabilize_compensation_limit = getattr(self.config, 'stabilize_compensation_limit', 0.5)
        self.imu_stabilization_enabled = bool(
            node.declare_parameter('imu_stabilization_enabled', True).value
        )
        self._roll_error_integral = 0.0
        self._pitch_error_integral = 0.0
        self._prev_roll_error = 0.0
        self._prev_pitch_error = 0.0

        if self.imu_stabilization_enabled:
            self._node.get_logger().info('IMU body stabilization enabled')
        else:
            self._node.get_logger().info('IMU body stabilization disabled')

        self.hop_transition_mapping = {BehaviorState.REST: BehaviorState.HOP, BehaviorState.HOP: BehaviorState.FINISHHOP, BehaviorState.FINISHHOP: BehaviorState.REST, BehaviorState.TROT: BehaviorState.HOP}
        self.trot_transition_mapping = {BehaviorState.REST: BehaviorState.TROT, BehaviorState.TROT: BehaviorState.REST, BehaviorState.HOP: BehaviorState.TROT, BehaviorState.FINISHHOP: BehaviorState.TROT}
        self.activate_transition_mapping = {BehaviorState.DEACTIVATED: BehaviorState.REST, BehaviorState.REST: BehaviorState.DEACTIVATED}

    def step_gait(self, state, command):
        contact_modes = self.gait_controller.contacts(state.ticks)
        new_foot_locations = np.zeros((3, 4))
        for leg_index in range(4):
            contact_mode = contact_modes[leg_index]
            foot_location = state.foot_locations[:, leg_index]
            if contact_mode == 1:
                new_location = self.stance_controller.next_foot_location(leg_index, state, command)
            else:
                swing_proportion = self.gait_controller.subphase_ticks(state.ticks) / self.config.swing_ticks
                new_location = self.swing_controller.next_foot_location(swing_proportion, leg_index, state, command)
            new_foot_locations[:, leg_index] = new_location
        return new_foot_locations, contact_modes

    def publish_task_space_command(self, rotated_foot_locations):
        task_space_message = TaskSpace()
        task_space_message.fr_foot = Point(
            x=float(rotated_foot_locations[0, 0] - self.config.LEG_ORIGINS[0, 0]),
            y=float(rotated_foot_locations[1, 0] - self.config.LEG_ORIGINS[1, 0]),
            z=float(rotated_foot_locations[2, 0] - self.config.LEG_ORIGINS[2, 0]),
        )
        task_space_message.fl_foot = Point(
            x=float(rotated_foot_locations[0, 1] - self.config.LEG_ORIGINS[0, 1]),
            y=float(rotated_foot_locations[1, 1] - self.config.LEG_ORIGINS[1, 1]),
            z=float(rotated_foot_locations[2, 1] - self.config.LEG_ORIGINS[2, 1]),
        )
        task_space_message.rr_foot = Point(
            x=float(rotated_foot_locations[0, 2] - self.config.LEG_ORIGINS[0, 2]),
            y=float(rotated_foot_locations[1, 2] - self.config.LEG_ORIGINS[1, 2]),
            z=float(rotated_foot_locations[2, 2] - self.config.LEG_ORIGINS[2, 2]),
        )
        task_space_message.rl_foot = Point(
            x=float(rotated_foot_locations[0, 3] - self.config.LEG_ORIGINS[0, 3]),
            y=float(rotated_foot_locations[1, 3] - self.config.LEG_ORIGINS[1, 3]),
            z=float(rotated_foot_locations[2, 3] - self.config.LEG_ORIGINS[2, 3]),
        )
        task_space_message.header.stamp = self._node.get_clock().now().to_msg()
        self.task_space_pub.publish(task_space_message)

    def publish_joint_space_command(self, angle_matrix):
        joint_space_message = JointSpace()
        joint_space_message.fr_foot = Angle(theta1=float(degrees(angle_matrix[0, 0])), theta2=float(degrees(angle_matrix[1, 0])), theta3=float(degrees(angle_matrix[2, 0])))
        joint_space_message.fl_foot = Angle(theta1=float(degrees(angle_matrix[0, 1])), theta2=float(degrees(angle_matrix[1, 1])), theta3=float(degrees(angle_matrix[2, 1])))
        joint_space_message.rr_foot = Angle(theta1=float(degrees(angle_matrix[0, 2])), theta2=float(degrees(angle_matrix[1, 2])), theta3=float(degrees(angle_matrix[2, 2])))
        joint_space_message.rl_foot = Angle(theta1=float(degrees(angle_matrix[0, 3])), theta2=float(degrees(angle_matrix[1, 3])), theta3=float(degrees(angle_matrix[2, 3])))
        joint_space_message.header.stamp = self._node.get_clock().now().to_msg()
        self.joint_space_pub.publish(joint_space_message)

    def _apply_pose(self, state, foot_locations, orientation, stabilise=True):
        rotated_foot_locations = foot_locations
        if stabilise:
            rotated_foot_locations = self.stabilise_with_IMU(rotated_foot_locations, orientation)
        state.foot_locations = foot_locations
        state.rotated_foot_locations = rotated_foot_locations
        state.joint_angles = self.inverse_kinematics(rotated_foot_locations, self.config)

    def _imu_attitude_compensation(self, orientation):
        if not self.imu_stabilization_enabled:
            return 0.0, 0.0

        _, pitch, roll = orientation
        dt = max(self.config.dt, 1e-4)

        # We target level body orientation during locomotion/rest.
        roll_error = -np.clip(roll, -self.stabilize_max_tilt, self.stabilize_max_tilt)
        pitch_error = np.clip(pitch, -self.stabilize_max_tilt, self.stabilize_max_tilt)
        if abs(roll_error) < self.stabilize_deadband:
            roll_error = 0.0
        if abs(pitch_error) < self.stabilize_deadband:
            pitch_error = 0.0

        self._roll_error_integral = np.clip(
            self._roll_error_integral + roll_error * dt,
            -self.stabilize_integral_limit,
            self.stabilize_integral_limit,
        )
        self._pitch_error_integral = np.clip(
            self._pitch_error_integral + pitch_error * dt,
            -self.stabilize_integral_limit,
            self.stabilize_integral_limit,
        )

        roll_error_derivative = (roll_error - self._prev_roll_error) / dt
        pitch_error_derivative = (pitch_error - self._prev_pitch_error) / dt
        self._prev_roll_error = roll_error
        self._prev_pitch_error = pitch_error

        roll_compensation = (
            self.stabilize_roll_kp * roll_error
            + self.stabilize_roll_ki * self._roll_error_integral
            + self.stabilize_roll_kd * roll_error_derivative
        )
        pitch_compensation = (
            self.stabilize_pitch_kp * pitch_error
            + self.stabilize_pitch_ki * self._pitch_error_integral
            + self.stabilize_pitch_kd * pitch_error_derivative
        )

        roll_compensation = np.clip(
            roll_compensation,
            -self.stabilize_compensation_limit,
            self.stabilize_compensation_limit,
        )
        pitch_compensation = np.clip(
            pitch_compensation,
            -self.stabilize_compensation_limit,
            self.stabilize_compensation_limit,
        )
        return roll_compensation, pitch_compensation

    def run(self, state, command):
        previous_state = state.behavior_state

        if command.joystick_control_event:
            state.behavior_state = self.activate_transition_mapping[state.behavior_state]
        elif command.trot_event:
            state.behavior_state = self.trot_transition_mapping[state.behavior_state]
        elif command.hop_event:
            state.behavior_state = self.hop_transition_mapping[state.behavior_state]

        if previous_state != state.behavior_state:
            self._node.get_logger().info(f'State changed from {previous_state!s} to {state.behavior_state!s}')
            # Avoid integrator carry-over when changing behavior modes.
            self._roll_error_integral = 0.0
            self._pitch_error_integral = 0.0
            self._prev_roll_error = 0.0
            self._prev_pitch_error = 0.0

        if state.behavior_state == BehaviorState.TROT:
            state.foot_locations, contact_modes = self.step_gait(state, command)
            rotated_foot_locations = euler2mat(command.roll, command.pitch, 0.0) @ state.foot_locations
            roll_compensation, pitch_compensation = self._imu_attitude_compensation(state.euler_orientation)
            rmat = euler2mat(roll_compensation, pitch_compensation, 0)
            rotated_foot_locations = rmat.T @ rotated_foot_locations
            state.joint_angles = self.inverse_kinematics(rotated_foot_locations, self.config)
            state.rotated_foot_locations = rotated_foot_locations

        elif state.behavior_state == BehaviorState.REST:
            yaw_proportion = command.yaw_rate / self.config.max_yaw_rate
            self.smoothed_yaw += self.config.dt * clipped_first_order_filter(
                self.smoothed_yaw,
                yaw_proportion * -self.config.max_stance_yaw,
                self.config.max_stance_yaw_rate,
                self.config.yaw_time_constant,
            )
            foot_locations = self.config.rest_stance.copy()
            foot_locations[2, :] = command.height
            rotated_foot_locations = euler2mat(command.roll, command.pitch, self.smoothed_yaw) @ foot_locations
            rotated_foot_locations = self.stabilise_with_IMU(rotated_foot_locations, state.euler_orientation)
            state.foot_locations = foot_locations
            state.joint_angles = self.inverse_kinematics(rotated_foot_locations, self.config)
            state.rotated_foot_locations = rotated_foot_locations

        state.ticks += 1
        state.pitch = command.pitch
        state.roll = command.roll
        state.height = command.height

    def set_pose_to_default(self, state):
        state.foot_locations = self.config.default_stance + np.array([0, 0, self.config.default_z_ref])[:, np.newaxis]
        state.joint_angles = self.inverse_kinematics(state.foot_locations, self.config)
        return state.joint_angles

    def stabilise_with_IMU(self, foot_locations, orientation):
        if not self.imu_stabilization_enabled:
            return foot_locations

        roll_compensation, pitch_compensation = self._imu_attitude_compensation(orientation)
        rmat = euler2mat(roll_compensation, pitch_compensation, 0)
        rotated_foot_locations = rmat.T @ foot_locations
        return rotated_foot_locations
