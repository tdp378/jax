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

        if state.behavior_state == BehaviorState.TROT:
            state.foot_locations, contact_modes = self.step_gait(state, command)
            rotated_foot_locations = euler2mat(command.roll, command.pitch, 0.0) @ state.foot_locations
            yaw, pitch, roll = state.euler_orientation
            correction_factor = 0.8
            max_tilt = 0.4
            roll_compensation = correction_factor * np.clip(roll, -max_tilt, max_tilt)
            pitch_compensation = correction_factor * np.clip(pitch, -max_tilt, max_tilt)
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

        elif state.behavior_state == BehaviorState.SIT:
            self._apply_pose(state, self.config.sit_stance, state.euler_orientation, stabilise=False)

        elif state.behavior_state == BehaviorState.LAY:
            self._apply_pose(state, self.config.lay_stance, state.euler_orientation, stabilise=False)

        state.ticks += 1
        state.pitch = command.pitch
        state.roll = command.roll
        state.height = command.height

    def set_pose_to_default(self, state):
        state.foot_locations = self.config.default_stance + np.array([0, 0, self.config.default_z_ref])[:, np.newaxis]
        state.joint_angles = self.inverse_kinematics(state.foot_locations, self.config)
        return state.joint_angles

    def stabilise_with_IMU(self, foot_locations, orientation):
        yaw, pitch, roll = orientation
        correction_factor = 0.5
        max_tilt = 0.4
        roll_compensation = correction_factor * np.clip(-roll, -max_tilt, max_tilt)
        pitch_compensation = correction_factor * np.clip(-pitch, -max_tilt, max_tilt)
        rmat = euler2mat(roll_compensation, pitch_compensation, 0)
        rotated_foot_locations = rmat.T @ foot_locations
        return rotated_foot_locations
