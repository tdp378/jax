import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('jax_description')
    default_model = os.path.join(pkg_share, 'urdf', 'jax.urdf')

    use_rviz = LaunchConfiguration('use_rviz')
    robot = LaunchConfiguration('robot')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value=default_model,
            description='Path to robot URDF or xacro',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz and joint_state_publisher_gui',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['xacro ', robot]),
                    value_type=str,
                ),
                'publish_frequency': 1000.0,
            }],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(use_rviz),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(use_rviz),
        ),
    ])
