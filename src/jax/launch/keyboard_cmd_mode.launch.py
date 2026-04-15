from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jax',
            executable='keyboard_cmd_mode.py',
            name='keyboard_cmd_mode',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}],
        ),
    ])
