"""Drive Jax in Gazebo using Twist teleop (``/cmd_vel``) via a Joy bridge.

Prerequisites
-------------
- Gazebo sim running with active ``leg_joint_position_controller``.
- In another terminal: ``teleop_twist_keyboard`` (or any ``/cmd_vel`` publisher).

This launch runs **only** ``cmd_vel_to_joy_bridge`` + ``jax_driver`` (sim, no IMU).
**Do not** also run ``jax.launch.py`` with keyboard or ``joy_node`` — one ``/joy`` publisher.

Example
-------
Terminal 1: ``ros2 launch jax jax_gazebo_sim.launch.py``

Terminal 2::

    source /opt/ros/jazzy/setup.bash
    source ~/JaxQuadruped/jax_ws/install/setup.bash
    ros2 launch jax twist_teleop_jax_sim.launch.py

Terminal 3 (teleop in a real TTY, or use ``twist_keyboard_test.launch.py`` for a new window)::

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Use ``i`` / ``,`` / ``j`` / ``l``; this is a coarse mapping, not a full gait planner.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge = Node(
        package='jax',
        executable='cmd_vel_to_joy_bridge.py',
        name='cmd_vel_to_joy_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    driver = Node(
        package='jax',
        executable='jax_driver.py',
        name='jax_driver',
        output='screen',
        arguments=['1', '0', '0'],
    )
    return LaunchDescription([bridge, driver])
