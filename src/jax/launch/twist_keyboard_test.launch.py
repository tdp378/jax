"""Keyboard smoke test: publishes geometry_msgs/Twist on /cmd_vel.

Jax leg control uses /joy + jax_driver, not cmd_vel — use this only to verify
keyboard → ROS (e.g. ``ros2 topic echo /cmd_vel`` in another terminal).

Requires: ``sudo apt install ros-jazzy-teleop-twist-keyboard``

``ros2 launch`` does **not** forward keystrokes to child processes (stdin is a pipe), so
this launch opens a **new terminal window** where ``teleop_twist_keyboard`` runs.

Ensure you already ran ``source /opt/ros/jazzy/setup.bash`` and
``source .../install/setup.bash`` in the shell **before** ``ros2 launch``, so the new
terminal inherits a usable environment (or configure ROS in ``~/.bashrc``).

Keys: ``i`` forward, ``,`` back, ``j``/``l`` turn.
"""
import shutil

from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo


def generate_launch_description():
    # One argv string for bash -lc (no extra shell layer from launch)
    inner = (
        'ros2 run teleop_twist_keyboard teleop_twist_keyboard '
        '--ros-args -p use_sim_time:=false; '
        'read -p "Press Enter to close... "'
    )

    actions = [
        LogInfo(
            msg=(
                'Opening a separate terminal for teleop (launch cannot pipe your keys into '
                'subprocesses). Use another terminal for: ros2 topic echo /cmd_vel'
            )
        ),
    ]

    if shutil.which('gnome-terminal'):
        actions.append(
            ExecuteProcess(
                cmd=[
                    'gnome-terminal',
                    '--title',
                    'teleop_twist_keyboard',
                    '--',
                    'bash',
                    '-lc',
                    inner,
                ],
                output='screen',
            )
        )
    elif shutil.which('xfce4-terminal'):
        actions.append(
            ExecuteProcess(
                cmd=[
                    'xfce4-terminal',
                    '--title=teleop_twist_keyboard',
                    '-x',
                    'bash',
                    '-lc',
                    inner,
                ],
                output='screen',
            )
        )
    elif shutil.which('konsole'):
        actions.append(
            ExecuteProcess(
                cmd=['konsole', '-e', 'bash', '-lc', inner],
                output='screen',
            )
        )
    elif shutil.which('xterm'):
        actions.append(
            ExecuteProcess(
                cmd=['xterm', '-title', 'teleop_twist_keyboard', '-e', 'bash', '-lc', inner],
                output='screen',
            )
        )
    else:
        actions.append(
            LogInfo(
                msg=(
                    'No gnome-terminal / xfce4-terminal / konsole / xterm found. '
                    'In a normal terminal run: ros2 run teleop_twist_keyboard teleop_twist_keyboard'
                )
            )
        )

    return LaunchDescription(actions)
