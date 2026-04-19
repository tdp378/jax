from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    is_sim = LaunchConfiguration('is_sim')
    is_physical = LaunchConfiguration('is_physical')
    use_imu = LaunchConfiguration('use_imu')
    driver_params = os.path.join(
        get_package_share_directory('jax'),
        'config',
        'driver.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('is_sim', default_value='0'),
        DeclareLaunchArgument('is_physical', default_value='1'),
        DeclareLaunchArgument('use_imu', default_value='0'),

        Node(
            package='jax_peripheral_interfacing',
            executable='jax_display_node.py',
            name='jax_LCD_node',
            output='screen',
            condition=IfCondition(is_physical),
            parameters=[driver_params, {'battery_state_topic': '/jax/battery'}],
        ),
        Node(
            package='jax_behaviors',
            executable='mode_manager.py',
            name='jax_mode_manager',
            output='screen',
        ),
        Node(
            package='jax',
            executable='jax_driver.py',
            name='jax_driver',
            output='screen',
            arguments=[is_sim, is_physical, use_imu],
            parameters=[driver_params],
        ),
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{
                'port': 9090,
                'address': '0.0.0.0',
            }],
        ),
        Node(
            package='rosapi',
            executable='rosapi_node',
            name='rosapi',
            output='screen',
        ),
        Node(
            package='jax_peripheral_interfacing',
            executable='jax_wifi_status_publisher.py',
            name='jax_wifi_status_publisher',
            output='screen',
        ),
    ])
