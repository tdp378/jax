from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
import os


def generate_launch_description():
    is_sim = LaunchConfiguration('is_sim')
    is_physical = LaunchConfiguration('is_physical')
    use_imu = LaunchConfiguration('use_imu')
    use_camera = LaunchConfiguration('use_camera')
    camera_params_file = LaunchConfiguration('camera_params_file')
    camera = LaunchConfiguration('camera')
    camera_format = LaunchConfiguration('camera_format')
    camera_width = LaunchConfiguration('camera_width')
    camera_height = LaunchConfiguration('camera_height')
    camera_stream_port = LaunchConfiguration('camera_stream_port')
    use_direct_i2c_imu = LaunchConfiguration('use_direct_i2c_imu')
    rosbridge_log_level = LaunchConfiguration('rosbridge_log_level')
    rosapi_log_level = LaunchConfiguration('rosapi_log_level')
    camera_container_log_level = LaunchConfiguration('camera_container_log_level')
    web_video_server_log_level = LaunchConfiguration('web_video_server_log_level')
    driver_params = os.path.join(
        get_package_share_directory('jax'),
        'config',
        'driver.yaml',
    )
    camera_params_default = os.path.join(
        get_package_share_directory('camera_ros'),
        'config',
        'camera.yaml',
    )
    camera_launch = os.path.join(
        get_package_share_directory('camera_ros'),
        'launch',
        'camera.launch.py',
    )

    return LaunchDescription([
        DeclareLaunchArgument('is_sim', default_value='0'),
        DeclareLaunchArgument('is_physical', default_value='1'),
        DeclareLaunchArgument('use_imu', default_value='0'),
        DeclareLaunchArgument('use_camera', default_value='1'),
        DeclareLaunchArgument('camera_params_file', default_value=camera_params_default),
        DeclareLaunchArgument('camera', default_value=''),
        DeclareLaunchArgument('camera_format', default_value=''),
        DeclareLaunchArgument('camera_width', default_value=''),
        DeclareLaunchArgument('camera_height', default_value=''),
        DeclareLaunchArgument('camera_stream_port', default_value='8080'),
        DeclareLaunchArgument('use_direct_i2c_imu', default_value='0'),
        DeclareLaunchArgument('rosbridge_log_level', default_value='warn'),
        DeclareLaunchArgument('rosapi_log_level', default_value='warn'),
        DeclareLaunchArgument('camera_container_log_level', default_value='warn'),
        DeclareLaunchArgument('web_video_server_log_level', default_value='warn'),

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
            package='jax_peripheral_interfacing',
            executable='bno08x_imu_publisher.py',
            name='bno08x_imu',
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", is_physical, "' == '1' and '", use_imu,
                "' == '1' and '", use_direct_i2c_imu, "' == '1'",
            ])),
            parameters=[driver_params],
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
            arguments=['--ros-args', '--log-level', rosbridge_log_level],
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
            arguments=['--ros-args', '--log-level', rosapi_log_level],
        ),
        Node(
            package='jax_peripheral_interfacing',
            executable='jax_wifi_status_publisher.py',
            name='jax_wifi_status_publisher',
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch),
            condition=IfCondition(use_camera),
            launch_arguments={
                'params_file': camera_params_file,
                'camera': camera,
                'format': camera_format,
                'width': camera_width,
                'height': camera_height,
                'use_image_view': '0',
                'use_web_video_server': is_physical,
                'web_video_server_port': camera_stream_port,
                'camera_container_log_level': camera_container_log_level,
                'web_video_server_log_level': web_video_server_log_level,
            }.items(),
        ),
    ])
