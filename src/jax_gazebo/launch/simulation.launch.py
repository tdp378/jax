import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc_share = get_package_share_directory('jax_description')
    gz_model_path = os.path.dirname(desc_share)
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    jax_gazebo_share = get_package_share_directory('jax_gazebo')

    world_path = os.path.join(jax_gazebo_share, 'launch', 'world', 'normal.world')
    ctrl_yaml_path = os.path.join(jax_gazebo_share, 'config', 'jax_gz_ros2_control.yaml')

    robot_description_path = os.path.join(desc_share, 'urdf', 'jax.urdf')
    robot_description_config = xacro.process_file(robot_description_path)
    desc_xml = robot_description_config.toxml().replace(
        '__JAX_GZ_ROS2_CONTROL_YAML__',
        ctrl_yaml_path,
    )

    robot_description = {
        'robot_description': desc_xml,
        'use_sim_time': True,
    }

    # -----------------------------
    # Launch arguments
    # -----------------------------
    mock_battery_voltage = LaunchConfiguration('mock_battery_voltage')
    mock_servo_buck_voltage = LaunchConfiguration('mock_servo_buck_voltage')
    mock_battery_drain_per_second = LaunchConfiguration('mock_battery_drain_per_second')
    cpu_temp_c = LaunchConfiguration('cpu_temp_c')
    cpu_temp_variance_c = LaunchConfiguration('cpu_temp_variance_c')
    use_odom = LaunchConfiguration('use_odom')
    odom_topic = LaunchConfiguration('odom_topic')
    odom_cmd_vel_topic = LaunchConfiguration('odom_cmd_vel_topic')
    odom_frame_id = LaunchConfiguration('odom_frame_id')
    odom_base_frame_id = LaunchConfiguration('odom_base_frame_id')
    odom_publish_rate_hz = LaunchConfiguration('odom_publish_rate_hz')
    use_imu = LaunchConfiguration('use_imu')
    use_mjpeg_server = LaunchConfiguration('use_mjpeg_server')
    mjpeg_server_port = LaunchConfiguration('mjpeg_server_port')
    mjpeg_server_frame_width = LaunchConfiguration('mjpeg_server_frame_width')
    mjpeg_server_frame_height = LaunchConfiguration('mjpeg_server_frame_height')
    mjpeg_server_frame_rate = LaunchConfiguration('mjpeg_server_frame_rate')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_delay = LaunchConfiguration('spawn_delay')
    controller_startup_seconds = LaunchConfiguration('controller_startup_seconds')
    jax_stack_startup_seconds = LaunchConfiguration('jax_stack_startup_seconds')

    # -----------------------------
    # Gazebo
    # -----------------------------
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    # -----------------------------
    # Robot description publisher
    # -----------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # -----------------------------
    # Gazebo <-> ROS clock bridge
    # -----------------------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/jax/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        output='screen',
    )

    # -----------------------------
    # rosbridge for Android/WebSocket
    # -----------------------------
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{
            'port': 9090,
            'address': '0.0.0.0',
        }],
    )

    # -----------------------------
    # rosapi for topic/service discovery
    # -----------------------------
    rosapi = Node(
        package='rosapi',
        executable='rosapi_node',
        name='rosapi',
        output='screen',
    )

    # -----------------------------
    # Spawn robot into Gazebo
    # -----------------------------
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        parameters=[{
            'world': 'default',
            'name': 'jax',
            'topic': 'robot_description',
            'x': 0.0,
            'y': 0.0,
            'z': spawn_z,
        }],
    )

    delayed_spawn = TimerAction(
        period=spawn_delay,
        actions=[spawn_entity],
    )

    # -----------------------------
    # ros2_control controller config
    # -----------------------------
    ctrl_file = PathJoinSubstitution([
        FindPackageShare('jax_gazebo'),
        'config',
        'jax_gz_ros2_control.yaml',
    ])

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager-timeout',
            '60',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    leg_joint_position_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'leg_joint_position_controller',
            '--param-file',
            ctrl_file,
            '--controller-manager-timeout',
            '60',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Match initial_value in URDF so the model stands before driver takes over
    pub_stand_pose = ExecuteProcess(
        cmd=[
            'bash',
            '-c',
            'sleep 1.0 && ros2 topic pub --once /leg_joint_position_controller/commands '
            'std_msgs/msg/Float64MultiArray '
            '"{data: [0.0, 0.72646626, 0.0, 0.0, 0.72646626, 0.0, 0.0, 0.72646626, 0.0, 0.0, 0.72646626, 0.0]}"',
        ],
        output='screen',
    )

    # -----------------------------
    # Jax control stack
    # -----------------------------
    # jax_driver.py:
    #   args = [is_sim, is_physical, use_imu]
    # In sim we want: is_sim=1, is_physical=0, use_imu from launch arg.
    jax_driver = Node(
        package='jax',
        executable='jax_driver.py',
        name='jax_driver',
        output='screen',
        arguments=['1', '0', use_imu],
        parameters=[
            {
                'use_sim_time': True,
                'gz_leg_command_topic': '/jax/trot_joint_commands',
            },
                   ],
    )

    jax_mode_manager = Node(
        package='jax_behaviors',
        executable='mode_manager.py',
        name='jax_mode_manager',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ],
    )

    mock_peripherals = Node(
        package='jax_peripheral_interfacing',
        executable='mock_peripherals.py',
        name='mock_peripherals',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'battery_voltage_level': mock_battery_voltage,
            'servo_buck_voltage_level': mock_servo_buck_voltage,
            'drain_per_second': mock_battery_drain_per_second,
            'cpu_temp_c': cpu_temp_c,
            'cpu_temp_variance_c': cpu_temp_variance_c,
        }],
    )

    odom_publisher = Node(
        package='jax_peripheral_interfacing',
        executable='odom_publisher.py',
        name='odom_publisher',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'odom_topic': odom_topic,
                'cmd_vel_topic': odom_cmd_vel_topic,
                'odom_frame_id': odom_frame_id,
                'base_frame_id': odom_base_frame_id,
                'publish_rate_hz': odom_publish_rate_hz,
            }
        ],
        condition=IfCondition(use_odom),
    )

    jax_display = Node(
        package='jax_peripheral_interfacing',
        executable='jax_display_node.py',
        name='jax_display_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'sim': True,
            'imu_topic': '/jax/imu',
        }],
    )

    mock_mjpeg_server = Node(
        package='jax_peripheral_interfacing',
        executable='mock_mjpeg_server.py',
        name='mock_mjpeg_server',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'host': '0.0.0.0',
                'port': mjpeg_server_port,
                'frame_width': mjpeg_server_frame_width,
                'frame_height': mjpeg_server_frame_height,
                'frame_rate': mjpeg_server_frame_rate,
            }
        ],
        condition=IfCondition(use_mjpeg_server),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mock_battery_voltage',
            default_value='16.4',
            description='Simulated battery voltage in volts.',
        ),
        DeclareLaunchArgument(
            'mock_servo_buck_voltage',
            default_value='5.0',
            description='Simulated servo buck voltage in volts.',
        ),
        DeclareLaunchArgument(
            'mock_battery_drain_per_second',
            default_value='0.0',
            description='Optional battery drain rate in volts per second.',
        ),
        DeclareLaunchArgument(
            'cpu_temp_c',
            default_value='54.0',
            description='Simulated CPU temperature in celsius.',
        ),
        DeclareLaunchArgument(
            'cpu_temp_variance_c',
            default_value='0.5',
            description='Alternating +/- variance around cpu_temp_c in celsius.',
        ),
        DeclareLaunchArgument(
            'use_odom',
            default_value='1',
            description='Enable odometry publisher for simulation.',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry topic name.',
        ),
        DeclareLaunchArgument(
            'odom_cmd_vel_topic',
            default_value='/cmd_vel',
            description='Input velocity topic used to generate odometry.',
        ),
        DeclareLaunchArgument(
            'odom_frame_id',
            default_value='odom',
            description='Odometry frame id.',
        ),
        DeclareLaunchArgument(
            'odom_base_frame_id',
            default_value='base_link',
            description='Base child frame id for odometry.',
        ),
        DeclareLaunchArgument(
            'odom_publish_rate_hz',
            default_value='30.0',
            description='Odometry publish rate in Hz.',
        ),
        DeclareLaunchArgument(
            'use_mjpeg_server',
            default_value='1',
            description='Enable mock MJPEG server for testing camera integration.',
        ),
        DeclareLaunchArgument(
            'mjpeg_server_port',
            default_value='8081',
            description='Port for mock MJPEG server (accessible at http://localhost:<port>/video).',
        ),
        DeclareLaunchArgument(
            'mjpeg_server_frame_width',
            default_value='640',
            description='Mock MJPEG frame width in pixels.',
        ),
        DeclareLaunchArgument(
            'mjpeg_server_frame_height',
            default_value='480',
            description='Mock MJPEG frame height in pixels.',
        ),
        DeclareLaunchArgument(
            'mjpeg_server_frame_rate',
            default_value='30',
            description='Mock MJPEG frame rate in Hz.',
        ),
        DeclareLaunchArgument(
            'use_imu',
            default_value='1',
            description='Enable simulated IMU subscription in jax_driver.',
        ),
        DeclareLaunchArgument(
            'spawn_z',
            default_value='0.45',
            description='Initial spawn height (m) above ground so the model clears the plane',
        ),
        DeclareLaunchArgument(
            'spawn_delay',
            default_value='3.0',
            description='Seconds to wait before spawning (lets gz sim finish loading)',
        ),
        DeclareLaunchArgument(
            'controller_startup_seconds',
            default_value='8.0',
            description=(
                'Seconds after launch to spawn controllers (after gz + robot spawn). '
                'Increase if controllers remain unconfigured.'
            ),
        ),
        DeclareLaunchArgument(
            'jax_stack_startup_seconds',
            default_value='10.0',
            description=(
                'Seconds after launch to start jax_driver and jax_mode_manager. '
                'Set later than controller startup so the controller is ready first.'
            ),
        ),

        AppendEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=gz_model_path,
            prepend=True,
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[leg_joint_position_spawner],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=leg_joint_position_spawner,
                on_exit=[pub_stand_pose],
            )
        ),

        TimerAction(
            period=controller_startup_seconds,
            actions=[joint_state_broadcaster_spawner],
        ),

        # Start semantic control stack after controllers are up
        TimerAction(
            period=jax_stack_startup_seconds,
            actions=[jax_driver, jax_mode_manager],
        ),

        gz_sim,
        robot_state_publisher,
        bridge,
        rosbridge,
        rosapi,
        mock_peripherals,
        odom_publisher,
        mock_mjpeg_server,
        jax_display,
        delayed_spawn,
    ])