from ament_index_python.packages import get_package_share_directory
from ament_index_python.resources import has_resource

from launch.actions import OpaqueFunction
from launch.actions import DeclareLaunchArgument
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode


def _is_enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_int(value: str):
    return int(value) if value else None


def _launch_setup(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file")
    camera_param = LaunchConfiguration("camera")
    format_param = LaunchConfiguration("format")
    width_param = LaunchConfiguration("width")
    height_param = LaunchConfiguration("height")
    use_image_view = _is_enabled(LaunchConfiguration("use_image_view").perform(context))
    use_web_video_server = _is_enabled(
        LaunchConfiguration("use_web_video_server").perform(context)
    )
    web_video_server_port = LaunchConfiguration("web_video_server_port")
    web_video_server_address = LaunchConfiguration("web_video_server_address")
    camera_container_log_level = LaunchConfiguration("camera_container_log_level")
    web_video_server_log_level = LaunchConfiguration("web_video_server_log_level")

    camera_override = LaunchConfiguration("camera").perform(context)
    format_override = LaunchConfiguration("format").perform(context)
    width_override = LaunchConfiguration("width").perform(context)
    height_override = LaunchConfiguration("height").perform(context)

    node_parameters = [params_file]
    launch_overrides = {}

    if camera_override:
        launch_overrides["camera"] = camera_param
    if format_override:
        launch_overrides["format"] = format_param
    if width_override:
        launch_overrides["width"] = _optional_int(width_override)
    if height_override:
        launch_overrides["height"] = _optional_int(height_override)

    if launch_overrides:
        node_parameters.append(launch_overrides)

    composable_nodes = [
        ComposableNode(
            name='camera',
            package='camera_ros',
            plugin='camera::CameraNode',
            parameters=node_parameters,
            extra_arguments=[{'use_intra_process_comms': True}],
        ),
    ]

    if use_image_view and has_resource("packages", "image_view"):
        composable_nodes.append(
            ComposableNode(
                package='image_view',
                plugin='image_view::ImageViewNode',
                remappings=[('/image', '/camera/image_raw')],
                extra_arguments=[{'use_intra_process_comms': True}],
            )
        )

    actions = [
        ComposableNodeContainer(
            name='camera_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            arguments=['--ros-args', '--log-level', camera_container_log_level],
            composable_node_descriptions=composable_nodes,
        )
    ]

    if use_web_video_server:
        actions.append(
            Node(
                package='web_video_server',
                executable='web_video_server',
                name='web_video_server',
                output='screen',
                arguments=['--ros-args', '--log-level', web_video_server_log_level],
                parameters=[{
                    'port': web_video_server_port,
                    'address': web_video_server_address,
                    'default_stream_type': 'mjpeg',
                }],
            )
        )

    return actions


def generate_launch_description() -> LaunchDescription:
    """
    Generate a launch description with for the camera node and a visualiser.

    Returns
    -------
        LaunchDescription: the launch description

    """
    # parameters
    params_file_default = get_package_share_directory('camera_ros') + '/config/camera.yaml'

    params_file_launch_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_file_default,
        description='camera parameter YAML file'
    )

    camera_param_name = "camera"
    camera_launch_arg = DeclareLaunchArgument(
        camera_param_name,
        default_value='',
        description='optional camera override; leave empty to use YAML'
    )

    format_param_name = "format"
    format_launch_arg = DeclareLaunchArgument(
        format_param_name,
        default_value='',
        description='optional pixel format override; leave empty to use YAML'
    )

    width_launch_arg = DeclareLaunchArgument(
        'width',
        default_value='',
        description='optional camera frame width override; leave empty to use YAML'
    )

    height_launch_arg = DeclareLaunchArgument(
        'height',
        default_value='',
        description='optional camera frame height override; leave empty to use YAML'
    )

    use_image_view_launch_arg = DeclareLaunchArgument(
        'use_image_view',
        default_value='0',
        description='show the camera stream in image_view if installed'
    )

    use_web_video_server_launch_arg = DeclareLaunchArgument(
        'use_web_video_server',
        default_value='1',
        description='start web_video_server for MJPEG HTTP streaming'
    )

    web_video_server_port_launch_arg = DeclareLaunchArgument(
        'web_video_server_port',
        default_value='8080',
        description='HTTP port for the MJPEG stream server'
    )

    web_video_server_address_launch_arg = DeclareLaunchArgument(
        'web_video_server_address',
        default_value='0.0.0.0',
        description='bind address for the MJPEG stream server'
    )

    camera_container_log_level_launch_arg = DeclareLaunchArgument(
        'camera_container_log_level',
        default_value='warn',
        description='ROS log level for the camera component container'
    )

    web_video_server_log_level_launch_arg = DeclareLaunchArgument(
        'web_video_server_log_level',
        default_value='warn',
        description='ROS log level for web_video_server'
    )

    return LaunchDescription([
        params_file_launch_arg,
        camera_launch_arg,
        format_launch_arg,
        width_launch_arg,
        height_launch_arg,
        use_image_view_launch_arg,
        use_web_video_server_launch_arg,
        web_video_server_port_launch_arg,
        web_video_server_address_launch_arg,
        camera_container_log_level_launch_arg,
        web_video_server_log_level_launch_arg,
        OpaqueFunction(function=_launch_setup),
    ])
