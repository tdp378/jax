cmake_minimum_required(VERSION 3.8)
project(jax_peripheral_interfacing)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/cpu_temp_publisher.py
  scripts/jax_display_node.py
  scripts/jax_wifi_status_publisher.py
  scripts/mock_mjpeg_server.py
  scripts/mock_peripherals.py
  scripts/odom_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
