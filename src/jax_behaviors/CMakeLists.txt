cmake_minimum_required(VERSION 3.8)
project(jax_behaviors)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/mode_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()