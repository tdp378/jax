cmake_minimum_required(VERSION 3.8)
project(jax)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/jax)

install(PROGRAMS
  scripts/jax_driver.py
  scripts/keyboard_cmd_mode.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
