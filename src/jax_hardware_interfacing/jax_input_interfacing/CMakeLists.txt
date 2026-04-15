cmake_minimum_required(VERSION 3.8)
project(jax_input_interfacing)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/jax_input_interfacing)

install(PROGRAMS
  scripts/Keyboard.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
