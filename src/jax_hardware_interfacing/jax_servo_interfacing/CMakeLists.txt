cmake_minimum_required(VERSION 3.8)
project(jax_servo_interfacing)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/jax_servo_interfacing)

ament_package()
