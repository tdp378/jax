#!/usr/bin/env python3
"""
Single mock node that simulates all peripheral sensors for the Gazebo sim:
battery, CPU temperature, Wi-Fi status, and a camera heartbeat.
"""
import random

import rclpy
from rclpy.node import Node
from jax_msgs.msg import ElectricalMeasurements
from sensor_msgs.msg import BatteryState, Image, Temperature
from std_msgs.msg import String


class MockPeripherals(Node):
    def __init__(self):
        super().__init__('mock_peripherals')

        # ---- Battery params ----
        self._battery_voltage = float(
            self.declare_parameter('battery_voltage_level', 16.4).value)
        self._servo_buck_voltage = float(
            self.declare_parameter('servo_buck_voltage_level', 5.0).value)
        self._drain_per_second = float(
            self.declare_parameter('drain_per_second', 0.0).value)
        self._min_battery_voltage = float(
            self.declare_parameter('min_battery_voltage', 14.0).value)
        self._battery_rate_hz = float(
            self.declare_parameter('battery_rate_hz', 2.0).value)

        # ---- CPU temp params ----
        self._cpu_temp_c = float(
            self.declare_parameter('cpu_temp_c', 54.0).value)
        self._cpu_temp_variance_c = float(
            self.declare_parameter('cpu_temp_variance_c', 0.5).value)
        self._cpu_temp_rate_hz = float(
            self.declare_parameter('cpu_temp_rate_hz', 1.0).value)
        self._temp_sign = 1.0

        # ---- Wi-Fi params ----
        self._wifi_ssid = self.declare_parameter('wifi_ssid', 'JaxNet-5G').value
        self._wifi_bars = int(self.declare_parameter('wifi_bars', 3).value)
        self._wifi_randomize = self.declare_parameter('wifi_randomize', True).value
        self._wifi_rate_hz = float(
            self.declare_parameter('wifi_rate_hz', 0.5).value)

        # ---- Camera heartbeat params ----
        self._cam_rate_hz = float(
            self.declare_parameter('cam_heartbeat_rate_hz', 1.0).value)
        self._cam_topic = self.declare_parameter(
            'cam_topic', '/camera/image_raw').value

        # ---- Publishers ----
        self._elec_pub = self.create_publisher(
            ElectricalMeasurements, '/electrical_measurements', 10)
        self._battery_pub = self.create_publisher(
            BatteryState, '/battery_state', 10)
        self._temp_pub = self.create_publisher(
            Temperature, '/cpu_temperature', 10)
        self._wifi_pub = self.create_publisher(
            String, '/jax/wifi_status', 10)
        self._cam_pub = self.create_publisher(
            Image, self._cam_topic, 10)

        # ---- Timers ----
        self.create_timer(1.0 / self._battery_rate_hz, self._publish_battery)
        self.create_timer(1.0 / self._cpu_temp_rate_hz, self._publish_cpu_temp)
        self.create_timer(1.0 / self._wifi_rate_hz, self._publish_wifi)
        self.create_timer(1.0 / self._cam_rate_hz, self._publish_cam_heartbeat)

        self.get_logger().info(
            f'Mock peripherals active — battery={self._battery_voltage:.1f}V, '
            f'temp={self._cpu_temp_c:.0f}C, wifi={self._wifi_ssid}')

    # ---- Battery ----
    def _publish_battery(self):
        elec = ElectricalMeasurements()
        elec.battery_voltage_level = float(self._battery_voltage)
        elec.servo_buck_voltage_level = float(self._servo_buck_voltage)
        self._elec_pub.publish(elec)

        bs = BatteryState()
        bs.voltage = float(self._battery_voltage)
        bs.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        bs.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        bs.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        bs.present = True
        denom = 16.8 - self._min_battery_voltage
        if denom > 0.0:
            bs.percentage = float(max(0.0, min(1.0,
                (self._battery_voltage - self._min_battery_voltage) / denom)))
        self._battery_pub.publish(bs)

        if self._drain_per_second > 0.0:
            self._battery_voltage = max(
                self._min_battery_voltage,
                self._battery_voltage - self._drain_per_second / self._battery_rate_hz)

    # ---- CPU temperature ----
    def _publish_cpu_temp(self):
        msg = Temperature()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'cpu'
        msg.variance = self._cpu_temp_variance_c ** 2
        msg.temperature = self._cpu_temp_c + (self._temp_sign * self._cpu_temp_variance_c)
        self._temp_sign *= -1.0
        self._temp_pub.publish(msg)

    # ---- Wi-Fi ----
    def _publish_wifi(self):
        if self._wifi_randomize:
            self._wifi_bars = max(1, min(4,
                self._wifi_bars + random.choice([-1, 0, 0, 0, 1])))
        msg = String()
        msg.data = f'{self._wifi_ssid}|{self._wifi_bars}'
        self._wifi_pub.publish(msg)

    # ---- Camera heartbeat (empty image just to signal "camera alive") ----
    def _publish_cam_heartbeat(self):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height = 1
        msg.width = 1
        msg.encoding = 'rgb8'
        msg.step = 3
        msg.data = bytes([0, 0, 0])
        self._cam_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockPeripherals()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
