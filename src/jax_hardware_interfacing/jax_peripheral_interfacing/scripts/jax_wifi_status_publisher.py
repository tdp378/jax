#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import re


class WifiStatusPublisher(Node):
    def __init__(self):
        super().__init__('wifi_status_publisher')

        self.publisher_ = self.create_publisher(String, 'jax/wifi_status', 10)

        # Tunables
        self.declare_parameter('wifi_interface', 'wlan0')
        self.declare_parameter('publish_period', 2.0)

        self.iface = self.get_parameter('wifi_interface').get_parameter_value().string_value
        period = self.get_parameter('publish_period').get_parameter_value().double_value

        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(f"Publishing Wi-Fi status from interface: {self.iface}")

    def timer_callback(self):
        ssid, bars = self.get_wifi_status()
        msg = String()
        # Defensive: always send a valid string
        if not ssid or not isinstance(bars, int):
            ssid, bars = "NO WIFI", 0
        msg.data = f"{ssid}|{bars}"
        self.publisher_.publish(msg)

    def get_wifi_status(self):
        """
        Returns:
            (ssid: str, bars: int)
        """
        try:
            out = subprocess.check_output(
                ["iw", "dev", self.iface, "link"],
                universal_newlines=True,
                timeout=1.0
            )
        except Exception:
            return ("NO WIFI", 0)

        if "Not connected" in out:
            return ("NO WIFI", 0)

        ssid_match = re.search(r"SSID:\s*(.+)", out)
        signal_match = re.search(r"signal:\s*(-?\d+)\s*dBm", out)

        ssid = ssid_match.group(1).strip() if ssid_match else "WIFI"

        if signal_match:
            dbm = int(signal_match.group(1))

            # Simple dBm-to-bars mapping
            if dbm >= -55:
                bars = 4
            elif dbm >= -67:
                bars = 3
            elif dbm >= -75:
                bars = 2
            elif dbm >= -85:
                bars = 1
            else:
                bars = 0
        else:
            bars = 1

        return (ssid, bars)


def main(args=None):
    rclpy.init(args=args)
    node = WifiStatusPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()