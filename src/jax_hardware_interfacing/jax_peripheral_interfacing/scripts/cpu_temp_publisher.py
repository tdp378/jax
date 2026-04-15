#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Temperature


class CpuTempPublisher(Node):
    def __init__(self):
        super().__init__('cpu_temp_publisher')

        self._topic = self.declare_parameter('cpu_temp_topic', '/cpu_temperature').value
        self._temp_c = float(self.declare_parameter('cpu_temp_c', 54.0).value)
        self._variance_c = float(self.declare_parameter('cpu_temp_variance_c', 0.0).value)
        self._rate_hz = float(self.declare_parameter('publish_rate_hz', 1.0).value)

        if self._rate_hz <= 0.0:
            self._rate_hz = 1.0

        self._pub = self.create_publisher(Temperature, self._topic, 10)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish)
        self._sign = 1.0

        self.get_logger().info(
            f'CPU temp publisher active on {self._topic} '
            f'(temp={self._temp_c:.2f} C, variance={self._variance_c:.2f} C)'
        )

    def _publish(self):
        msg = Temperature()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'cpu'
        msg.variance = self._variance_c * self._variance_c
        msg.temperature = self._temp_c + (self._sign * self._variance_c)
        self._sign *= -1.0
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CpuTempPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()