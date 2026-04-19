#!/usr/bin/env python3
import signal
import subprocess
import sys
import time

import RPi.GPIO as GPIO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64


def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)


def shutdown(node: Node):
    GPIO.cleanup()
    node.get_logger().warn('BATTERY VOLTAGE TOO LOW. COMMENCING SHUTDOWN PROCESS')
    time.sleep(5)
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])


def main():
    rclpy.init()
    node = rclpy.create_node('battery_monitor')
    message_rate = 50.0
    period = 1.0 / message_rate

    signal.signal(signal.SIGINT, signal_handler)

    GPIO.setmode(GPIO.BCM)

    estop_pin_number = 5
    battery_pin1_number = 6
    battery_pin2_number = 13
    battery_pin3_number = 19

    GPIO.setup(estop_pin_number, GPIO.IN)
    GPIO.setup(battery_pin1_number, GPIO.IN)
    GPIO.setup(battery_pin2_number, GPIO.IN)
    GPIO.setup(battery_pin3_number, GPIO.IN)

    battery_percentage_publisher = node.create_publisher(Float64, '/battery_percentage', 10)
    # Add BatteryState publisher for LCD
    try:
        from sensor_msgs.msg import BatteryState
    except ImportError:
        print('sensor_msgs not found, battery state will not be published!')
        BatteryState = None
    battery_state_publisher = node.create_publisher(BatteryState, '/battery_state', 10) if BatteryState else None
    estop_publisher = node.create_publisher(Bool, '/emergency_stop_status', 10)
    current_estop_bit = 0

    number_of_low_battery_detections = 0

    estop_bit = GPIO.input(estop_pin_number)
    battery_bit1 = GPIO.input(battery_pin1_number)
    battery_bit2 = GPIO.input(battery_pin2_number)
    battery_bit3 = GPIO.input(battery_pin3_number)

    if estop_bit == 0:
        estop_publisher.publish(Bool(data=False))
    elif estop_bit == 1:
        estop_publisher.publish(Bool(data=True))
        current_estop_bit = 1

    try:
        while rclpy.ok():
            estop_bit = GPIO.input(estop_pin_number)
            battery_bit1 = GPIO.input(battery_pin1_number)
            battery_bit2 = GPIO.input(battery_pin2_number)
            battery_bit3 = GPIO.input(battery_pin3_number)
            print('estop: ', battery_bit1)
            print('bit1: ', battery_bit1)
            print('bit2: ', battery_bit2)
            print('bit3: ', battery_bit3)

            battery_bits = [battery_bit1, battery_bit2, battery_bit3]

            if estop_bit == 1 and current_estop_bit == 0:
                current_estop_bit = 1
                estop_publisher.publish(Bool(data=True))

            if estop_bit == 0 and current_estop_bit == 1:
                current_estop_bit = 0
                estop_publisher.publish(Bool(data=False))

            num = int(''.join([str(b) for b in battery_bits]), 2)

            value = 0.0

            if num == 0:
                value = 0.0
            elif num == 1:
                value = 0.125
            elif num == 2:
                value = 0.25
            elif num == 3:
                value = 0.375
            elif num == 4:
                value = 0.5
            elif num == 5:
                value = 0.625
            elif num == 6:
                value = 0.75
            elif num == 7:
                value = 1.0

            battery_percentage_publisher.publish(Float64(data=value))
            # Publish BatteryState for LCD
            if BatteryState and battery_state_publisher:
                bs = BatteryState()
                # Map value (0.0-1.0) to voltage (13.6V empty, 16.8V full)
                bs.voltage = 13.6 + (16.8 - 13.6) * value
                bs.percentage = value
                bs.present = True
                bs.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
                bs.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
                bs.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
                battery_state_publisher.publish(bs)

            if value == 0.0:
                number_of_low_battery_detections = number_of_low_battery_detections + 1
                if number_of_low_battery_detections > 30:
                    print('Would shut down if activated')
            else:
                if number_of_low_battery_detections > 0:
                    number_of_low_battery_detections = number_of_low_battery_detections - 1

            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    finally:
        GPIO.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
