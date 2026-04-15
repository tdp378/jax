#!/usr/bin/env python3

import math
import socket
import subprocess
import time
import statistics
import os
import platform
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState, Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from luma.core.interface.serial import spi
    from luma.lcd.device import st7789
    from PIL import Image as PILImage
    _HAS_LUMA = True
except ImportError:
    _HAS_LUMA = False

@dataclass
class DisplayState:
    mode: str = "BOOT"
    battery_percent: int = 100
    battery_voltage: float = 16.8
    wifi_text: str = "N/A"
    wifi_bars: int = 0
    imu_ok: bool = False
    ros_ok: bool = True
    cam_ok: bool = False
    cpu_temp_c: float = 0.0
    sim: bool = True
    status_text: str = "STARTING"

class DesktopDisplayBackend:
    def __init__(self, window_name="ApeX-1 LCD Preview", scale=3):
        self.window_name = window_name
        self.scale = scale
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        preview = cv2.resize(img_bgr, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.window_name, preview)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class WaveshareDisplayBackend:
    def __init__(self, spi_port=0, spi_device=0, dc_pin=25, rst_pin=27):
        serial = spi(port=spi_port, device=spi_device, gpio_DC=dc_pin, gpio_RST=rst_pin)
        self.v_offset = 34
        self.device = st7789(serial, width=320, height=172, rotate=0)

    def show(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        self.device.set_window(0, self.v_offset, self.device._w, self.v_offset + self.device._h)
        buf = list(pil_img.convert("RGB").tobytes())
        self.device.data(buf)

    def close(self):
        pass


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "No Network"


class JaxDisplayNode(Node):
    def __init__(self):
        super().__init__('jax_display_node')

        # ---------------- Parameters ----------------
        self.declare_parameter("mode_topic", "/jax_mode")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("battery_state_topic", "/battery_state")
        self.declare_parameter("cam_topic", "/camera/image_raw")
        self.declare_parameter("sim", True)
        self.declare_parameter("battery_voltage_full", 16.8)
        self.declare_parameter("battery_voltage_empty", 13.6)
        self.declare_parameter("low_battery_threshold", 14.0)
        self.declare_parameter("refresh_hz", 10.0)
        self.declare_parameter("robot_name", "JAX 1.0")
        self.declare_parameter("boot_duration", 2.5)
        self.declare_parameter("mode_flash_duration", 1.2)
        self.declare_parameter("use_lcd", False)
        self.declare_parameter("spi_port", 0)
        self.declare_parameter("spi_device", 0)
        self.declare_parameter("dc_pin", 25)
        self.declare_parameter("rst_pin", 27)
        self.declare_parameter("publish_image_topic", True)

        self.v_full = self.get_parameter("battery_voltage_full").value
        self.v_empty = self.get_parameter("battery_voltage_empty").value
        self.v_low = self.get_parameter("low_battery_threshold").value
        self.robot_name = self.get_parameter("robot_name").value
        self.boot_duration = self.get_parameter("boot_duration").value
        self.mode_flash_duration = self.get_parameter("mode_flash_duration").value
        cam_topic = self.get_parameter("cam_topic").value

        # ---------------- State ----------------
        self.state = DisplayState()
        self.state.sim = self.get_parameter("sim").value
        self.v_history = []
        self.v_window = 50
        self.v_calibration = 0.781
        self.v_jump_threshold = 1.2

        self.last_imu_time = 0.0
        self.last_cam_time = 0.0
        self.start_time = time.time()
        self.flash_mode = None
        self.flash_until = 0.0
        self.local_ip = _get_local_ip()

        # ---------------- Subscribers ----------------
        self.create_subscription(
            String, self.get_parameter("mode_topic").value, self.mode_cb, 10)
        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self.imu_cb, 10)
        self.create_subscription(
            BatteryState, self.get_parameter("battery_state_topic").value, self.battery_cb, 10)
        self.create_subscription(
            String, '/jax/wifi_status', self.wifi_status_cb, 10)
        self.create_subscription(
            Image, cam_topic, self.cam_cb, 10)

        # ---------------- Image topic publisher ----------------
        self.bridge = CvBridge()
        self.image_pub = None
        if self.get_parameter("publish_image_topic").value:
            self.image_pub = self.create_publisher(Image, '/jax/display', 10)

        # ---------------- Backend ----------------
        on_pi = (os.path.exists("/dev/spidev0.0")
                 and platform.machine() in ["aarch64", "armv7l"])
        use_lcd = self.get_parameter("use_lcd").value

        if (use_lcd or on_pi) and _HAS_LUMA:
            self.backend = WaveshareDisplayBackend(
                spi_port=self.get_parameter("spi_port").value,
                spi_device=self.get_parameter("spi_device").value,
                dc_pin=self.get_parameter("dc_pin").value,
                rst_pin=self.get_parameter("rst_pin").value,
            )
        else:
            self.backend = DesktopDisplayBackend()

        self.timer = self.create_timer(
            1.0 / self.get_parameter("refresh_hz").value, self.update)

    def mode_cb(self, msg: String):
        new_mode = msg.data.strip().upper()
        if new_mode != self.state.mode and new_mode != "BOOT":
            self.flash_mode = new_mode
            self.flash_until = time.time() + self.mode_flash_duration
        self.state.mode = new_mode

    def imu_cb(self, msg: Imu):
        self.last_imu_time = time.time()
        self.state.imu_ok = True

    def cam_cb(self, msg: Image):
        self.last_cam_time = time.time()
        self.state.cam_ok = True

    def battery_cb(self, msg: BatteryState):
        raw_v = msg.voltage
        if abs(raw_v - self.state.battery_voltage) > self.v_jump_threshold:
            self.v_history = [raw_v]
        else:
            self.v_history.append(raw_v)
            if len(self.v_history) > self.v_window:
                self.v_history.pop(0)

        if self.v_history:
            self.state.battery_voltage = statistics.median(self.v_history)
        pct = 100.0 * (self.state.battery_voltage - self.v_empty) / (self.v_full - self.v_empty)
        self.state.battery_percent = int(max(0.0, min(100.0, pct)))

    def wifi_status_cb(self, msg: String):
        try:
            ssid, bars = msg.data.split("|", 1)
            self.state.wifi_text, self.state.wifi_bars = ssid, int(bars)
        except Exception:
            self.state.wifi_text = "ERROR"

    def update(self):
        now = time.time()
        self.state.imu_ok = (now - self.last_imu_time) < 1.0
        self.state.cam_ok = (now - self.last_cam_time) < 2.0
        self.state.ros_ok = rclpy.ok()

        if (now - self.start_time) < self.boot_duration:
            img = self.render_boot_screen(now - self.start_time)
        elif self.state.battery_voltage <= self.v_low:
            img = self.render_low_battery_warning(self.state.battery_voltage, now)
        elif self.flash_mode is not None and now < self.flash_until:
            img = self.render_mode_flash(self.flash_mode, self.flash_until - now)
        else:
            img = self.render_dashboard(self.state)

        self.backend.show(img)

        if self.image_pub is not None:
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.image_pub.publish(ros_img)

    def draw_led(self, img, x, y, color, label):
        dim_color = tuple(int(c * 0.3) for c in color)
        cv2.circle(img, (x, y), 6, dim_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x - 12, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)

    def render_low_battery_warning(self, voltage, now) -> np.ndarray:
        width, height = 320, 172
        img = np.zeros((height, width, 3), dtype=np.uint8)

        flash = int(now * 2) % 2 == 0
        bg_color = (0, 0, 180) if flash else (0, 0, 60)
        img[:] = bg_color

        cv2.putText(img, "LOW BATTERY", (35, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"VOLTAGE: {voltage:.2f}V", (65, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "CHARGE IMMEDIATELY", (75, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    def render_dashboard(self, s: DisplayState) -> np.ndarray:
        width, height = 320, 172
        img = np.zeros((height, width, 3), dtype=np.uint8)
        BG_BLACK = (10, 5, 5)
        CYAN = (255, 230, 100)
        BLUE = (255, 100, 0)
        WHITE = (240, 240, 240)
        GRAY = (40, 40, 40)
        L_GRN, L_RED = (0, 255, 0), (0, 0, 255)

        img[:] = BG_BLACK

        # Battery bar
        bx, by, bw, bh = 25, 20, 55, 110
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), CYAN, 2)
        fill_h = int((bh - 8) * (s.battery_percent / 100.0))
        for i in range(fill_h):
            rel = i / (bh - 8)
            col = tuple(int(BLUE[j] + (CYAN[j] - BLUE[j]) * rel) for j in range(3))
            cv2.line(img, (bx + 4, by + bh - 4 - i), (bx + bw - 4, by + bh - 4 - i), col, 1)
        cv2.putText(img, f"{s.battery_voltage:.1f}V", (bx + 2, by + bh + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

        # WiFi & IP
        cv2.putText(img, s.wifi_text[:12], (100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        for i in range(4):
            bar_x, bar_h = 265 + (i * 8), 6 + (i * 4)
            cv2.rectangle(img, (bar_x, 38 - bar_h), (bar_x + 5, 38),
                          CYAN if i < s.wifi_bars else GRAY, -1)
        cv2.putText(img, self.local_ip, (100, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)

        # LED indicators
        ly = 92
        self.draw_led(img, 120, ly, L_GRN if s.imu_ok else L_RED, "IMU")
        self.draw_led(img, 165, ly, L_GRN if s.ros_ok else L_RED, "ROS")
        self.draw_led(img, 210, ly, L_GRN if s.cam_ok else L_RED, "CAM")

        # Mode box
        cv2.rectangle(img, (100, 125), (305, 158), BLUE, -1)
        cv2.putText(img, f"MODE: {s.mode}", (115, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
        return img

    def render_boot_screen(self, t):
        img = np.zeros((172, 320, 3), dtype=np.uint8)
        img[:] = (10, 10, 10)
        cv2.putText(img, self.robot_name, (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

    def render_mode_flash(self, mode, remaining):
        img = np.zeros((172, 320, 3), dtype=np.uint8)
        img[:] = (255, 100, 0)
        cv2.putText(img, mode, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        return img

    def destroy_node(self):
        self.backend.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JaxDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
