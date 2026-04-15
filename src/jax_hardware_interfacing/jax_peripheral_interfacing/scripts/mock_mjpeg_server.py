#!/usr/bin/env python3
"""
Mock MJPEG server for testing app camera integration.
Streams placeholder images so app can test video UI logic without real camera.
"""

import io
import time
import socket
import rclpy
from rclpy.node import Node
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MockMJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves MJPEG stream."""
    
    # Class variable shared across all instances
    frame_generator = None
    
    def do_GET(self):
        """Handle GET request for MJPEG stream."""
        if self.path == '/video':
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpegboundary')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Keep-Alive', 'timeout=5, max=100')
                self.end_headers()
                
                # Stream frames continuously with timeout protection
                frame_timeout = 10  # seconds
                frames_sent = 0
                
                while True:
                    try:
                        frame_data = self.frame_generator.get_frame()
                        
                        # Write frame with proper boundary
                        self.wfile.write('--jpegboundary\r\n'.encode())
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame_data)}\r\n'.encode())
                        self.wfile.write(b'Content-Transfer-Encoding: binary\r\n\r\n')
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        
                        frames_sent += 1
                        
                        # Rate limit to specified FPS
                        time.sleep(1.0 / self.frame_generator.frame_rate)
                        
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                    except Exception:
                        # Feed a fallback frame on error instead of dying
                        time.sleep(0.1)
                        continue
                        
            except Exception:
                pass
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'404 Not Found (use /video endpoint)')
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class FrameGenerator:
    """Generates placeholder MJPEG frames."""
    
    def __init__(self, width=640, height=480, frame_rate=30):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_interval = 1.0 / frame_rate
        self.last_frame_time = time.time()
    
    def get_frame(self):
        """Generate a placeholder JPEG frame efficiently."""
        try:
            # Create image with gradient using numpy for efficiency
            gradient_array = np.linspace(20, 120, self.height, dtype=np.uint8)
            img_array = np.tile(gradient_array, (self.width, 1)).T  # Transpose to get rows
            img_array = np.stack([img_array] * 3, axis=2)  # Convert to RGB
            
            # Create PIL image from array
            img = Image.fromarray(img_array, 'RGB')
            
            # Add text overlay
            draw = ImageDraw.Draw(img)
            elapsed = time.time() - self.start_time
            
            try:
                draw.text((10, 10), f"Frame: {self.frame_count}", fill=(0, 255, 0))
                draw.text((10, 30), f"Time: {elapsed:.1f}s", fill=(0, 255, 0))
            except Exception:
                pass  # Skip text if font fails
            
            # Convert to JPEG
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=75)
            jpeg_data = jpeg_buffer.getvalue()
            jpeg_buffer.close()
            
            self.frame_count = (self.frame_count + 1) % 1000000  # Reset after 1M frames
            return jpeg_data
            
        except Exception as e:
            # Return a minimal error frame instead of crashing
            error_img = Image.new('RGB', (self.width, self.height), color='red')
            jpeg_buffer = io.BytesIO()
            error_img.save(jpeg_buffer, format='JPEG', quality=50)
            jpeg_data = jpeg_buffer.getvalue()
            jpeg_buffer.close()
            return jpeg_data


class MockMJPEGServerNode(Node):
    """ROS 2 node that runs the mock MJPEG server."""
    
    def __init__(self):
        super().__init__('mock_mjpeg_server')
        
        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8081)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 30)
        
        # Get parameters
        host = self.get_parameter('host').value
        port = self.get_parameter('port').value
        width = self.get_parameter('frame_width').value
        height = self.get_parameter('frame_height').value
        fps = self.get_parameter('frame_rate').value
        
        # Create frame generator
        self.frame_gen = FrameGenerator(width=width, height=height, frame_rate=fps)
        MockMJPEGHandler.frame_generator = self.frame_gen
        
        # Create custom HTTPServer with SO_REUSEADDR enabled
        class ReusableHTTPServer(HTTPServer):
            def server_bind(self):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()
        
        # Start HTTP server in background thread
        # Use empty string for host to bind to all interfaces including external
        bind_host = '' if host == '0.0.0.0' else host
        self.server = ReusableHTTPServer((bind_host, port), MockMJPEGHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        self.get_logger().info(f'Mock MJPEG server started at http://0.0.0.0:{port}/video')
        self.get_logger().info(f'Connect from network: http://<your-ip>:{port}/video')
        self.get_logger().info(f'Resolution: {width}x{height}, FPS: {fps}')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = MockMJPEGServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
