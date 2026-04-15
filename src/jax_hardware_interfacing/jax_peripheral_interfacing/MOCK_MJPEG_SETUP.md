# Mock MJPEG Server Setup

The simulation now includes a mock MJPEG video server that allows you to test your app's video UI logic without needing real camera hardware or a live video stream.

## Quick Start

Launch the simulation with default settings:

```bash
ros2 launch jax_gazebo simulation.launch.py
```

By default, the mock MJPEG server will start on **`http://localhost:8081/video`**.

## Using in Your App

In your WebView component, set the `videoUrl` to:

```
http://localhost:8081/video
```

The server will stream placeholder JPEG frames at 30 FPS. Your app will receive a continuous video stream that it can render in the WebView, allowing you to test:
- Video UI layout and rendering
- App logic when video is present
- Stream interruption handling
- Video performance on your device

## Customization

You can customize the mock server parameters when launching:

```bash
# Change video resolution and frame rate
ros2 launch jax_gazebo simulation.launch.py \
  mjpeg_server_frame_width:=1280 \
  mjpeg_server_frame_height:=720 \
  mjpeg_server_frame_rate:=60

# Change the server port
ros2 launch jax_gazebo simulation.launch.py \
  mjpeg_server_port:=8082

# Disable the server entirely
ros2 launch jax_gazebo simulation.launch.py \
  use_mjpeg_server:=0
```

## Server Details

- **Default Port:** 8081
- **Default Resolution:** 640×480
- **Default FPS:** 30
- **Endpoint:** `/video`
- **Format:** MJPEG stream (multipart/x-mixed-replace)
- **Frame Source:** Generated placeholder images with timestamp overlays

## Testing the Server

You can test the server is working with curl:

```bash
# Get headers only
curl -I http://localhost:8081/video

# Download a single frame
curl -s http://localhost:8081/video --output frame.jpg --limit-output --max-time 0.1
```

Or use ffmpeg to verify the stream:

```bash
ffmpeg -i http://localhost:8081/video -vframes 1 -f image2 frame.jpg
```

## How It Works

The mock server:
1. Generates placeholder images with gradients and frame timing information
2. Encodes them as JPEG frames at the configured frame rate
3. Streams them in MJPEG format (multipart over HTTP)
4. Allows your app's WebView to continuously fetch and display frames

This allows full testing of your app's video UI integration without needing:
- A working camera
- A real mjpeg_server instance
- Actual video hardware

## App-Side Integration Example

In your React Native or Flutter app, your WebView might look like:

```javascript
// React Native Example
<WebView
  source={{ uri: 'http://localhost:8081/video' }}
  style={{ width: 300, height: 225 }}
/>

// Or in HTML/JavaScript
<img src="http://localhost:8081/video" />
```

The mock server will handle the MJPEG stream transparently.
