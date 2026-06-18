# Architecture

The server is built as a small real-time video pipeline:

```text
USB Camera / RTSP / Video File
        |
        v
Jetson Nano / ARM64 Linux
        |
        v
Frame Capture
        |
        v
Optional Processing / Inference Hook
        |
        v
MJPEG Streaming Server
        |
        v
Browser / Client Viewer
```

Core modules:

- `server/camera/`: OpenCV camera, RTSP, and video file sources.
- `server/processing/`: overlay drawing and optional inference hook.
- `server/streaming/`: frame buffer and JPEG encoding.
- `server/main.py`: HTTP server exposing `/`, `/video.mjpg`, `/snapshot.jpg`, and `/status.json`.

The MVP uses MJPEG over HTTP because it is easy to inspect from any browser and simple to debug on embedded Linux.

