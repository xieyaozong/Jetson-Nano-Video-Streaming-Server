# Jetson Nano Video Streaming Server

Real-time video streaming server for Jetson Nano and other Jetson edge devices. The project focuses on a reliable camera-to-browser pipeline first, with a clean optional hook for future AI inference such as YOLO or TensorRT.

![Streaming pipeline](diagrams/streaming_pipeline.png)

## Pipeline

```text
USB Camera / RTSP / Video File
        |
        v
Jetson Nano
        |
        v
Frame Capture
        |
        v
Optional Processing / Inference Hook
        |
        v
Streaming Server
        |
        v
Browser / Client Viewer
```

## Features

- USB camera, RTSP stream, or video file input
- OpenCV capture path with V4L2/GStreamer backend selection
- MJPEG browser streaming
- `/status.json` endpoint with FPS and client count
- `/snapshot.jpg` endpoint for the latest frame
- FPS overlay and optional inference overlay
- Docker and Docker Compose setup for Jetson-style deployment
- Optional systemd service template
- Future AI hook for YOLO/TensorRT integration

## Project Layout

```text
jetson-nano-video-streaming-server/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── server/
│   ├── main.py
│   ├── config.py
│   ├── camera/
│   │   ├── camera_source.py
│   │   ├── usb_camera.py
│   │   ├── rtsp_source.py
│   │   └── video_file_source.py
│   ├── streaming/
│   │   ├── frame_buffer.py
│   │   ├── mjpeg_streamer.py
│   │   ├── rtsp_streamer.py
│   │   └── websocket_streamer.py
│   ├── processing/
│   │   ├── frame_preprocessor.py
│   │   ├── overlay.py
│   │   └── inference_hook.py
│   └── utils/
│       ├── fps_counter.py
│       ├── device_monitor.py
│       └── logger.py
├── client/
│   ├── simple_viewer.html
│   ├── viewer.py
│   └── README.md
├── scripts/
│   ├── run_local.sh
│   ├── run_docker.sh
│   ├── check_camera.py
│   ├── benchmark_fps.py
│   └── install_systemd_service.sh
├── systemd/
│   └── jetson-streaming-server.service
├── docs/
│   ├── architecture.md
│   ├── jetson_setup.md
│   ├── docker_on_jetson.md
│   ├── latency_test.md
│   ├── troubleshooting.md
│   └── future_ai_inference_integration.md
├── sample_data/
│   ├── README.md
│   └── .gitkeep
└── diagrams/
    └── streaming_pipeline.png
```

## Quick Start

Create the environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the camera:

```bash
python scripts/check_camera.py --source 0 --backend v4l2 --width 1280 --height 720 --fourcc auto
```

Start the server:

```bash
python -m server.main \
  --source-type usb \
  --source 0 \
  --backend v4l2 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc auto
```

Open:

```text
http://<jetson-ip>:8080/
```

Direct endpoints:

```text
http://<jetson-ip>:8080/video.mjpg
http://<jetson-ip>:8080/snapshot.jpg
http://<jetson-ip>:8080/status.json
```

## Input Sources

USB camera:

```bash
python -m server.main --source-type usb --source 0 --backend v4l2
```

RTSP stream:

```bash
python -m server.main --source-type rtsp --source rtsp://user:pass@camera.local/stream1
```

Video file:

```bash
python -m server.main --source-type file --source sample_data/sample.mp4
```

## Optional AI Inference Hook

Inference is disabled by default. Enable it when a model is available:

```bash
python -m server.main \
  --source-type usb \
  --source 0 \
  --enable-inference \
  --model yolo11n.pt \
  --device auto \
  --imgsz 640 \
  --conf 0.25
```

The hook is intentionally isolated in `server/processing/inference_hook.py` so it can later be replaced with TensorRT engine inference.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The Compose setup uses host networking and maps `/dev/video0` for USB camera access.

## FPS Benchmark

Measure raw capture FPS:

```bash
python scripts/benchmark_fps.py --source 0 --backend v4l2 --width 1280 --height 720 --fourcc auto
```

Check live stream status:

```bash
curl http://127.0.0.1:8080/status.json
```

## Systemd

Edit `systemd/jetson-streaming-server.service` for your install path, then:

```bash
sudo ./scripts/install_systemd_service.sh
sudo systemctl start jetson-streaming-server
sudo systemctl status jetson-streaming-server
```

## GitHub Safety

Private runtime files are ignored:

- `captures/`
- `reports/`
- `logs/`
- `models/`
- `benchmarks/`
- private local folders
- large sample videos under `sample_data/`

Before pushing, verify:

```bash
git add --dry-run .
```
