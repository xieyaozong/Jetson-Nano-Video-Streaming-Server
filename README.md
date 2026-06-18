# Jetson Nano Video Streaming Server

Real-time video streaming for Jetson Nano and other Jetson edge devices, with an optional AI inference hook for YOLO/TensorRT-style processing.

This project is built around the practical edge pipeline: get frames from a USB camera, RTSP stream, or video file; keep the stream responsive; expose a browser-friendly MJPEG feed; and leave a clean place to attach AI inference without rewriting the streaming stack.

![Jetson stream with AI inference](docs/assets/ai-stream-preview.jpg)

## What It Does

- Serves live camera video to any browser over MJPEG
- Supports USB camera, RTSP stream, and video file input
- Shows live FPS, stream FPS, inference state, and detection count overlays
- Exposes `/video.mjpg`, `/snapshot.jpg`, and `/status.json`
- Includes an optional YOLO inference hook
- Keeps the inference path modular for future TensorRT integration
- Includes Docker, systemd, camera check, and FPS benchmark helpers

## Pipeline

```text
USB Camera / RTSP / Video File
        |
        v
Jetson Nano / Edge Device
        |
        v
Frame Capture
        |
        v
Optional Processing / AI Inference Hook
        |
        v
MJPEG Streaming Server
        |
        v
Browser / Client Viewer
```

![Streaming pipeline](diagrams/streaming_pipeline.png)

## Repository Layout

```text
server/
  main.py                         # HTTP/MJPEG server entry point
  config.py                       # CLI and environment config
  camera/                         # USB, RTSP, and video-file sources
  streaming/                      # frame buffer and JPEG encoding
  processing/                     # overlay and inference hook
  utils/                          # FPS counter, logging, device stats

client/
  simple_viewer.html              # browser viewer
  viewer.py                       # opens the stream URL

scripts/
  check_camera.py                 # verify camera mode
  benchmark_fps.py                # measure capture FPS
  run_local.sh
  run_docker.sh
  install_systemd_service.sh

docs/
  architecture.md
  jetson_setup.md
  docker_on_jetson.md
  latency_test.md
  troubleshooting.md
  future_ai_inference_integration.md
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
python scripts/check_camera.py \
  --source 0 \
  --backend v4l2 \
  --width 1280 \
  --height 720 \
  --fourcc auto
```

Start the stream:

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

Open this from the Jetson or another device on the same network:

```text
http://<jetson-ip>:8080/
```

## Endpoints

```text
/             Browser viewer
/video.mjpg   MJPEG stream
/snapshot.jpg Latest encoded frame
/status.json  Stream FPS, source FPS, clients, inference state
```

Example:

```bash
curl http://127.0.0.1:8080/status.json
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

## AI Inference Hook

Inference is disabled by default so the base server stays lightweight.

Enable YOLO inference:

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

The hook lives in:

```text
server/processing/inference_hook.py
```

That file is the intended integration point for future TensorRT engines, custom detection models, tracking, alerts, or telemetry.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The Compose setup uses host networking and maps `/dev/video0` for USB camera access.

## FPS Benchmark

Measure raw camera capture FPS:

```bash
python scripts/benchmark_fps.py \
  --source 0 \
  --backend v4l2 \
  --width 1280 \
  --height 720 \
  --fourcc auto
```

## Systemd

Edit `systemd/jetson-streaming-server.service` for your install path, then:

```bash
sudo ./scripts/install_systemd_service.sh
sudo systemctl start jetson-streaming-server
sudo systemctl status jetson-streaming-server
```

## Notes For Jetson

- Use JetPack's system OpenCV when possible.
- Many USB cameras need MJPG for 720p 30 FPS.
- If YUYV is slow at 720p, try `--fourcc MJPG` or `--fourcc auto`.
- Some MJPG cameras print JPEG decode warnings while frames still display normally.

## Private Files

The repository ignores local runtime output and private artifacts:

- `captures/`
- `reports/`
- `logs/`
- `models/`
- `benchmarks/`
- `private*`
- `.env`
- large sample videos under `sample_data/`

