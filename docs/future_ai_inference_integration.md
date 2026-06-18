# Future AI Inference Integration

The streaming server includes `server/processing/inference_hook.py`.

Current behavior:

- Inference is disabled by default.
- Passing `--enable-inference` loads the YOLO hook.
- Detections are drawn by `server/processing/overlay.py`.

Example:

```bash
python -m server.main \
  --enable-inference \
  --model yolo11n.pt \
  --device auto \
  --imgsz 640 \
  --conf 0.25
```

Possible next steps:

- Replace the YOLO hook with TensorRT engine inference.
- Run inference on a worker thread to keep camera capture smooth.
- Add class-specific routing for alerts or telemetry.
- Publish structured detections over WebSocket or MQTT.

