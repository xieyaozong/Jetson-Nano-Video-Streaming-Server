# Latency And FPS Test

Record these numbers from the Jetson device that runs the camera. PC-side edits should update tables and notes only after the hardware run is complete.

Measure raw camera FPS:

```bash
python scripts/benchmark_fps.py --source 0 --backend v4l2 --width 1280 --height 720 --fourcc auto
```

Check stream status:

```bash
curl http://127.0.0.1:8080/status.json
```

Useful fields:

- `source_fps`: frame read rate from camera or stream.
- `encoded_fps`: JPEG encode and publish rate.
- `clients`: connected MJPEG clients.
- `inference_enabled`: whether the optional inference hook is active.
- `detections`: detections in the latest frame when inference is enabled.

