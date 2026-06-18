# Troubleshooting

## Camera opens but FPS is low

Check supported modes:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Many USB cameras cannot sustain 720p at 30 FPS with YUYV. Try MJPG:

```bash
python -m server.main --fourcc MJPG --width 1280 --height 720
```

## Browser shows no video

Check the direct stream endpoint:

```bash
curl -I http://127.0.0.1:8080/video.mjpg
```

Then check status:

```bash
curl http://127.0.0.1:8080/status.json
```

## MJPG decode warnings

Some USB cameras print `Corrupt JPEG data` warnings while frames still decode normally. If the stream looks correct, this is usually a camera transport warning rather than a server failure.

