# Docker On Jetson

Build and run:

```bash
cp .env.example .env
docker compose up --build
```

The Compose file uses:

- `network_mode: host` so clients can reach the stream directly.
- `privileged: true` and `/dev/video0` mapping for USB camera access.

For production deployments, pin the base image to the JetPack/L4T version that matches the device.

