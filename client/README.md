# Client Viewer

The server exposes a browser page at `/`, so a separate client is optional.

- `simple_viewer.html` can point at any MJPEG endpoint.
- `viewer.py` opens the server page in the default browser.

```bash
python client/viewer.py --host 127.0.0.1 --port 8080
```

