# Jetson Setup

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip v4l-utils
```

Create the Python environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check camera modes:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
python scripts/check_camera.py --source 0 --backend v4l2
```

Start the server:

```bash
python -m server.main --source-type usb --source 0 --backend v4l2 --fourcc auto
```

