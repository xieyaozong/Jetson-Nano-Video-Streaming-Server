#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="jetson-streaming-server.service"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

sudo cp "$PROJECT_DIR/systemd/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "Installed $SERVICE_NAME"
echo "Start it with: sudo systemctl start $SERVICE_NAME"

