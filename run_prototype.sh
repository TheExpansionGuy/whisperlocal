#!/bin/bash
# Run the sherpa-onnx streaming prototype from source (dev mode).
# Terminal needs Accessibility permission (System Settings → Privacy → Accessibility).
cd "$(dirname "$0")"
source .venv/bin/activate
exec python main.py
