#!/bin/bash
# Fast local update — copies Python files into the installed app without rebuilding.
# No permission reset, no reinstall. Just restart.
set -e

APP="/Applications/WhisperLocal.app"
RES="$APP/Contents/Resources"

if [ ! -d "$APP" ]; then
  echo "WhisperLocal.app not found in /Applications — run build.sh first."
  exit 1
fi

echo "▸ Updating Python sources..."
cp main.py    "$RES/"
cp overlay.py "$RES/"

echo "▸ Restarting WhisperLocal..."
pkill -x WhisperLocal 2>/dev/null || true
sleep 0.4
open "$APP"

echo "✓ Done — WhisperLocal updated and restarted."
