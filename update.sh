#!/bin/bash
# Fast local update — copies Python sources into the installed app and restarts.
# Resources is on sys.path (see main.py), so plain .py files here are what load.
set -e
APP="/Applications/WhisperLocal.app"
RES="$APP/Contents/Resources"

echo "▸ Updating Python sources..."
cp main.py             "$RES/main.py"
cp overlay.py          "$RES/overlay.py"
cp trainer.py          "$RES/trainer.py"
cp review_editor.py    "$RES/review_editor.py"
cp transcribe_worker.py "$RES/transcribe_worker.py"

# Remove any stale compiled copy of overlay in the zip so the fresh .py wins
ZIP=$(ls "$RES/lib/python"*.zip 2>/dev/null || true)
if [ -n "$ZIP" ]; then
  python3 - "$ZIP" <<'EOF'
import sys, zipfile, os
zip_path = sys.argv[1]
tmp = zip_path + ".tmp"
removed = 0
with zipfile.ZipFile(zip_path) as zin, \
     zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename == "overlay.pyc":
            removed += 1
            continue
        zout.writestr(item, zin.read(item.filename))
os.replace(tmp, zip_path)
if removed:
    print(f"  removed stale overlay.pyc from zip")
EOF
fi

# Clear any cached bytecode
find "$RES" -name "overlay*.pyc" -not -path "*/lib/*" -delete 2>/dev/null || true

echo "▸ Restarting WhisperLocal..."
pkill -x WhisperLocal 2>/dev/null || true
sleep 0.5
open "$APP"
echo "✓ Done"
