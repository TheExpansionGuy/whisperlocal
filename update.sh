#!/bin/bash
set -e
APP="/Applications/WhisperLocal.app"
RES="$APP/Contents/Resources"
ZIP=$(ls "$RES/lib/python"*.zip)

echo "▸ Updating main.py..."
cp main.py "$RES/main.py"

echo "▸ Compiling and injecting overlay.py into bundle zip..."
python3 - "$ZIP" <<'EOF'
import sys, zipfile, py_compile, tempfile, os, shutil

zip_path = sys.argv[1]
src = "overlay.py"

# Compile overlay.py to a temp pyc
tmp = tempfile.mktemp(suffix=".pyc")
py_compile.compile(src, tmp, doraise=True)

# Read the existing zip and rebuild it with updated overlay.pyc
tmp_zip = zip_path + ".tmp"
with zipfile.ZipFile(zip_path, "r") as zin, \
     zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename == "overlay.pyc":
            zout.write(tmp, "overlay.pyc")
        else:
            zout.writestr(item, zin.read(item.filename))

os.replace(tmp_zip, zip_path)
os.unlink(tmp)
print("  overlay.pyc updated in zip")
EOF

echo "▸ Restarting WhisperLocal..."
pkill -x WhisperLocal 2>/dev/null || true
sleep 0.5
open "$APP"
echo "✓ Done"
