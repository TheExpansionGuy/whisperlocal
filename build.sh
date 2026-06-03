#!/bin/bash
set -e

cd "$(dirname "$0")"

# Activate venv if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "▸ Generating icon…"
python icon.py

echo "▸ Building app bundle…"
python setup.py py2app 2>&1 | grep -v "^zip\|^byte\|^copying\|^stripping"

echo "▸ Ensuring PortAudio dylib is not zipped…"
python3 - <<'EOF'
import zipfile, os, pathlib

resources = pathlib.Path("dist/WhisperLocal.app/Contents/Resources")
zip_path = next(resources.glob("lib/python*.zip"), None)
if not zip_path:
    print("  no zip found, skipping")
    exit()

with zipfile.ZipFile(zip_path) as z:
    targets = [n for n in z.namelist() if "_sounddevice_data" in n]
    if not targets:
        print("  _sounddevice_data not in zip, all good")
        exit()
    print(f"  extracting {len(targets)} entries from zip…")
    for name in targets:
        dest = resources / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(name))
print("  done")
EOF

echo ""
echo "✓ Built: dist/WhisperLocal.app"
echo ""
echo "To install, drag it to /Applications:"
echo "  cp -r dist/WhisperLocal.app /Applications/"
echo ""
echo "Then launch from Spotlight or /Applications."
