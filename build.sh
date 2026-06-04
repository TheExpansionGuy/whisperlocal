#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "▸ Generating icon…"
python icon.py

echo "▸ Building app bundle…"
python setup.py py2app 2>&1 | grep -v "^zip\|^byte\|^copying\|^stripping\|^creating"

echo "▸ Ensuring PortAudio dylib is not zipped…"
python3 - <<'EOF'
import zipfile, pathlib
resources = pathlib.Path("dist/WhisperLocal.app/Contents/Resources")
zip_path = next(resources.glob("lib/python*.zip"), None)
if zip_path:
    with zipfile.ZipFile(zip_path) as z:
        targets = [n for n in z.namelist() if "_sounddevice_data" in n]
    if targets:
        with zipfile.ZipFile(zip_path) as z:
            for name in targets:
                dest = resources / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(name))
        print(f"  extracted {len(targets)} sounddevice entries")
EOF

echo "▸ Embedding mlx packages (can't be bundled by py2app)…"
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
DST="dist/WhisperLocal.app/Contents/Resources/venv/lib/site-packages"
mkdir -p "$DST"
for pkg in mlx mlx_whisper tiktoken; do
  if [ -d "$SITE/$pkg" ]; then
    cp -r "$SITE/$pkg" "$DST/"
    echo "  copied $pkg"
  fi
done
# Also copy any mlx .dist-info for version metadata
for d in "$SITE"/mlx*.dist-info "$SITE"/mlx_whisper*.dist-info; do
  [ -d "$d" ] && cp -r "$d" "$DST/"
done
echo "  embedded ($(du -sh $DST | cut -f1))"

echo ""
echo "✓ Built: dist/WhisperLocal.app"
echo ""
echo "To install:"
echo "  cp -r dist/WhisperLocal.app /Applications/"
