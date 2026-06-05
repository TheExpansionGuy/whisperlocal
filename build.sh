#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "▸ Generating icon…"
python icon.py

echo "▸ Building app bundle…"
# Run py2app with a higher recursion limit to handle complex dependency graphs
python3 -c "
import sys
sys.setrecursionlimit(10000)
import runpy
sys.argv = ['setup.py', 'py2app']
runpy.run_path('setup.py', run_name='__main__')
" 2>&1 | grep -v "^zip\|^byte\|^copying\|^stripping\|^creating\|RecursionError\|return visitor\|self.visit\|File.*site-packages"

# Verify binary was created
if [ ! -f "dist/WhisperLocal.app/Contents/MacOS/WhisperLocal" ]; then
  echo "❌ Build failed — binary not created"
  exit 1
fi
echo "  binary OK"

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

echo "▸ Bundling transcription worker + venv reference…"
RES="dist/WhisperLocal.app/Contents/Resources"
cp transcribe_worker.py "$RES/transcribe_worker.py"
cp trainer.py "$RES/trainer.py"
cp overlay.py "$RES/overlay.py"
cp review_editor.py "$RES/review_editor.py"
python3 -c "import sys; print(sys.executable)" > "$RES/venv_python.txt"
echo "  venv: $(cat $RES/venv_python.txt)"

echo "▸ Installing to /Applications (clean replace)…"
rm -rf /Applications/WhisperLocal.app
cp -r dist/WhisperLocal.app /Applications/

echo ""
echo "✓ Built and installed: /Applications/WhisperLocal.app"
