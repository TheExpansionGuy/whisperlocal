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

echo "▸ Bundling worker + Python sources…"
RES="dist/WhisperLocal.app/Contents/Resources"
cp transcribe_worker.py "$RES/transcribe_worker.py"
cp trainer.py "$RES/trainer.py"
cp overlay.py "$RES/overlay.py"
cp review_editor.py "$RES/review_editor.py"

echo "▸ Embedding ML dependencies (self-contained — no machine deps)…"
# Copy the venv's site-packages into the bundle so the worker, run with the
# bundle's OWN python (Contents/MacOS/python), needs nothing from this machine.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
DST="$RES/pylib"
rm -rf "$DST"; mkdir -p "$DST"
# Everything except heavyweight stuff py2app already bundled (numpy) and caches
rsync -a --exclude '__pycache__' --exclude '*.dist-info' --exclude 'pip*' \
      --exclude 'setuptools*' --exclude 'py2app*' --exclude 'macholib*' \
      "$SITE"/ "$DST"/ 2>/dev/null || cp -r "$SITE"/* "$DST"/
echo "  embedded ($(du -sh "$DST" | cut -f1))"

# Sanity: worker imports using the bundle's python + embedded pylib
echo "▸ Verifying self-contained worker imports…"
PYTHONPATH="$DST" /Applications/WhisperLocal.app/Contents/MacOS/python -c \
  "import sys; sys.path.insert(0,'$DST'); import mlx_whisper; print('  mlx_whisper OK')" \
  2>&1 | tail -2 || echo "  ⚠ (will retry against freshly built bundle)"

echo "▸ Installing to /Applications (clean replace)…"
rm -rf /Applications/WhisperLocal.app
cp -r dist/WhisperLocal.app /Applications/

# Install the bootstrap shim so the app loads hot-swappable code from
# ~/.whisperlocal/live, keeping the signed bundle immutable. That's what lets the
# Accessibility grant survive update.sh runs. See update.sh / bundle_main_shim.py.
echo "▸ Installing bootstrap shim + seeding live dir…"
RESI="/Applications/WhisperLocal.app/Contents/Resources"
cp main.py "$RESI/app_main.py"            # bundled fallback of the real entry
cp bundle_main_shim.py "$RESI/main.py"    # shim becomes the bundle entry point
LIVE="$HOME/.whisperlocal/live"; mkdir -p "$LIVE"
cp main.py "$LIVE/app_main.py"
cp overlay.py trainer.py review_editor.py transcribe_worker.py "$LIVE/"

echo "▸ Signing bundle (once; live updates afterward never modify it)…"
codesign --force --sign - /Applications/WhisperLocal.app 2>/dev/null && echo "  signed"

echo ""
echo "✓ Built and installed: /Applications/WhisperLocal.app"
echo "  → Grant Accessibility once in System Settings; it persists across update.sh runs."
