#!/bin/bash
# Fast local update — copies Python sources into the LIVE hot-swap dir that the
# app actually loads from (~/.whisperlocal/live, via the bundle's bootstrap shim).
#
# The signed app bundle is deliberately NOT touched, so its code signature — and
# therefore the macOS Accessibility grant that lets the app paste — stays valid
# across every update. No re-signing, no re-granting. (This replaced the old
# approach of copying into the bundle, which broke the signature every time and
# silently killed paste.)
set -e
APP="/Applications/WhisperLocal.app"
LIVE="$HOME/.whisperlocal/live"
mkdir -p "$LIVE"

echo "▸ Updating live Python sources (bundle untouched, signature preserved)..."
cp main.py              "$LIVE/app_main.py"   # real entry, launched by the shim
cp overlay.py           "$LIVE/overlay.py"
cp trainer.py           "$LIVE/trainer.py"
cp review_editor.py     "$LIVE/review_editor.py"
cp transcribe_worker.py "$LIVE/transcribe_worker.py"

echo "▸ Restarting WhisperLocal..."
pkill -9 -x WhisperLocal 2>/dev/null || true
pkill -9 -f transcribe_worker 2>/dev/null || true
sleep 1
# Ensure no stragglers before relaunching (duplicate instances fight over the hotkey)
while pgrep -x WhisperLocal >/dev/null; do sleep 0.3; done
open "$APP"
echo "✓ Done (no re-sign needed — Accessibility grant preserved)"
