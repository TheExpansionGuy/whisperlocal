#!/bin/bash
set -e

echo "▸ Creating virtual environment…"
python3 -m venv .venv
source .venv/bin/activate

echo "▸ Installing dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Grant Accessibility access to your terminal:"
echo "     System Settings → Privacy & Security → Accessibility"
echo ""
echo "To run in dev mode:"
echo "  source .venv/bin/activate && python main.py"
echo ""
echo "To build the .app:"
echo "  source .venv/bin/activate && ./build.sh"
echo ""
echo "Usage: Hold Right Option ⌥ to record, release to transcribe & type."
