# WhisperLocal bootstrap shim.
#
# This file is sealed inside the signed app bundle as Contents/Resources/main.py
# and MUST NOT change after the bundle is signed — its whole job is to never
# change, so the bundle's code signature (and therefore the macOS Accessibility
# grant that lets the app paste) stays stable across code updates.
#
# It loads the real, hot-swappable application code from ~/.whisperlocal/live
# (written by update.sh). If that live dir is missing, it falls back to the
# copy bundled alongside it (Contents/Resources/app_main.py).
import os
import sys
import runpy
from pathlib import Path

_live = Path.home() / ".whisperlocal" / "live"
_bundled = Path(__file__).resolve().parent          # the app's Resources dir
_src = _live if (_live / "app_main.py").is_file() else _bundled

sys.path.insert(0, str(_src))
os.environ["WHISPERLOCAL_SRC"] = str(_src)
runpy.run_path(str(_src / "app_main.py"), run_name="__main__")
