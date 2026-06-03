import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyperclip
import rumps
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".whisperlocal" / "config.json"
DEFAULTS = {"model": "base.en", "filler_removal": True, "history": []}
HISTORY_MAX = 10

FILLER_RE = re.compile(
    r"\b(uh+|um+|hmm+|ah+|er+|like|you know)\b[,.]?\s*", re.IGNORECASE
)

SAMPLE_RATE = 16000
HOTKEY = keyboard.Key.alt_r
MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]

# Locate assets relative to this file (works both in dev and py2app bundle)
if getattr(sys, "frozen", False):
    _RESOURCES = Path(sys.executable).parent.parent / "Resources"
else:
    _RESOURCES = Path(__file__).parent

ICON_IDLE        = str(_RESOURCES / "assets" / "menubar.png")
ICON_RECORDING   = str(_RESOURCES / "assets" / "menubar_rec.png")
ICON_TRANSCRIBING = str(_RESOURCES / "assets" / "menubar_proc.png")

# Waveform: sliding window of RMS levels
WAVEFORM_BINS = 48
WAVEFORM_WINDOW = collections.deque([0.02] * WAVEFORM_BINS, maxlen=WAVEFORM_BINS)

PARTIAL_INTERVAL = 2.0
MAX_RECORD_SECS = 30  # auto-stop if user forgets to release


def load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return {**DEFAULTS, **data}
    except Exception:
        return dict(DEFAULTS)


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    CONFIG_PATH.chmod(0o600)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class WhisperLocal(rumps.App):
    def __init__(self):
        super().__init__("", quit_button=None)
        self.icon = ICON_IDLE
        self.template = True
        self.cfg = load_config()
        self.model = None
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self._kb = keyboard.Controller()
        self._model_lock = threading.Lock()
        self._partial_timer = None
        self._timeout_timer = None
        self._cancelled = False
        self._transcribing = False
        self._target_element = None   # AX element focused when hotkey was pressed
        self._target_app = None       # NSRunningApplication at that moment

        # Overlay (AppKit panel — created lazily on main thread)
        from overlay import OverlayPanel
        self._overlay = OverlayPanel.alloc().init()

        self._build_menu()
        self._request_accessibility()
        threading.Thread(target=self._load_model, daemon=True).start()
        self._start_listener()

    # ------------------------------------------------------------------
    # Accessibility permission
    # ------------------------------------------------------------------

    def _request_accessibility(self):
        """Prompt if not trusted, then watch in background and restart listener when granted."""
        try:
            import ApplicationServices as AS
            if AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: False}):
                return  # already trusted
            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
            rumps.notification(
                "WhisperLocal",
                "Accessibility required",
                "Grant access in System Settings → Privacy & Security → Accessibility.",
                sound=False,
            )
            threading.Thread(target=self._watch_accessibility, daemon=True).start()
        except Exception:
            pass

    def _watch_accessibility(self):
        """Poll until Accessibility is granted, then restart the listener — no relaunch needed."""
        import ApplicationServices as AS
        while True:
            time.sleep(1.5)
            try:
                if AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: False}):
                    self._start_listener()
                    rumps.notification("WhisperLocal", "Ready", "Hold ⌥ to start dictating.", sound=False)
                    return
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Menu construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        self.menu.clear()

        self._history_menu = rumps.MenuItem("Recent")
        self._populate_history(self._history_menu)
        self.menu.add(self._history_menu)
        self.menu.add(rumps.separator)

        model_item = rumps.MenuItem("Model")
        for m in MODELS:
            item = rumps.MenuItem(m, callback=self._set_model)
            item.state = int(m == self.cfg["model"])
            model_item.add(item)
        self.menu.add(model_item)

        filler_item = rumps.MenuItem("Remove Filler Words", callback=self._toggle_filler)
        filler_item.state = int(self.cfg["filler_removal"])
        self.menu.add(filler_item)

        mic_item = rumps.MenuItem("Microphone")
        self._build_mic_menu(mic_item)
        self.menu.add(mic_item)

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Check Accessibility…", callback=self._open_accessibility))
        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Quit", callback=self._quit))

    def _populate_history(self, menu):
        history = self.cfg.get("history", [])
        if not history:
            empty = rumps.MenuItem("No transcriptions yet")
            empty.set_callback(None)
            menu.add(empty)
        else:
            for text in reversed(history[-HISTORY_MAX:]):
                label = text if len(text) <= 60 else text[:57] + "…"
                item = rumps.MenuItem(label, callback=self._copy_history)
                item._full_text = text
                menu.add(item)
            menu.add(rumps.separator)
            menu.add(rumps.MenuItem("Clear History", callback=self._clear_history))

    def _refresh_history_menu(self):
        self._history_menu.clear()
        self._populate_history(self._history_menu)

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _set_model(self, sender):
        if sender.title == self.cfg["model"]:
            return
        for item in self.menu["Model"].values():
            item.state = int(item.title == sender.title)
        self.cfg["model"] = sender.title
        save_config(self.cfg)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _toggle_filler(self, sender):
        self.cfg["filler_removal"] = not self.cfg["filler_removal"]
        sender.state = int(self.cfg["filler_removal"])
        save_config(self.cfg)

    def _build_mic_menu(self, parent):
        parent.clear() if hasattr(parent, "_menu") and parent._menu else None
        try:
            devices = sd.query_devices()
            inputs = [(i, d["name"]) for i, d in enumerate(devices)
                      if d["max_input_channels"] > 0]
        except Exception:
            inputs = []

        saved = self.cfg.get("input_device")

        # Default option
        item = rumps.MenuItem("System Default", callback=self._set_mic)
        item._device_index = None
        item.state = int(saved is None)
        parent.add(item)

        for idx, name in inputs:
            label = name if len(name) <= 48 else name[:45] + "…"
            item = rumps.MenuItem(label, callback=self._set_mic)
            item._device_index = idx
            item.state = int(saved == idx)
            parent.add(item)

    def _set_mic(self, sender):
        self.cfg["input_device"] = sender._device_index
        save_config(self.cfg)
        for item in self.menu["Microphone"].values():
            if hasattr(item, "_device_index"):
                item.state = int(item._device_index == sender._device_index)
        # Stop any active recording so next press uses new device
        if self.recording:
            self._cancel()

    def _copy_history(self, sender):
        pyperclip.copy(sender._full_text)

    def _clear_history(self, _):
        self.cfg["history"] = []
        save_config(self.cfg)
        self._refresh_history_menu()

    def _open_accessibility(self, _):
        subprocess.run([
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ])

    def _quit(self, _):
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _set_state(self, state: str):
        icons = {"idle": ICON_IDLE, "recording": ICON_RECORDING, "transcribing": ICON_TRANSCRIBING}
        self.icon = icons.get(state, ICON_IDLE)

    def _load_model(self):
        self._set_state("transcribing")
        with self._model_lock:
            self.model = WhisperModel(
                self.cfg["model"], device="cpu", compute_type="int8"
            )
        self._set_state("idle")

    # ------------------------------------------------------------------
    # Hotkey listener
    # ------------------------------------------------------------------

    def _start_listener(self):
        if hasattr(self, "_listener") and self._listener and self._listener.is_alive():
            try:
                self._listener.stop()
            except Exception:
                pass
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        # Escape or second Option tap cancels
        if key == keyboard.Key.esc:
            self._cancel()
            return
        if key == HOTKEY and (self.recording or self._transcribing):
            self._cancel()
            return
        if key != HOTKEY or self.recording or self.model is None:
            return
        self._cancelled = False
        self._target_element, self._target_app = self._snapshot_focus()
        self.recording = True
        self.audio_chunks = []
        WAVEFORM_WINDOW.extend([0.02] * WAVEFORM_BINS)
        self._set_state("recording")
        self._overlay.show_async()
        self._overlay.push_state("recording")
        self.stream = sd.InputStream(
            device=self.cfg.get("input_device"),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_cb,
        )
        self.stream.start()
        self._start_partial_timer()
        self._start_timeout()

    def _on_release(self, key):
        if key != HOTKEY or not self.recording:
            return
        self._stop_timeout()
        self.recording = False
        self._stop_partial_timer()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._transcribing = True
        self._set_state("transcribing")
        self._overlay.push_state("transcribing")
        threading.Thread(target=self._transcribe_final, daemon=True).start()

    def _cancel(self):
        """Stop recording and hide overlay without transcribing."""
        if not self.recording and not self._transcribing:
            return
        self._cancelled = True
        self._stop_timeout()
        self._stop_partial_timer()
        self.recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.audio_chunks = []
        self._transcribing = False
        self._overlay.push_state("idle")
        self._overlay.hide_async()
        self._set_state("idle")

    def _audio_cb(self, indata, frames, t, status):
        if not self.recording:
            return
        chunk = indata.copy()
        self.audio_chunks.append(chunk)
        # RMS for waveform
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        WAVEFORM_WINDOW.append(min(rms * 6.0, 1.0))
        self._overlay.push_levels(list(WAVEFORM_WINDOW))

    # ------------------------------------------------------------------
    # Partial (live) transcription
    # ------------------------------------------------------------------

    def _start_partial_timer(self):
        self._partial_timer = threading.Timer(PARTIAL_INTERVAL, self._partial_tick)
        self._partial_timer.daemon = True
        self._partial_timer.start()

    def _stop_partial_timer(self):
        if self._partial_timer:
            self._partial_timer.cancel()
            self._partial_timer = None

    def _start_timeout(self):
        self._timeout_timer = threading.Timer(MAX_RECORD_SECS, self._on_release, [HOTKEY])
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _stop_timeout(self):
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _partial_tick(self):
        if not self.recording or not self.audio_chunks:
            return
        chunks = list(self.audio_chunks)
        threading.Thread(target=self._run_partial, args=(chunks,), daemon=True).start()
        # Reschedule
        self._partial_timer = threading.Timer(PARTIAL_INTERVAL, self._partial_tick)
        self._partial_timer.daemon = True
        self._partial_timer.start()

    def _run_partial(self, chunks):
        try:
            audio = np.concatenate(chunks).flatten()
            if len(audio) < SAMPLE_RATE * 0.5:
                return
            with self._model_lock:
                segs, _ = self.model.transcribe(
                    audio, language="en", vad_filter=True
                )
                text = " ".join(s.text for s in segs).strip()
            if text:
                self._overlay.push_text(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final transcription
    # ------------------------------------------------------------------

    def _transcribe_final(self):
        try:
            if self._cancelled or not self.audio_chunks:
                return
            audio = np.concatenate(self.audio_chunks).flatten()
            if len(audio) < SAMPLE_RATE * 0.3:
                return
            with self._model_lock:
                segments, _ = self.model.transcribe(
                    audio, language="en", vad_filter=True
                )
                text = " ".join(s.text for s in segments).strip()
            if self.cfg["filler_removal"]:
                text = FILLER_RE.sub("", text).strip()
            if not text:
                return
            self._overlay.push_text(text)
            time.sleep(0.8)   # show final text briefly before hiding
            self._add_history(text)
            self._paste(text)
            rumps.notification("WhisperLocal", None, text[:100], sound=False)
        finally:
            self._transcribing = False
            self._overlay.push_state("idle")
            self._overlay.hide_async()
            self._set_state("idle")

    def _add_history(self, text: str):
        history = self.cfg.setdefault("history", [])
        history.append(text)
        self.cfg["history"] = history[-HISTORY_MAX:]
        save_config(self.cfg)
        self._refresh_history_menu()

    # ------------------------------------------------------------------
    # Text injection
    # ------------------------------------------------------------------

    def _snapshot_focus(self):
        """Capture the focused AX element and frontmost app right now."""
        try:
            import ApplicationServices as AS
            from AppKit import NSWorkspace
            system = AS.AXUIElementCreateSystemWide()
            err, el = AS.AXUIElementCopyAttributeValue(
                system, AS.kAXFocusedUIElementAttribute, None
            )
            element = el if err == 0 else None
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return element, app
        except Exception:
            return None, None

    def _paste(self, text: str):
        """Always use clipboard paste. AX is attempted first as a bonus but never relied on."""
        self._ax_insert(text)  # best-effort, ignored if it fails or does nothing
        self._clipboard_paste(text)

    def _ax_insert(self, text: str) -> bool:
        try:
            import ApplicationServices as AS
            if self._target_element is None:
                return False
            AS.AXUIElementSetAttributeValue(
                self._target_element, AS.kAXSelectedTextAttribute, text
            )
        except Exception:
            pass
        return False  # always fall through to clipboard paste

    def _clipboard_paste(self, text: str):
        """Re-activate target app then paste from clipboard using pynput (Accessibility-backed)."""
        try:
            if self._target_app:
                self._target_app.activateWithOptions_(1 << 1)
                time.sleep(0.25)
        except Exception:
            pass

        prev = pyperclip.paste()
        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            with self._kb.pressed(keyboard.Key.cmd):
                self._kb.tap("v")
            time.sleep(0.15)
        finally:
            pyperclip.copy(prev)


if __name__ == "__main__":
    WhisperLocal().run()
