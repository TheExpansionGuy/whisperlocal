import collections
import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

# MLX transcription runs in a subprocess via the venv Python (avoids bundling issues)
import numpy as np
import pyperclip
import rumps
import sounddevice as sd
from AppKit import NSEvent, NSSound
from PyObjCTools import AppHelper

# Ensure our bundled Resources dir is importable (for trainer.py)
if getattr(sys, "frozen", False):
    _res = Path(sys.executable).parent.parent / "Resources"
    if str(_res) not in sys.path:
        sys.path.insert(0, str(_res))
else:
    sys.path.insert(0, str(Path(__file__).parent))

import trainer
from review_editor import ReviewEditor
from pynput import keyboard

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Minimal consumer build: hides experimental features (review mode, LLM cleanup,
# personalization/voice-training, low-power, model/mic pickers). The full
# experience lives on the 'experimental' git branch. Flip to False to re-enable.
MINIMAL = True

# Real-time streaming model (sherpa-onnx) selectable from the menu as an
# alternative to Whisper: instant, but lower accuracy + no punctuation.
_STREAM_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26"

def _streaming_model_dir() -> str:
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).parent.parent / "Resources" / "streaming-model")
    return str(Path(__file__).parent / "models" / _STREAM_MODEL_NAME)

CONFIG_PATH = Path.home() / ".whisperlocal" / "config.json"
DEFAULTS = {"model": "mlx-community/whisper-small.en-mlx", "filler_removal": True,
            "llm_cleanup": False, "low_power": False, "sounds": True,
            "personalize": True, "review": False, "engine": "whisper", "history": []}

LOW_POWER_MODEL = "mlx-community/whisper-base.en-mlx"  # lighter model when low-power on
HISTORY_MAX = 10

FILLER_RE = re.compile(
    r"\b(uh+|um+|hmm+|ah+|er+|like|you know)\b[,.]?\s*", re.IGNORECASE
)

SAMPLE_RATE = 16000
HOTKEY = keyboard.Key.alt_r
HOTKEY_ALT = keyboard.Key.alt  # catch both left and right Option
MODELS = {
    "Tiny   (fastest)":  "mlx-community/whisper-tiny.en-mlx",
    "Base   (default)":  "mlx-community/whisper-base.en-mlx",
    "Small  (better)":   "mlx-community/whisper-small.en-mlx",
    "Medium (best)":     "mlx-community/whisper-medium.en-mlx",
}

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

FIRST_CHUNK_DELAY = 0.5  # transcribe the first slice quickly so text appears fast
CHUNK_INTERVAL  = 0.8   # how often to re-transcribe the live tail
SETTLE_MARGIN   = 0.5   # keep committed point close to the live edge
MIN_TAIL_SECS   = 0.4   # minimum audio before a transcription pass
MAX_TAIL_SECS   = 4.5   # force-commit beyond this so passes stay fast + final pass is small
MAX_RECORD_SECS = 300


def _slog(msg: str):
    try:
        with open(Path.home() / ".whisperlocal" / "stream.log", "a") as f:
            f.write(f"{time.time():.1f} {msg}\n")
    except Exception:
        pass


def _collapse_repeats(text: str) -> str:
    """Collapse runs of 3+ identical consecutive words to a single one.
    Guards against Whisper's repetition hallucination ('the the the the')."""
    out = []
    for w in text.split():
        if len(out) >= 2 and out[-1].lower() == w.lower() == out[-2].lower():
            continue
        out.append(w)
    return " ".join(out)


def _norm_word(w: str) -> str:
    """Normalize a word for agreement comparison (lowercase, strip punctuation)."""
    return re.sub(r"[^\w']", "", w).lower()


# Custom corrections for terms Whisper consistently mis-hears.
# Keys are matched case-insensitively as whole phrases.
CORRECTIONS = {
    "q a n": "Qwen",
    "quen": "Qwen",
    "mlx": "MLX",
    "whisper local": "WhisperLocal",
    "claude": "Claude",
    "github": "GitHub",
}


def _apply_corrections(text: str) -> str:
    # Built-in corrections + anything the user has taught via Edit & Train
    combined = dict(CORRECTIONS)
    try:
        combined.update(trainer.learned_corrections())
    except Exception:
        pass
    for wrong, right in combined.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def _fix_spacing(text: str) -> str:
    """Ensure a single space after sentence punctuation (. ! ? , ;)."""
    text = re.sub(r"([.!?,;])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _common_prefix_len(a, b) -> int:
    """Length of the longest matching prefix of two normalized word lists."""
    n = 0
    for wa, wb in zip(a, b):
        if wa == wb:
            n += 1
        else:
            break
    return n


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
        if MINIMAL:
            # Force experimental features off for the minimal build
            for k in ("review", "llm_cleanup", "low_power", "personalize"):
                self.cfg[k] = False
        self.model = None
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self._kb = keyboard.Controller()
        self._model_lock     = threading.Lock()
        self._partial_timer  = None
        self._timeout_timer  = None
        self._cancelled      = False
        self._transcribing   = False
        self._target_element = None
        self._target_app     = None
        self._model_ready     = False
        self._committed_text  = ""   # settled transcript, no longer re-transcribed
        self._committed_samples = 0  # audio samples already finalized
        self._prev_words      = []   # last tick's uncommitted hypothesis (normalized)
        self._paused          = False
        self._editing         = False  # review editor open
        self._append_mode     = False  # this recording appends to the open editor
        self._review_produced = ""
        self._last_output     = ""   # last text we pasted (for Edit & Train)

        # Overlay (AppKit panel — created lazily on main thread)
        from overlay import OverlayPanel
        self._overlay = OverlayPanel.alloc().init()
        self._editor = ReviewEditor.alloc().init()
        self._review_audio = None

        self._stream_worker = None
        self._stream_lock = threading.Lock()       # serialize writes to worker stdin
        self._stream_final_event = threading.Event()
        self._stream_final_text = ""
        self._build_menu()
        self._request_accessibility()
        threading.Thread(target=self._start_engine, daemon=True).start()
        self._start_listener()

    def _engine(self) -> str:
        return self.cfg.get("engine", "whisper")   # "whisper" | "hybrid"

    def _use_streaming(self) -> bool:
        # sherpa drives the LIVE display in hybrid (and pure streaming) mode
        return self._engine() in ("streaming", "hybrid")

    def _toggle_engine(self, sender):
        # Toggle between plain Whisper and the hybrid (sherpa live + Whisper final)
        self.cfg["engine"] = "hybrid" if self._engine() != "hybrid" else "whisper"
        sender.state = int(self._engine() == "hybrid")
        save_config(self.cfg)
        self._model_ready = False
        self._set_status("loading…")
        threading.Thread(target=self._start_engine, daemon=True).start()

    def _start_engine(self):
        """Load the engine(s) the selected mode needs."""
        eng = self._engine()
        if eng == "hybrid":
            self._load_streaming()   # sherpa for live display
            self._load_model()       # whisper for the accurate final paste
        elif eng == "streaming":
            self._load_streaming()
        else:
            self._load_model()

    def _load_streaming(self):
        """Start the sherpa streaming worker subprocess + reader thread."""
        self._set_state("transcribing")
        try:
            with self._model_lock:
                # Don't kill the whisper worker in hybrid mode — it does the final paste
                if (self._engine() != "hybrid" and hasattr(self, "_worker")
                        and self._worker and self._worker.poll() is None):
                    try: self._worker.terminate()
                    except Exception: pass
                    self._worker = None
                script = (str(Path(sys.executable).parent.parent / "Resources" / "streaming_worker.py")
                          if getattr(sys, "frozen", False)
                          else str(Path(__file__).parent / "streaming_worker.py"))
                self._stream_worker = subprocess.Popen(
                    [self._venv_python(), script, _streaming_model_dir()],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=self._worker_env(), bufsize=0)
                ready, _, _ = select.select([self._stream_worker.stdout], [], [], 60)
                line = self._stream_worker.stdout.readline().strip() if ready else b"TIMEOUT"
            if line == b"READY":
                self._model_ready = True
                self._set_status("⚡ Real-time mode — hold ⌥")
                threading.Thread(target=self._stream_reader, daemon=True).start()
            else:
                raise RuntimeError(line.decode(errors="replace")[:80])
        except Exception as e:
            print(f"streaming load error: {e}")
            self._set_status(f"Real-time ❌ {str(e)[:50]}")
        self._set_state("idle")

    def _stream_reader(self):
        """Read partial/final lines from the streaming worker."""
        w = self._stream_worker
        while w and w.poll() is None:
            try:
                line = w.stdout.readline()
            except Exception:
                break
            if not line:
                break
            if line.startswith(b"P "):
                if self.recording and self._use_streaming():
                    self._overlay.push_text_parts("", line[2:].decode(errors="replace").strip())
            elif line.startswith(b"F "):
                self._stream_final_text = line[2:].decode(errors="replace").strip()
                self._stream_final_event.set()

    def _stream_send(self, cmd: bytes, data: bytes = b""):
        if not self._stream_worker or self._stream_worker.poll() is not None:
            return
        try:
            with self._stream_lock:
                self._stream_worker.stdin.write(cmd + b"\n" + data)
                self._stream_worker.stdin.flush()
        except Exception as e:
            print(f"stream send error: {e}")

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

        if MINIMAL:
            self._status_item = rumps.MenuItem("Starting…")
            self._status_item.set_callback(None)
            self.menu.add(self._status_item)
            self.menu.add(rumps.separator)
            self._history_menu = rumps.MenuItem("Recent (click to copy)")
            self._populate_history(self._history_menu)
            self.menu.add(self._history_menu)
            filler_item = rumps.MenuItem("Remove Filler Words", callback=self._toggle_filler)
            filler_item.state = int(self.cfg["filler_removal"])
            self.menu.add(filler_item)
            rt_item = rumps.MenuItem("Real-time preview (beta)", callback=self._toggle_engine)
            rt_item.state = int(self._engine() == "hybrid")
            self.menu.add(rt_item)
            self.menu.add(rumps.MenuItem("Check Accessibility…", callback=self._open_accessibility))
            self.menu.add(rumps.separator)
            self.menu.add(rumps.MenuItem("Quit", callback=self._quit))
            return

        self._history_menu = rumps.MenuItem("Recent")
        self._populate_history(self._history_menu)
        self.menu.add(self._history_menu)
        self.menu.add(rumps.separator)

        model_item = rumps.MenuItem("Model")
        for label, repo in MODELS.items():
            item = rumps.MenuItem(label, callback=self._set_model)
            item.state = int(repo == self.cfg["model"])
            model_item.add(item)
        self.menu.add(model_item)

        filler_item = rumps.MenuItem("Remove Filler Words", callback=self._toggle_filler)
        filler_item.state = int(self.cfg["filler_removal"])
        self.menu.add(filler_item)

        mic_item = rumps.MenuItem("Microphone")
        self._build_mic_menu(mic_item)
        self.menu.add(mic_item)

        llm_item = rumps.MenuItem("AI Cleanup (LLM)", callback=self._toggle_llm)
        llm_item.state = int(self.cfg.get("llm_cleanup", False))
        self.menu.add(llm_item)

        lp_item = rumps.MenuItem("Low Power Mode", callback=self._toggle_low_power)
        lp_item.state = int(self.cfg.get("low_power", False))
        self.menu.add(lp_item)

        snd_item = rumps.MenuItem("Sounds", callback=self._toggle_sounds)
        snd_item.state = int(self.cfg.get("sounds", True))
        self.menu.add(snd_item)

        self.menu.add(rumps.MenuItem("Edit Last & Train…", callback=self._edit_and_train))

        review_item = rumps.MenuItem("Review before paste", callback=self._toggle_review)
        review_item.state = int(self.cfg.get("review", False))
        self.menu.add(review_item)

        pers_item = rumps.MenuItem("Personalize (learn my voice)", callback=self._toggle_personalize)
        pers_item.state = int(self.cfg.get("personalize", True))
        self.menu.add(pers_item)
        self._pers_status = rumps.MenuItem(self._personalize_status())
        self._pers_status.set_callback(None)
        self.menu.add(self._pers_status)
        self.menu.add(rumps.MenuItem("Voice Training…", callback=self._show_training))

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Check Accessibility…", callback=self._open_accessibility))
        self._status_item = rumps.MenuItem("Listener: starting…")
        self._status_item.set_callback(None)
        self.menu.add(self._status_item)
        self._power_item = rumps.MenuItem("⚡ Power: —")
        self._power_item.set_callback(None)
        self.menu.add(self._power_item)
        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Quit", callback=self._quit))

        # Live CPU/RAM monitor (proxy for heat) — updates every 2s
        self._power_timer = rumps.Timer(self._update_power, 2)
        self._power_timer.start()

    def _update_power(self, _=None):
        """Sample CPU% and RAM of the app + worker as a heat proxy."""
        try:
            pids = [str(os.getpid())]
            if hasattr(self, "_worker") and self._worker and self._worker.poll() is None:
                pids.append(str(self._worker.pid))
            out = subprocess.run(
                ["ps", "-o", "%cpu=,rss=", "-p", ",".join(pids)],
                capture_output=True, text=True, timeout=2).stdout
            cpu = 0.0
            rss = 0
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    cpu += float(parts[0])
                    rss += int(parts[1])
            self._power_item.title = f"⚡ CPU {cpu:.0f}%   RAM {rss/1024:.0f}MB"
            if hasattr(self, "_overlay"):
                self._overlay.push_power(f"{cpu:.0f}%")
        except Exception:
            pass

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
        new_model = MODELS[sender.title]
        if new_model == self.cfg["model"]:
            return
        for item in self.menu["Model"].values():
            item.state = int(item.title == sender.title)
        self.cfg["model"] = new_model
        save_config(self.cfg)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _toggle_filler(self, sender):
        self.cfg["filler_removal"] = not self.cfg["filler_removal"]
        sender.state = int(self.cfg["filler_removal"])
        save_config(self.cfg)

    def _toggle_llm(self, sender):
        self.cfg["llm_cleanup"] = not self.cfg.get("llm_cleanup", False)
        sender.state = int(self.cfg["llm_cleanup"])
        save_config(self.cfg)

    def _toggle_low_power(self, sender):
        self.cfg["low_power"] = not self.cfg.get("low_power", False)
        sender.state = int(self.cfg["low_power"])
        save_config(self.cfg)
        # Model changes with low-power, so restart the worker
        self._model_ready = False
        threading.Thread(target=self._load_model, daemon=True).start()

    def _toggle_review(self, sender):
        self.cfg["review"] = not self.cfg.get("review", False)
        sender.state = int(self.cfg["review"])
        save_config(self.cfg)

    def _toggle_personalize(self, sender):
        self.cfg["personalize"] = not self.cfg.get("personalize", True)
        sender.state = int(self.cfg["personalize"])
        save_config(self.cfg)
        self._refresh_personalize_status()

    def _show_training(self, _):
        s = trainer.stats()
        bar = trainer.progress_bar(s["progress"], 16)
        curve = trainer.sparkline(trainer.accuracy_curve())
        mb = trainer.corpus_bytes() / (1024 * 1024)
        body = (
            f"🎙  Voice Level {s['level']}\n\n"
            f"{bar}   {s['into_level']}/{trainer.SAMPLES_PER_LEVEL} to next level\n\n"
            f"Accuracy:        {s['accuracy']}%   (you edit {s['edit_rate']}% of dictations)\n"
            f"Accuracy trend:  {curve or '—'}\n\n"
            f"Dictations:      {s['dictations']}\n"
            f"Training samples:{s['train_samples']}  (only your edited ones)\n"
            f"Voice data:      {mb:.0f} MB\n\n"
            "Review mode logs how often you edit — the less you edit over time, "
            "the better it has learned your voice. Edited dictations become "
            "training data for the next voice update."
        )
        ready = s.get("ready_to_train")
        rumps.alert(title="Voice Training", message=body,
                    ok=("Train now" if ready else "OK"))

    def _personalize_status(self) -> str:
        try:
            s = trainer.stats()
            bar = trainer.progress_bar(s["progress"])
            return f"  🎙 Voice Lv {s['level']}  {bar}  {s['accuracy']}% accurate"
        except Exception:
            return "  🎙 Voice Lv 0"

    def _refresh_personalize_status(self):
        if hasattr(self, "_pers_status"):
            AppHelper.callAfter(
                lambda: setattr(self._pers_status, "title", self._personalize_status()))

    def _toggle_sounds(self, sender):
        self.cfg["sounds"] = not self.cfg.get("sounds", True)
        sender.state = int(self.cfg["sounds"])
        save_config(self.cfg)
        if self.cfg["sounds"]:
            self._play_sound("Tink")

    def _edit_and_train(self, _):
        """Show the last transcription in an editable dialog; learn from the edits."""
        produced = self._last_output.strip()
        if not produced:
            rumps.alert("Edit & Train", "No transcription yet — dictate something first.")
            return
        resp = rumps.Window(
            message="Fix anything wrong. Your corrections train the app.",
            title="Edit Last & Train",
            default_text=produced,
            ok="Save & Learn",
            cancel="Cancel",
            dimensions=(420, 120),
        ).run()
        if not resp.clicked:
            return
        edited = resp.text.strip()
        n_new = trainer.record_correction(produced, edited)
        if edited and edited != produced:
            try:
                trainer.update_last_text(edited)  # relabel the banked audio sample
            except Exception:
                pass
        self._last_output = edited
        if edited and edited != produced:
            rumps.notification(
                "WhisperLocal",
                "Learned from your edit",
                f"{n_new} new correction(s) added." if n_new else "Saved as training data.",
                sound=False,
            )

    def _play_sound(self, name: str):
        if not self.cfg.get("sounds", True):
            return
        def _go():
            try:
                snd = NSSound.alloc().initWithContentsOfFile_byReference_(
                    f"/System/Library/Sounds/{name}.aiff", True)
                if snd:
                    snd.setVolume_(0.22)
                    snd.play()
            except Exception:
                pass
        AppHelper.callAfter(_go)

    def _effective_model(self) -> str:
        return LOW_POWER_MODEL if self.cfg.get("low_power", False) else self.cfg["model"]

    def _chunk_interval(self) -> float:
        return 1.5 if self.cfg.get("low_power", False) else CHUNK_INTERVAL

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
        try:
            if hasattr(self, "_worker") and self._worker.poll() is None:
                self._worker.terminate()
        except Exception:
            pass
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _set_state(self, state: str):
        icons = {"idle": ICON_IDLE, "recording": ICON_RECORDING, "transcribing": ICON_TRANSCRIBING}
        icon = icons.get(state, ICON_IDLE)
        # Always set the status-bar icon on the main thread (AppKit isn't thread-safe)
        AppHelper.callAfter(lambda: setattr(self, "icon", icon))

    def _set_status(self, text: str):
        """Set the menu status line on the main thread."""
        if hasattr(self, "_status_item"):
            AppHelper.callAfter(lambda: setattr(self._status_item, "title", text))

    def _bundle_pylib(self) -> str:
        """Path to the embedded ML deps inside the app bundle (self-contained)."""
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).parent.parent / "Resources" / "pylib")
        return ""

    def _venv_python(self) -> str:
        """Interpreter for the worker. In the bundle, use the app's OWN python
        (Contents/MacOS/python) so we depend on nothing outside the .app."""
        if getattr(sys, "frozen", False):
            bundled = Path(sys.executable).parent / "python"   # Contents/MacOS/python
            if bundled.exists():
                return str(bundled)
        return sys.executable

    def _worker_script(self) -> str:
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).parent.parent / "Resources" / "transcribe_worker.py")
        return str(Path(__file__).parent / "transcribe_worker.py")

    def _worker_env(self) -> dict:
        venv_python = self._venv_python()
        env = {
            "HOME":   str(Path.home()),
            "USER":   os.environ.get("USER", ""),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "PATH":   f"{Path(venv_python).parent}:/usr/bin:/bin:/usr/local/bin",
            "LANG":   "en_US.UTF-8",
        }
        paths = []
        pylib = self._bundle_pylib()
        if pylib and Path(pylib).exists():
            paths.append(pylib)         # embedded mlx/mlx_whisper/sherpa_onnx/etc.
        if getattr(sys, "frozen", False):
            paths.append(str(Path(sys.executable).parent.parent / "Resources"))  # streaming_engine.py
        else:
            paths.append(str(Path(__file__).parent))
        env["PYTHONPATH"] = ":".join(paths)
        return env

    def _load_model(self):
        """Start persistent worker — stays alive between transcriptions."""
        self._set_state("transcribing")
        model = self._effective_model()
        # Shut down the streaming worker unless hybrid mode needs it for live display
        if self._engine() != "hybrid" and self._stream_worker and self._stream_worker.poll() is None:
            try: self._stream_worker.terminate()
            except Exception: pass
            self._stream_worker = None
        try:
            with self._model_lock:
                # Terminate any existing worker (e.g. when switching models)
                if hasattr(self, "_worker") and self._worker and self._worker.poll() is None:
                    try:
                        self._worker.terminate()
                    except Exception:
                        pass
                self._worker = subprocess.Popen(
                    [self._venv_python(), self._worker_script(), model, "en"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._worker_env(),
                )
                # Wait for READY with a generous timeout (first run downloads the model)
                ready, _, _ = select.select([self._worker.stdout], [], [], 180)
                line = self._worker.stdout.readline().strip() if ready else b"TIMEOUT"
            if line == b"READY":
                self._model_ready = True
                if self._engine() == "hybrid":
                    self._set_status("⚡ Real-time + Whisper — hold ⌥")
                else:
                    tag = "🔋 " if self.cfg.get("low_power", False) else ""
                    model_short = model.split("/")[-1]
                    self._set_status(f"MLX ✅  {tag}{model_short} — hold ⌥")
            elif line.startswith(b"ERROR:"):
                raise RuntimeError(line.decode())
            else:
                raise RuntimeError(f"Unexpected: {line[:80]}")
        except Exception as e:
            print(f"Worker start error: {e}")
            self._set_status(f"MLX ❌ {str(e)[:60]}")
        self._set_state("idle")

    def _run_transcription(self, audio: np.ndarray, prompt: str = "") -> str:
        """Return plain text only."""
        return self._run_transcription_full(audio, prompt).get("text", "")

    def _run_cleanup(self, text: str) -> str:
        """Send text to the worker's LLM cleanup. Returns cleaned text (or original)."""
        if not hasattr(self, "_worker") or self._worker.poll() is not None:
            return text
        try:
            examples = trainer.few_shot_examples(3)
            with self._model_lock:
                header = json.dumps({"cleanup": text, "examples": examples}).encode()
                self._worker.stdin.write(header + b"\n")
                self._worker.stdin.flush()
                line = self._worker.stdout.readline()
            if not line.strip():
                return text
            resp = json.loads(line.strip())
            secs = resp.get("secs", 0)
            print(f"LLM cleanup took {secs:.2f}s")
            return resp.get("text", text)
        except Exception as e:
            print(f"Cleanup error: {e}")
            return text

    def _run_transcription_full(self, audio: np.ndarray, prompt: str = "",
                                words: bool = False) -> dict:
        """Send audio to worker, return {'text', optionally 'words'}.
        Self-healing: if the worker has died or stops responding, restart it."""
        empty = {"text": "", "words": []}
        if not hasattr(self, "_worker") or self._worker is None or self._worker.poll() is not None:
            self._restart_worker_async()
            return empty
        try:
            data = audio.tobytes()
            header = json.dumps(
                {"n": len(data), "prompt": prompt[-200:], "words": words}).encode()
            self._worker.stdin.write(header + b"\n" + data)
            self._worker.stdin.flush()
            # Wait for a response with a timeout — never block forever.
            ready, _, _ = select.select([self._worker.stdout], [], [], 25)
            if not ready:
                print("worker unresponsive — restarting")
                self._restart_worker_async()
                return empty
            line = self._worker.stdout.readline()
            if not line.strip():
                return empty
            return json.loads(line.strip())
        except Exception as e:
            print(f"Transcription error: {e}; restarting worker")
            self._restart_worker_async()
            return empty

    def _restart_worker_async(self):
        """Kill any wedged worker and start a fresh one (off-thread, no deadlock).
        _load_model waits for the model lock, so it runs once the caller releases."""
        try:
            if hasattr(self, "_worker") and self._worker and self._worker.poll() is None:
                self._worker.kill()
        except Exception:
            pass
        self._model_ready = False
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------
    # Hotkey listener
    # ------------------------------------------------------------------

    def _start_listener(self):
        """Use NSEvent global monitor for modifier keys (more reliable than pynput for Option)."""
        # NSFlagsChangedMask = 1 << 12  — fires on every modifier key change
        NSFlagsChangedMask = 1 << 12
        NSKeyDownMask      = 1 << 10

        alt_was_down = [False]
        shift_was_down = [False]

        def flags_handler(event):
            flags = event.modifierFlags()
            alt_now   = bool(flags & 0x00080000)  # NSAlternateKeyMask
            shift_now = bool(flags & 0x00020000)  # NSShiftKeyMask

            # Right Option edge → start / stop recording
            if alt_now and not alt_was_down[0]:
                alt_was_down[0] = True
                self._on_press(keyboard.Key.alt_r)
            elif not alt_now and alt_was_down[0]:
                alt_was_down[0] = False
                self._on_release(keyboard.Key.alt_r)

            # Shift tap while recording → toggle pause
            if shift_now and not shift_was_down[0]:
                shift_was_down[0] = True
                if self.recording:
                    self._toggle_pause()
            elif not shift_now and shift_was_down[0]:
                shift_was_down[0] = False

        def key_handler(event):
            # keyCode 53 = Escape
            if event.keyCode() == 53:
                self._on_press(keyboard.Key.esc)

        # Global monitors fire for OTHER apps; local monitors fire when OUR app
        # is active (e.g. while the review editor is open) — need both so you can
        # hold ⌥ to append more speech while editing.
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSFlagsChangedMask, flags_handler)
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask, key_handler)

        def local_flags(event):
            flags_handler(event)
            return event

        def local_key(event):
            key_handler(event)
            return event

        NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSFlagsChangedMask, local_flags)
        NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask, local_key)

        ready = "hold ⌥ to dictate" if self._model_ready else "loading model…"
        self._set_status(f"Listener: ✅ active — {ready}")

    def _on_press(self, key):
        # Called from NSEvent monitor (main thread) or pynput fallback
        if key == keyboard.Key.esc:
            self._cancel()
            return
        if self._is_hotkey(key) and (self.recording or self._transcribing):
            self._cancel()
            return
        if not self._is_hotkey(key) or self.recording or not self._model_ready:
            return
        self._cancelled         = False
        self._committed_text    = ""
        self._committed_samples = 0
        self._prev_words        = []
        self._paused            = False
        self._append_mode       = self._editing   # if editor open, this take appends
        if not self._editing:
            self._target_element, self._target_app = self._snapshot_focus()
        self.recording    = True
        self.audio_chunks = []
        if self._use_streaming():
            self._stream_final_event.clear()
            self._stream_send(b"START")
        WAVEFORM_WINDOW.extend([0.02] * WAVEFORM_BINS)
        self._set_state("recording")
        self._play_sound("Tink")        # soft start cue
        self._overlay.show_async()
        self._overlay.push_state("recording")
        # Open the mic OFF the main thread — PortAudio open/close can block, and
        # blocking the main thread freezes the overlay + menu bar.
        threading.Thread(target=self._begin_capture, daemon=True).start()

    def _begin_capture(self):
        # Refresh PortAudio's device list every time so device changes (unplug /
        # switch / sleep) are always picked up. Safe here — it's a worker thread.
        try:
            sd._terminate(); sd._initialize()
        except Exception as e:
            print(f"audio reinit: {e}")
        if not self._open_stream():
            self.recording = False
            self._overlay.push_state("idle")
            self._overlay.hide_async()
            self._set_state("idle")
            rumps.notification("WhisperLocal", "No microphone",
                               "Couldn't access a mic. Check it's connected.", sound=False)
            return
        if not self.recording:          # released already (quick tap) — clean up
            self._close_stream()
            return
        if not self._use_streaming():   # Whisper needs the chunk timer; sherpa doesn't
            self._start_partial_timer()
        self._start_timeout()

    def _close_stream(self):
        """Stop+close the audio stream. MUST run off the main thread (can block)."""
        s = self.stream
        self.stream = None
        if s:
            try:
                s.stop(); s.close()
            except Exception:
                pass

    def _open_stream(self) -> bool:
        """Open the mic stream, refreshing the audio device list first so a
        hot-plugged / unplugged mic is picked up (PortAudio caches devices)."""
        for attempt in (1, 2):
            try:
                if attempt == 2:
                    # Refresh PortAudio's device list and retry on the default device
                    try:
                        sd._terminate(); sd._initialize()
                    except Exception:
                        pass
                device = self.cfg.get("input_device") if attempt == 1 else None
                self.stream = sd.InputStream(
                    device=device, samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", callback=self._audio_cb)
                self.stream.start()
                return True
            except Exception as e:
                print(f"mic open attempt {attempt} failed: {e}")
                self.stream = None
        return False

    def _on_release(self, key):
        if not self._is_hotkey(key) or not self.recording:
            return
        self._stop_timeout()
        self.recording = False
        self._stop_partial_timer()
        self._transcribing = True
        self._set_state("transcribing")
        self._overlay.push_state("transcribing")
        # Stream close + transcription both happen off the main thread.
        threading.Thread(target=self._transcribe_final, daemon=True).start()

    def _is_hotkey(self, key):
        return key in (HOTKEY, HOTKEY_ALT, keyboard.Key.alt_l)

    def _cancel(self):
        """Stop recording and hide overlay without transcribing."""
        if not self.recording and not self._transcribing:
            return
        self._cancelled = True
        self._stop_timeout()
        self._stop_partial_timer()
        self.recording = False
        threading.Thread(target=self._close_stream, daemon=True).start()  # off main thread
        self.audio_chunks = []
        self._transcribing = False
        self._overlay.push_state("idle")
        self._overlay.hide_async()
        self._set_state("idle")

    def _toggle_pause(self):
        """Pause/resume recording mid-dictation (tap Shift while holding ⌥)."""
        self._paused = not self._paused
        if self._paused:
            self._stop_partial_timer()
            self._overlay.push_state("paused")
        else:
            self._overlay.push_state("recording")
            self._start_partial_timer()

    def _audio_cb(self, indata, frames, t, status):
        if not self.recording or self._paused:
            return  # paused → drop audio so the gap isn't recorded
        chunk = indata.copy()
        self.audio_chunks.append(chunk)
        # STREAMING: ship audio frames to the sherpa worker (partials come back
        # asynchronously via the reader thread).
        if self._use_streaming():
            self._stream_send(b"A" + str(chunk.nbytes).encode(), chunk.flatten().tobytes())
        # Decibel-based level so quiet / distant speech still shows clearly.
        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-9
        db = 20.0 * np.log10(rms)          # ~ -60 (silence) .. 0 (max)
        level = (db + 60.0) / 45.0         # map -60..-15 dB -> 0..1
        level = float(min(1.0, max(0.0, level)))
        WAVEFORM_WINDOW.append(level)
        self._overlay.push_levels(list(WAVEFORM_WINDOW))

    # ------------------------------------------------------------------
    # Streaming transcription — transcribe chunks WHILE recording
    # ------------------------------------------------------------------

    def _start_partial_timer(self):
        self._partial_timer = threading.Timer(FIRST_CHUNK_DELAY, self._chunk_tick)
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

    def _chunk_tick(self):
        """Fire periodically while recording: transcribe newly-arrived audio."""
        if not self.recording:
            return
        threading.Thread(target=self._process_chunk, daemon=True).start()
        # Reschedule (slower cadence in low-power mode)
        self._partial_timer = threading.Timer(self._chunk_interval(), self._chunk_tick)
        self._partial_timer.daemon = True
        self._partial_timer.start()

    def _process_chunk(self):
        """LocalAgreement-2: commit only words that two consecutive runs agree on.
        A word is locked in when the model produces it identically twice — far more
        robust than committing on a timer (self-correcting, repetition-resistant)."""
        if not self._model_lock.acquire(blocking=False):
            _slog("skip(lock)")
            return  # a transcription is already in flight; skip this tick
        try:
            if not self.audio_chunks:
                return
            audio = np.concatenate(self.audio_chunks).flatten()
            tail = audio[self._committed_samples:]
            tail_dur = len(tail) / SAMPLE_RATE
            if tail_dur < MIN_TAIL_SECS:
                return

            # Silence gating: skip the (expensive) transcription pass if the tail
            # is essentially quiet — nothing to transcribe, saves a lot of compute.
            tail_rms = float(np.sqrt(np.mean(tail ** 2)))
            if tail_rms < 0.004:
                _slog(f"skip(silent) tail={tail_dur:.1f}")
                return

            _t0 = time.time()
            result = self._run_transcription_full(
                tail, self._committed_text, words=True)
            _dur = time.time() - _t0
            words = result.get("words", [])
            if not words:
                _slog(f"pass tail={tail_dur:.1f} dur={_dur:.2f} words=0")
                return

            new_norm = [_norm_word(wd["w"]) for wd in words]

            # Agreement = longest common prefix with the previous hypothesis
            agree = _common_prefix_len(self._prev_words, new_norm)

            # Don't commit a word still within SETTLE_MARGIN of the live edge
            # (it may still be mid-utterance even if it matched).
            while agree > 0 and words[agree - 1]["end"] > tail_dur - SETTLE_MARGIN:
                agree -= 1

            # Safeguard: if the tail has grown too long (continuous speech where
            # passes never agree), force-commit everything older than SETTLE_MARGIN
            # so the tail stays short and passes stay fast (prevents the stall).
            if tail_dur > MAX_TAIL_SECS:
                forced = agree
                while forced < len(words) and words[forced]["end"] <= tail_dur - SETTLE_MARGIN:
                    forced += 1
                agree = max(agree, forced)

            _slog(f"pass tail={tail_dur:.1f} dur={_dur:.2f} words={len(words)} "
                  f"agree={agree} committed_chars={len(self._committed_text)}")
            if agree > 0:
                committed = " ".join(wd["w"] for wd in words[:agree]).strip()
                self._committed_text = _collapse_repeats(
                    (self._committed_text + " " + committed).strip())
                self._committed_samples += int(words[agree - 1]["end"] * SAMPLE_RATE)
                # Remaining hypothesis is now relative to the new committed point
                self._prev_words = new_norm[agree:]
            else:
                self._prev_words = new_norm

            # Live display: committed (solid) + settling tail (shimmer)
            tail_guess = _collapse_repeats(" ".join(wd["w"] for wd in words[agree:]).strip())
            committed_disp = _collapse_repeats(self._committed_text.strip())
            self._overlay.push_text_parts(committed_disp, tail_guess)
        except Exception as e:
            print(f"Chunk error: {e}")
        finally:
            self._model_lock.release()

    def _transcribe_final(self):
        """On release: produce the final text and paste."""
        try:
            self._close_stream()   # stop the mic off the main thread
            if self._cancelled:
                return

            # In hybrid mode, tell sherpa to wrap up its live display, but the
            # PASTE comes from Whisper below (accurate). In pure-streaming mode,
            # sherpa's output IS the final.
            if self._use_streaming():
                self._stream_send(b"FINAL")
            if self._engine() == "streaming":
                self._stream_final_event.wait(timeout=6)
                final = self._stream_final_text.strip()
                if self.cfg.get("filler_removal"):
                    final = FILLER_RE.sub("", final).strip()
                if final:
                    self._commit_final(final)
                return

            # Start from committed text; try to transcribe the small remaining tail.
            final = self._committed_text

            # Start from committed text; try to transcribe the small remaining tail.
            final = self._committed_text
            try:
                with self._model_lock:
                    if self.audio_chunks:
                        audio = np.concatenate(self.audio_chunks).flatten()
                        tail = audio[self._committed_samples:]
                        if len(tail) >= int(SAMPLE_RATE * 0.2):
                            text = self._run_transcription(tail, self._committed_text)
                            if text:
                                final = (self._committed_text + " " + text).strip()
            except Exception as e:
                print(f"Final transcription error (using committed text): {e}")

            final = _collapse_repeats(final.strip())
            final = _apply_corrections(final)
            final = _fix_spacing(final)
            if self.cfg["filler_removal"]:
                final = FILLER_RE.sub("", final).strip()
            if not final:
                return

            # Append mode: this take adds to the open review editor, no paste.
            if self._append_mode:
                self._overlay.hide_async()
                AppHelper.callAfter(self._editor.appendText_, final)
                return

            # Optional LLM cleanup pass (failures fall back to raw text)
            if self.cfg.get("llm_cleanup", False):
                self._overlay.push_text(final)
                self._overlay.push_state("polishing")  # sparkle + "Enhancing"
                try:
                    cleaned = self._run_cleanup(final)
                    if cleaned:
                        final = cleaned
                except Exception as e:
                    print(f"Cleanup failed, pasting raw: {e}")

            # Capture the audio now (for personalization / review labelling)
            self._review_audio = (np.concatenate(self.audio_chunks).flatten()
                                  if self.audio_chunks else None)

            if self.cfg.get("review", False):
                # Hand off to the inline editor; paste happens on confirm.
                self._overlay.hide_async()
                AppHelper.callAfter(self._present_review, final)
                return

            self._commit_final(final)
        except Exception as e:
            print(f"Final paste error: {e}")
        finally:
            self._transcribing = False
            if not self.cfg.get("review", False):
                self._overlay.push_state("idle")
                self._overlay.hide_async()
                self._set_state("idle")

    def _present_review(self, final):
        """Show the editable confirm field (main thread)."""
        self._review_produced = final           # what the model produced (for edit detection)
        self._editing = True
        self._editor._on_submit = self._on_review_submit
        self._editor._on_cancel = self._on_review_cancel
        self._editor.presentText_(final)

    def _on_review_submit(self, edited):
        final = (edited or "").strip()
        self._editing = False
        self._set_state("idle")
        if not final:
            return

        produced = (self._review_produced or "").strip()
        was_edited = trainer.record_outcome(produced, final)   # logs accuracy metric

        # Only edited dictations are useful training data — save just those.
        if was_edited and self.cfg.get("personalize", True):
            try:
                trainer.save_sample(self._review_audio, SAMPLE_RATE, final)
            except Exception as e:
                print(f"Sample save error: {e}")
        self._refresh_personalize_status()

        # Re-activate the target app (main thread → safe) then paste
        try:
            if self._target_app:
                self._target_app.activateWithOptions_(1 << 1)
        except Exception as e:
            print(f"reactivate error: {e}")
        self._commit_final(final, play=True, paste_delay=0.3)

    def _on_review_cancel(self):
        self._editing = False
        self._set_state("idle")

    def _commit_final(self, final, play=True, paste_delay=0.0):
        """Paste + store history. (Training data is saved separately, only on edit.)"""
        final = final.strip()
        if not final:
            return
        # Paste FIRST so nothing downstream can ever block it.
        if paste_delay > 0:
            threading.Timer(paste_delay, lambda: self._paste(final + " ")).start()
        else:
            self._paste(final + " ")
        if play:
            self._play_sound("Pop")
        self._last_output = final
        try:
            self._add_history(final)
        except Exception as e:
            print(f"history error: {e}")

    def _add_history(self, text: str):
        history = self.cfg.setdefault("history", [])
        history.append(text)
        self.cfg["history"] = history[-HISTORY_MAX:]
        save_config(self.cfg)
        if hasattr(self, "_history_menu"):   # minimal build has no Recent menu
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
        """Insert text. For anything but very short snippets, use an instant
        clipboard paste (⌘V) — typing char-by-char is slow for long transcripts.
        Short snippets are typed directly so the clipboard isn't touched."""
        if len(text) <= 25:
            try:
                self._kb.type(text)
                return
            except Exception as e:
                print(f"type failed ({e}); using clipboard")
        # Instant clipboard paste — length-independent
        try:
            prev = pyperclip.paste()
        except Exception:
            prev = ""
        try:
            pyperclip.copy(text)
            time.sleep(0.04)
            with self._kb.pressed(keyboard.Key.cmd):
                self._kb.tap("v")
            time.sleep(0.12)
        except Exception as e:
            print(f"clipboard paste failed: {e}")
        finally:
            try:
                pyperclip.copy(prev)
            except Exception:
                pass


if __name__ == "__main__":
    WhisperLocal().run()
