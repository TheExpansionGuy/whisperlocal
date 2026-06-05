# WhisperLocal

A fully local, push-to-talk dictation app for macOS. Hold a key, speak, release — your words are transcribed on-device and typed into whatever app is focused. No cloud, no accounts, no data ever leaves your Mac.

Built with [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) running on Apple Silicon's Neural Engine.

![menu bar app](assets/menubar@2x.png)

## Features

- **Push-to-talk** — hold **Right Option (⌥)** to record, release to transcribe and paste. Press **Esc** (or tap ⌥ again) to cancel.
- **Streaming transcription** — text is transcribed *while you speak*, not after. Settled text is committed at natural pause boundaries so only the most recent audio is reprocessed — fast, with near-instant paste on release.
- **Fully local & private** — uses MLX Whisper on the Apple Neural Engine. Nothing is sent to any server.
- **Live overlay** — a floating pill at the bottom of the screen shows a waveform, recording timer, and the transcript building up in real time. It stays compact for controls and expands when there's text to read.
- **Model picker** — choose Tiny / Base / Small / Medium from the menu bar to trade speed for accuracy.
- **Microphone picker** — select any input device.
- **Filler-word removal** — strips "um", "uh", "like", "you know", etc. (toggleable).
- **History** — the last 10 transcriptions, click to re-copy.
- **Smart text injection** — inserts via the macOS Accessibility API, falling back to clipboard paste, and restores focus to your original app.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Python 3.x (for building from source)

## Installation

### From source

```bash
git clone https://github.com/TheExpansionGuy/whisperlocal.git
cd whisperlocal

# Set up the virtual environment and dependencies
./setup.sh

# Build and install the .app to /Applications
source .venv/bin/activate && ./build.sh
```

### Permissions

WhisperLocal needs two macOS permissions, which it will prompt for on first launch:

1. **Accessibility** — required for the global ⌥ hotkey and for typing text into other apps.
   Grant it in **System Settings → Privacy & Security → Accessibility**, then make sure WhisperLocal is toggled on.
2. **Microphone** — to record your voice.

> On first run, the selected Whisper model (~250MB for `small.en`) downloads once from Hugging Face, then runs entirely offline.

## Usage

1. Launch WhisperLocal — a microphone icon appears in your menu bar.
2. Wait for the menu to show **"MLX ✅ … — hold ⌥"** (the model warms up on launch).
3. Click into any text field.
4. **Hold Right Option**, speak, then **release**. The transcript is typed in automatically.

## Development

The app is plain Python. Two workflows:

| Command | What it does |
|---|---|
| `./build.sh` | Full rebuild with py2app, bundles the MLX worker, installs to `/Applications`. Needed when dependencies change. |
| `./update.sh` | Fast update — copies `main.py` / `overlay.py` / `transcribe_worker.py` into the installed bundle and restarts. No permission reset. Use this for pure-Python changes. |

### Architecture

- **`main.py`** — the rumps menu-bar app: hotkey handling (via `NSEvent`), audio capture (`sounddevice`), streaming transcription orchestration, and text injection.
- **`overlay.py`** — the floating pill UI, drawn entirely in a single `NSView` (waveform, indicator, timer, live transcript) with all updates marshalled to the main thread.
- **`transcribe_worker.py`** — a persistent subprocess that loads the MLX Whisper model once and transcribes audio chunks on demand over a pipe, avoiding per-transcription startup cost. Runs in the project venv so MLX and its native dependencies load correctly.
- **`icon.py`** — generates the app icon (soundprint design) and menu-bar template icons.

### How streaming works

While recording, a timer fires every ~1.2s and transcribes only the audio since the last *committed* point. Whisper returns segments with timestamps; any segment that ends comfortably before the live edge of the audio is considered "settled", committed to the final transcript, and never reprocessed. Only the unsettled tail is re-transcribed each tick. On release, the small remaining tail is transcribed and the full text is pasted.

## Roadmap

- [ ] Upgrade to `large-v3-turbo` for higher accuracy
- [ ] Custom vocabulary / correction dictionary
- [ ] Optional local LLM cleanup pass (punctuation, grammar, homophones)
- [ ] Code signing + notarization
- [ ] Sparkle auto-updates

## License

MIT
