from setuptools import setup

APP = ["main.py"]
DATA_FILES = [
    ("", ["overlay.py", "trainer.py", "transcribe_worker.py"]),
    ("assets", [
        "assets/menubar.png",
        "assets/menubar@2x.png",
        "assets/menubar_rec.png",
        "assets/menubar_rec@2x.png",
        "assets/menubar_proc.png",
        "assets/menubar_proc@2x.png",
    ]),
]
OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/icon.icns",
    "plist": {
        "CFBundleName": "WhisperLocal",
        "CFBundleDisplayName": "WhisperLocal",
        "CFBundleIdentifier": "com.whisperlocal.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0",
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "WhisperLocal records audio to transcribe your speech locally."
        ),
        "NSAppleEventsUsageDescription": (
            "WhisperLocal uses System Events to paste transcribed text into the active app."
        ),
    },
    # mlx can't be bundled by py2app — we embed the venv instead (see build.sh)
    "packages": [
        "sounddevice",
        "_sounddevice_data",
        "numpy",
        "rumps",
        "pynput",
        "pyperclip",
        "huggingface_hub",
        "ApplicationServices",
        "AppKit",
        "Foundation",
    ],
    "excludes": ["tkinter", "matplotlib", "scipy", "PIL", "mlx", "mlx_whisper"],
    "strip": False,
}

setup(
    name="WhisperLocal",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
