#!/usr/bin/env python3
"""Standalone transcription worker — called as subprocess by the main app.
Reads raw float32 audio bytes from stdin, writes JSON text to stdout.
Runs in the venv so mlx_whisper and all its deps are available.
"""
import sys
import json
import numpy as np
import mlx_whisper

def main():
    model   = sys.argv[1]
    lang    = sys.argv[2] if len(sys.argv) > 2 else "en"
    data    = sys.stdin.buffer.read()
    audio   = np.frombuffer(data, dtype=np.float32)
    result  = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        language=lang,
        verbose=False,
    )
    print(json.dumps(result.get("text", "")))

if __name__ == "__main__":
    main()
