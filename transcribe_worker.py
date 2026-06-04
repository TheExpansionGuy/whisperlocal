#!/usr/bin/env python3
"""Persistent transcription worker — stays alive between requests to avoid startup cost.
Protocol:
  startup: prints "READY\\n" when model is warm
  per request: reads "<n_bytes>\\n" header, then n_bytes of float32 audio
               writes JSON text + "\\n"
  shutdown: reads "QUIT\\n"
"""
import sys
import json
import numpy as np
import mlx_whisper

def main():
    model = sys.argv[1]
    lang  = sys.argv[2] if len(sys.argv) > 2 else "en"

    # Warm up
    silence = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=model, language=lang, verbose=False)
    print("READY", flush=True)

    while True:
        header = sys.stdin.readline().strip()
        if not header or header == "QUIT":
            break
        n_bytes = int(header)
        data    = sys.stdin.buffer.read(n_bytes)
        audio   = np.frombuffer(data, dtype=np.float32)
        result  = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model, language=lang, verbose=False)
        text = result.get("text", "").strip()
        print(json.dumps(text), flush=True)

if __name__ == "__main__":
    main()
