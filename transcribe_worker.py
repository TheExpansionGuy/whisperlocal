#!/usr/bin/env python3
"""Persistent transcription worker.
Protocol:
  startup: prints "READY\\n" once warm
  per request: reads "<n_bytes>\\n" then n_bytes float32 audio, writes JSON text + "\\n"
  shutdown: reads "QUIT\\n"
"""
import sys
import io
import json
import numpy as np

def main():
    model = sys.argv[1]
    lang  = sys.argv[2] if len(sys.argv) > 2 else "en"

    # Silence stdout/stderr during import and warm-up so nothing
    # accidentally gets written to the pipe before READY
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout   = io.StringIO()
    sys.stderr   = io.StringIO()

    try:
        import mlx_whisper
        silence = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(
            silence, path_or_hf_repo=model, language=lang, verbose=False)
    except Exception as e:
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr
        print(f"ERROR: {e}", flush=True)
        return
    finally:
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        header = sys.stdin.buffer.readline().strip()
        if not header or header == b"QUIT":
            break
        try:
            n_bytes = int(header.decode())
            data    = sys.stdin.buffer.read(n_bytes)
            audio   = np.frombuffer(data, dtype=np.float32)
            result  = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model, language=lang, verbose=False)
            text = result.get("text", "").strip()
            print(json.dumps(text), flush=True)
        except Exception as e:
            print(json.dumps(""), flush=True)
            print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
