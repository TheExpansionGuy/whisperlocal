#!/usr/bin/env python3
"""Persistent streaming transcription worker.
Protocol:
  startup:     prints "READY\\n"
  per request: reads JSON header line: {"n": <bytes>, "prompt": "<prev text>"}
               then reads n bytes of float32 audio
               writes JSON: {"text": "<transcribed>"} + "\\n"
  shutdown:    reads "QUIT\\n"
"""
import sys
import io
import json
import numpy as np

def main():
    model = sys.argv[1]
    lang  = sys.argv[2] if len(sys.argv) > 2 else "en"

    # Silence all output during import + warm-up
    _out, _err = sys.stdout, sys.stderr
    sys.stdout  = io.StringIO()
    sys.stderr  = io.StringIO()
    try:
        import mlx_whisper
        silence = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(silence, path_or_hf_repo=model,
                               language=lang, verbose=False)
    except Exception as e:
        sys.stdout, sys.stderr = _out, _err
        print(f"ERROR:{e}", flush=True)
        return
    finally:
        sys.stdout, sys.stderr = _out, _err

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        header_line = sys.stdin.buffer.readline().strip()
        if not header_line or header_line == b"QUIT":
            break
        try:
            header  = json.loads(header_line.decode())
            n_bytes = header["n"]
            prompt  = header.get("prompt", "")
            data    = sys.stdin.buffer.read(n_bytes)
            audio   = np.frombuffer(data, dtype=np.float32)

            kwargs = dict(path_or_hf_repo=model, language=lang, verbose=False)
            if prompt:
                kwargs["initial_prompt"] = prompt

            result = mlx_whisper.transcribe(audio, **kwargs)
            text   = result.get("text", "").strip()
            sys.stdout.write(json.dumps({"text": text}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"text": ""}) + "\n")
            sys.stdout.flush()
            print(f"chunk error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
