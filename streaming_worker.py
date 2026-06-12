#!/usr/bin/env python3
"""Persistent streaming-ASR worker (sherpa-onnx). Runs in the bundle's python.

Protocol (binary stdin, line stdout):
  startup:        prints "READY\\n"
  START\\n         begin a new utterance
  A<nbytes>\\n+bytes   feed float32 audio; worker emits "P <partial>\\n" if changed
  FINAL\\n         flush; worker emits "F <final text>\\n"
  QUIT\\n          exit
"""
import sys
import io
import numpy as np


def main():
    model_dir = sys.argv[1]

    _out, _err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        from streaming_engine import StreamingEngine
        engine = StreamingEngine(model_dir)
    except Exception as e:
        sys.stdout, sys.stderr = _out, _err
        sys.stdout.write(f"ERROR:{e}\n"); sys.stdout.flush()
        return
    finally:
        sys.stdout, sys.stderr = _out, _err

    sys.stdout.write("READY\n"); sys.stdout.flush()

    last_partial = ""
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        cmd = line.strip()
        if cmd == b"QUIT":
            break
        elif cmd == b"START":
            engine.start()
            last_partial = ""
        elif cmd == b"FINAL":
            final = engine.finalize()
            sys.stdout.write("F " + final.replace("\n", " ") + "\n")
            sys.stdout.flush()
            last_partial = ""
        elif cmd.startswith(b"A"):
            try:
                n = int(cmd[1:])
                data = sys.stdin.buffer.read(n)
                audio = np.frombuffer(data, dtype=np.float32)
                partial = engine.feed(audio)
                if partial and partial != last_partial:
                    last_partial = partial
                    sys.stdout.write("P " + partial.replace("\n", " ") + "\n")
                    sys.stdout.flush()
            except Exception as e:
                print(f"feed err: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
