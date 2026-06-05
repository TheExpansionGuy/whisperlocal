#!/usr/bin/env python3
"""Persistent streaming transcription worker.
Protocol:
  startup:     prints "READY\\n"
  transcribe:  JSON header {"n": <bytes>, "prompt": "...", "words": bool}
               then n bytes of float32 audio  ->  {"text", "words"}
  cleanup:     JSON header {"cleanup": "<text>"}  (no audio)  ->  {"text"}
  shutdown:    reads "QUIT\\n"
"""
import sys
import io
import json
import time
import numpy as np

# Lazy-loaded LLM for optional cleanup
_LLM = {"model": None, "tokenizer": None}
LLM_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _llm_cleanup(text):
    """Polish dictated text with a small local LLM. Returns cleaned text + seconds taken."""
    t0 = time.time()
    try:
        import mlx_lm
        if _LLM["model"] is None:
            _out, _err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            try:
                _LLM["model"], _LLM["tokenizer"] = mlx_lm.load(LLM_REPO)
            finally:
                sys.stdout, sys.stderr = _out, _err

        prompt = (
            "You clean up dictated speech-to-text. Fix punctuation, capitalization, "
            "and obvious transcription errors. Do NOT add, remove, or rephrase content. "
            "Return ONLY the corrected text with no preamble.\n\n"
            f"Text: {text}\n\nCorrected:"
        )
        tok = _LLM["tokenizer"]
        messages = [{"role": "user", "content": prompt}]
        formatted = tok.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(formatted, str):
            chat = formatted
        else:
            chat = tok.decode(formatted)

        _out, _err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            cleaned = mlx_lm.generate(
                _LLM["model"], tok, prompt=chat, max_tokens=400, verbose=False)
        finally:
            sys.stdout, sys.stderr = _out, _err

        cleaned = cleaned.strip()
        return cleaned or text, time.time() - t0
    except Exception as e:
        print(f"LLM cleanup error: {e}", file=sys.stderr)
        return text, time.time() - t0


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

            # LLM cleanup request (no audio payload)
            if "cleanup" in header:
                cleaned, secs = _llm_cleanup(header["cleanup"])
                sys.stdout.write(json.dumps({"text": cleaned, "secs": secs}) + "\n")
                sys.stdout.flush()
                continue

            n_bytes = header["n"]
            prompt  = header.get("prompt", "")
            words_wanted = header.get("words", False)
            data    = sys.stdin.buffer.read(n_bytes)
            audio   = np.frombuffer(data, dtype=np.float32)

            kwargs = dict(path_or_hf_repo=model, language=lang, verbose=False)
            if prompt:
                kwargs["initial_prompt"] = prompt
            if words_wanted:
                kwargs["word_timestamps"] = True

            result = mlx_whisper.transcribe(audio, **kwargs)
            text   = result.get("text", "").strip()
            out    = {"text": text}

            if words_wanted:
                words = []
                for seg in result.get("segments", []):
                    for wd in seg.get("words", []):
                        words.append({
                            "w":   wd.get("word", "").strip(),
                            "end": wd.get("end", 0.0),
                        })
                out["words"] = words

            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"text": "", "words": []}) + "\n")
            sys.stdout.flush()
            print(f"chunk error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
