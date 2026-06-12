"""Real-time streaming ASR via sherpa-onnx (streaming Zipformer transducer).

Unlike Whisper (batch), this consumes audio frame-by-frame and emits text
instantly with no re-transcription. Trade-off: lower accuracy than Whisper and
no casing/punctuation (we patch casing here; punctuation would need a separate
restore model or the LLM cleanup pass).
"""
import re
from pathlib import Path

import sherpa_onnx


def _pick(d: Path, prefix: str):
    # Prefer int8 (smaller/faster); fall back to fp
    return (next(d.glob(f"{prefix}*int8.onnx"), None)
            or next(d.glob(f"{prefix}*.onnx")))


class StreamingEngine:
    def __init__(self, model_dir: str, num_threads: int = 2):
        m = Path(model_dir)
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(m / "tokens.txt"),
            encoder=str(_pick(m, "encoder")),
            decoder=str(_pick(m, "decoder")),
            joiner=str(_pick(m, "joiner")),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )
        self._stream = None

    def start(self):
        self._stream = self.rec.create_stream()

    def feed(self, audio_f32) -> str:
        """Push audio (float32 @16k) and return the current partial transcript."""
        if self._stream is None:
            return ""
        self._stream.accept_waveform(16000, audio_f32)
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        return _format(self.rec.get_result(self._stream))

    def finalize(self) -> str:
        """Flush and return the final transcript (instant — nothing left to do)."""
        if self._stream is None:
            return ""
        self._stream.input_finished()
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        txt = _format(self.rec.get_result(self._stream))
        self._stream = None
        return txt


def _format(raw: str) -> str:
    """Transducer output is ALL CAPS with no punctuation — make it readable."""
    t = (raw or "").strip().lower()
    if not t:
        return ""
    # Capitalize the first letter and standalone 'i'
    t = t[0].upper() + t[1:]
    t = re.sub(r"\bi\b", "I", t)
    return t
