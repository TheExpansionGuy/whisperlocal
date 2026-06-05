"""Learning loop for WhisperLocal.

Captures (what the app produced) → (what the user corrected it to) pairs, and:
  1. Stores every pair to corrections.jsonl (raw training data)
  2. Derives word/phrase-level replacements → learned.json (instant corrections)
  3. Exposes recent pairs as few-shot examples for the LLM cleanup prompt
"""
import json
import re
import wave
import difflib
from pathlib import Path

import numpy as np

DIR          = Path.home() / ".whisperlocal"
PAIRS_PATH   = DIR / "corrections.jsonl"
LEARNED_PATH = DIR / "learned.json"
METRICS_PATH = DIR / "metrics.jsonl"   # per-dictation outcome (edited?, similarity)
PROMOTE_AFTER = 2          # a replacement becomes a standing correction after N sightings


def record_outcome(produced: str, final: str) -> bool:
    """Log one dictation outcome. Returns True if the user edited it.
    The edit rate over time is our accuracy metric (less editing = better)."""
    produced = (produced or "").strip()
    final = (final or "").strip()
    ratio = difflib.SequenceMatcher(a=produced, b=final).ratio() if produced or final else 1.0
    edited = produced != final
    DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a") as f:
        f.write(json.dumps({"edited": edited, "ratio": round(ratio, 4)}) + "\n")
    return edited


def _metric_rows():
    try:
        return [json.loads(l) for l in METRICS_PATH.read_text().splitlines() if l.strip()]
    except Exception:
        return []


def accuracy_pct(n: int = 30) -> int:
    """Recent accuracy = average similarity between produced and confirmed text."""
    rows = _metric_rows()[-n:]
    if not rows:
        return 100
    return int(round(100 * sum(r.get("ratio", 1.0) for r in rows) / len(rows)))


def edit_rate_pct(n: int = 30) -> int:
    rows = _metric_rows()[-n:]
    if not rows:
        return 0
    return int(round(100 * sum(1 for r in rows if r.get("edited")) / len(rows)))


def accuracy_curve(buckets: int = 8):
    """Average accuracy per time-bucket, oldest→newest, for the graph."""
    rows = _metric_rows()
    if not rows:
        return []
    size = max(1, len(rows) // buckets)
    out = []
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        out.append(int(round(100 * sum(r.get("ratio", 1.0) for r in chunk) / len(chunk))))
    return out[-buckets:]

# --- Ambient training corpus (audio → text pairs for future fine-tuning) ----
TRAIN_DIR    = DIR / "training"
MANIFEST     = TRAIN_DIR / "manifest.jsonl"
MAX_SAMPLES  = 1000        # hard cap; oldest are pruned (~ up to a few hundred MB)


def save_sample(audio_f32, sample_rate: int, text: str) -> int:
    """Save one (audio, text) training sample as 16-bit WAV + manifest line.
    Prunes oldest samples beyond MAX_SAMPLES. Returns total sample count."""
    text = (text or "").strip()
    if not text or audio_f32 is None or len(audio_f32) < sample_rate * 0.3:
        return sample_count()
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Stable, monotonic-ish id from existing count + content length
    idx = _next_index()
    name = f"s{idx:06d}.wav"
    pcm = np.clip(np.asarray(audio_f32) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(TRAIN_DIR / name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

    with MANIFEST.open("a") as f:
        f.write(json.dumps({"audio": name, "text": text, "corrected": False}) + "\n")

    _prune()
    return sample_count()


def update_last_text(corrected: str):
    """Mark the most recent sample with the user's corrected text (better label)."""
    rows = _manifest_rows()
    if not rows:
        return
    rows[-1]["text"] = corrected.strip()
    rows[-1]["corrected"] = True
    MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in rows))


def sample_count() -> int:
    return len(_manifest_rows())


# --- Gamified progress -------------------------------------------------------
SAMPLES_PER_LEVEL = 50      # verified dictations per training milestone


def stats() -> dict:
    """Progress info for the gamified 'voice model' display.
    XP/level comes from total dictations (engagement); training samples are
    only the edited ones; accuracy comes from the edit-rate metric."""
    metrics = _metric_rows()
    total_dictations = len(metrics)
    train_samples = len(_manifest_rows())   # only edited dictations are saved

    level = total_dictations // SAMPLES_PER_LEVEL
    into_level = total_dictations % SAMPLES_PER_LEVEL
    progress = into_level / SAMPLES_PER_LEVEL if SAMPLES_PER_LEVEL else 0
    return {
        "dictations": total_dictations,
        "train_samples": train_samples,
        "accuracy": accuracy_pct(),
        "edit_rate": edit_rate_pct(),
        "level": level,
        "into_level": into_level,
        "to_next": SAMPLES_PER_LEVEL - into_level,
        "progress": progress,
        "ready_to_train": into_level == 0 and total_dictations > 0 and train_samples > 0,
    }


def progress_bar(progress: float, width: int = 10) -> str:
    filled = int(round(progress * width))
    return "▓" * filled + "░" * (width - filled)


_SPARK = "▁▂▃▄▅▆▇█"

def sparkline(values, lo=70, hi=100) -> str:
    """Render a list of 0-100 values as a unicode sparkline (clamped to lo..hi)."""
    if not values:
        return ""
    out = []
    for v in values:
        t = (max(lo, min(hi, v)) - lo) / max(1, (hi - lo))
        out.append(_SPARK[min(len(_SPARK) - 1, int(t * (len(_SPARK) - 1)))])
    return "".join(out)


def corpus_bytes() -> int:
    try:
        return sum(p.stat().st_size for p in TRAIN_DIR.glob("*.wav"))
    except Exception:
        return 0


def _manifest_rows():
    try:
        return [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    except Exception:
        return []


def _next_index() -> int:
    rows = _manifest_rows()
    if not rows:
        return 0
    try:
        return int(rows[-1]["audio"][1:7]) + 1
    except Exception:
        return len(rows)


def _prune():
    rows = _manifest_rows()
    if len(rows) <= MAX_SAMPLES:
        return
    drop = rows[:len(rows) - MAX_SAMPLES]
    for r in drop:
        try:
            (TRAIN_DIR / r["audio"]).unlink()
        except Exception:
            pass
    keep = rows[len(rows) - MAX_SAMPLES:]
    MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in keep))


def _load_learned() -> dict:
    try:
        return json.loads(LEARNED_PATH.read_text())
    except Exception:
        return {}


def _save_learned(d: dict):
    DIR.mkdir(parents=True, exist_ok=True)
    LEARNED_PATH.write_text(json.dumps(d, indent=2))


def learned_corrections() -> dict:
    """Return {wrong_phrase: right_phrase} that have been seen enough to apply."""
    data = _load_learned()
    return {k: v["to"] for k, v in data.items() if v.get("count", 0) >= PROMOTE_AFTER}


def _word_diffs(before: str, after: str):
    """Yield (old_phrase, new_phrase) for each replaced span."""
    a = before.split()
    b = after.split()
    sm = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b])
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            old = " ".join(a[i1:i2]).strip()
            new = " ".join(b[j1:j2]).strip()
            # Keep short, sensible replacements (avoid whole-sentence rewrites)
            if old and new and len(old) <= 40 and i2 - i1 <= 4 and j2 - j1 <= 4:
                yield old, new


def record_correction(produced: str, corrected: str) -> int:
    """Store the pair, learn replacements. Returns number of new replacements learned."""
    produced = (produced or "").strip()
    corrected = (corrected or "").strip()
    if not corrected or produced == corrected:
        return 0

    DIR.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("a") as f:
        f.write(json.dumps({"produced": produced, "corrected": corrected}) + "\n")

    learned = _load_learned()
    n_new = 0
    for old, new in _word_diffs(produced, corrected):
        key = old.lower()
        entry = learned.get(key, {"to": new, "count": 0})
        entry["to"] = new            # latest correction wins
        entry["count"] = entry.get("count", 0) + 1
        if entry["count"] == PROMOTE_AFTER:
            n_new += 1
        learned[key] = entry
    _save_learned(learned)
    return n_new


def few_shot_examples(n: int = 3):
    """Return up to n recent (produced, corrected) pairs for LLM context."""
    try:
        lines = PAIRS_PATH.read_text().strip().splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-n:]:
        try:
            d = json.loads(line)
            out.append((d["produced"], d["corrected"]))
        except Exception:
            pass
    return out
