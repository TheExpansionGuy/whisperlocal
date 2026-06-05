"""Learning loop for WhisperLocal.

Captures (what the app produced) → (what the user corrected it to) pairs, and:
  1. Stores every pair to corrections.jsonl (raw training data)
  2. Derives word/phrase-level replacements → learned.json (instant corrections)
  3. Exposes recent pairs as few-shot examples for the LLM cleanup prompt
"""
import json
import re
import difflib
from pathlib import Path

DIR          = Path.home() / ".whisperlocal"
PAIRS_PATH   = DIR / "corrections.jsonl"
LEARNED_PATH = DIR / "learned.json"
PROMOTE_AFTER = 2          # a replacement becomes a standing correction after N sightings


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
