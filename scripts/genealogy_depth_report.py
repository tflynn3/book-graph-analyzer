from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from book_graph_analyzer.worldbible.genealogy import extract_genealogy_from_text
from book_graph_analyzer.worldbible.genealogy_validation import evaluate_genealogy_threshold

DATA_OUT = ROOT / "data" / "output"

BOOK_FILES = {
    "The Fellowship of the Ring": DATA_OUT / "fellowship_events.json",
    "The Two Towers": DATA_OUT / "twotowers_events.json",
    "The Return of the King": DATA_OUT / "return_events.json",
}


def _event_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", {})
    chunks: list[str] = []
    for ev in events.values() if isinstance(events, dict) else []:
        if not isinstance(ev, dict):
            continue
        desc = str(ev.get("description") or "").strip()
        src = str(ev.get("source_text") or "").strip()
        if desc:
            chunks.append(desc)
        if src:
            chunks.append(src)
    return ". ".join(chunks)


def _baseline_count(text: str) -> int:
    # Approximate pre-wave behavior using legacy rule subset.
    import re

    name = r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)*)"
    legacy = [
        re.compile(rf"\b{name}\s+son of\s+{name}\b"),
        re.compile(rf"\b{name}\s+daughter of\s+{name}\b"),
        re.compile(rf"\b{name}\s+child of\s+{name}\b"),
        re.compile(rf"\b{name}\s+father of\s+{name}\b"),
        re.compile(rf"\b{name}\s+mother of\s+{name}\b"),
        re.compile(rf"\b{name}\s+brother of\s+{name}\b"),
        re.compile(rf"\b{name}\s+sister of\s+{name}\b"),
        re.compile(rf"\b{name}\s+wed\s+{name}\b"),
        re.compile(rf"\b{name}\s+married\s+{name}\b"),
        re.compile(rf"\b{name}\s*,\s*son of\s+{name}\b"),
        re.compile(rf"\b{name}\s*,\s*daughter of\s+{name}\b"),
    ]
    pairs = set()
    for pat in legacy:
        for m in pat.finditer(text):
            pairs.add((m.group(1), m.group(2), pat.pattern))
            pairs.add((m.group(2), m.group(1), pat.pattern))
    return len(pairs)


def _after_count(text: str) -> int:
    rels = extract_genealogy_from_text(text, llm_client=None, min_relations_for_fallback=2)
    return len(rels)


def main() -> None:
    rows = []
    for book, path in BOOK_FILES.items():
        if not path.exists():
            rows.append({"book": book, "before": 0, "after": 0, "threshold": None, "passed": False})
            continue
        text = _event_text(path)
        before = _baseline_count(text)
        after = _after_count(text)
        gate = evaluate_genealogy_threshold(book, after)
        rows.append(
            {
                "book": book,
                "before": before,
                "after": after,
                "threshold": gate.threshold,
                "passed": gate.passed,
            }
        )

    print("book,before,after,threshold,passed")
    for r in rows:
        print(f"{r['book']},{r['before']},{r['after']},{r['threshold']},{r['passed']}")


if __name__ == "__main__":
    main()
