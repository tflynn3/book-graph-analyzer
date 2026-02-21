"""
Disambiguation Dictionary

JSON-backed mapping of surface forms (aliases, epithets) to canonical
entity IDs, with context-dependent overrides.

Example entry:
  {
    "the Enemy": {
      "default": "char_sauron",
      "context_overrides": {
        "First Age": "char_morgoth",
        "Silmarillion": "char_morgoth"
      },
      "confidence": 0.90,
      "notes": "In the First Age, 'the Enemy' refers to Morgoth."
    }
  }

Context keys can be:
  - Story era:   "First Age", "Second Age", "Third Age", "Before Time"
  - Book/source: "Silmarillion", "The Hobbit", "Fellowship of the Ring", etc.
  - Any freeform key (matched case-insensitively)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in Tolkien disambiguation entries (always available as a baseline)
# ---------------------------------------------------------------------------
_BUILTIN_ENTRIES: dict[str, dict] = {
    # Gandalf cluster
    "mithrandir": {"default": "char_gandalf", "confidence": 0.99},
    "olórin": {"default": "char_gandalf", "confidence": 0.99},
    "olorin": {"default": "char_gandalf", "confidence": 0.99},
    "tharkûn": {"default": "char_gandalf", "confidence": 0.99},
    "tharkun": {"default": "char_gandalf", "confidence": 0.99},
    "the grey pilgrim": {"default": "char_gandalf", "confidence": 0.97},
    "the white rider": {"default": "char_gandalf", "confidence": 0.95},
    "gandalf the grey": {"default": "char_gandalf", "confidence": 0.99},
    "gandalf the white": {"default": "char_gandalf", "confidence": 0.99},
    "incánus": {"default": "char_gandalf", "confidence": 0.90},
    "incanus": {"default": "char_gandalf", "confidence": 0.90},

    # Sauron cluster with era override
    "the enemy": {
        "default": "char_sauron",
        "context_overrides": {
            "First Age": "char_morgoth",
            "Before Time": "char_morgoth",
            "Silmarillion": "char_morgoth",
        },
        "confidence": 0.85,
        "notes": "In the First Age, 'the Enemy' = Morgoth; in the Third Age = Sauron",
    },
    "the dark lord": {
        "default": "char_sauron",
        "context_overrides": {
            "First Age": "char_morgoth",
            "Silmarillion": "char_morgoth",
        },
        "confidence": 0.90,
    },
    "annatar": {"default": "char_sauron", "confidence": 0.99},
    "mairon": {"default": "char_sauron", "confidence": 0.99},
    "the lord of the rings": {"default": "char_sauron", "confidence": 0.95},
    "the necromancer": {"default": "char_sauron", "confidence": 0.90},

    # Aragorn cluster
    "strider": {"default": "char_aragorn", "confidence": 0.95},
    "estel": {"default": "char_aragorn", "confidence": 0.98},
    "elessar": {"default": "char_aragorn", "confidence": 0.95},
    "longshanks": {"default": "char_aragorn", "confidence": 0.85},
    "the dunadan": {"default": "char_aragorn", "confidence": 0.90},
    "the ranger": {"default": "char_aragorn", "confidence": 0.70},  # Context-dependent

    # Morgoth cluster
    "melkor": {"default": "char_morgoth", "confidence": 0.99},
    "bauglir": {"default": "char_morgoth", "confidence": 0.99},
    "the black enemy": {"default": "char_morgoth", "confidence": 0.90},

    # Frodo
    "mr. frodo": {"default": "char_frodo", "confidence": 0.97},
    "mr frodo": {"default": "char_frodo", "confidence": 0.97},
    "the ring-bearer": {"default": "char_frodo", "confidence": 0.85},

    # Gollum
    "sméagol": {"default": "char_gollum", "confidence": 0.99},
    "smeagol": {"default": "char_gollum", "confidence": 0.99},
    "déagol": {"default": "char_deagol", "confidence": 0.99},
    "deagol": {"default": "char_deagol", "confidence": 0.99},

    # Saruman
    "curunír": {"default": "char_saruman", "confidence": 0.99},
    "curunir": {"default": "char_saruman", "confidence": 0.99},
    "sharkey": {"default": "char_saruman", "confidence": 0.98},
    "the white wizard": {"default": "char_saruman", "confidence": 0.85},  # Could be Gandalf pre-turning

    # Places
    "the shire": {"default": "place_shire", "confidence": 0.99},
    "bag-end": {"default": "place_bag_end", "confidence": 0.99},
    "imladris": {"default": "place_rivendell", "confidence": 0.99},
    "karningul": {"default": "place_rivendell", "confidence": 0.99},
    "lothlórien": {"default": "place_lothlorien", "confidence": 0.99},
    "lorien": {"default": "place_lothlorien", "confidence": 0.90},
    "the golden wood": {"default": "place_lothlorien", "confidence": 0.90},
    "khazad-dûm": {"default": "place_moria", "confidence": 0.99},
    "the black land": {"default": "place_mordor", "confidence": 0.95},
    "the land of shadow": {"default": "place_mordor", "confidence": 0.90},

    # Objects
    "the one ring": {"default": "obj_one_ring", "confidence": 0.99},
    "the ruling ring": {"default": "obj_one_ring", "confidence": 0.99},
    "isildur's bane": {"default": "obj_one_ring", "confidence": 0.92},
    "anduril": {"default": "obj_anduril", "confidence": 0.99},
    "the sword that was broken": {"default": "obj_narsil", "confidence": 0.90},
    "glamdring": {"default": "obj_glamdring", "confidence": 0.99},
    "sting": {"default": "obj_sting", "confidence": 0.99},
}


class DisambiguationDict:
    """
    Context-aware entity disambiguation dictionary.

    Maps surface forms (lower-cased) to canonical entity IDs.
    Supports context overrides based on story era or source book.

    Usage:
        d = DisambiguationDict()
        entity_id, conf = d.resolve("the Enemy", era="First Age")
        # → ("char_morgoth", 0.85)

        d.load(Path("data/disambiguation.json"))  # Merge with file
        d.save(Path("data/disambiguation.json"))   # Persist
    """

    def __init__(self, load_builtins: bool = True) -> None:
        self._entries: dict[str, dict] = {}
        if load_builtins:
            self._entries.update({k: dict(v) for k, v in _BUILTIN_ENTRIES.items()})

    def load(self, path: Path) -> None:
        """Load and merge entries from a JSON file."""
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for surface, entry in data.items():
            # Existing entries are overwritten by file (file = human override)
            self._entries[surface.lower()] = entry
        logger.info("Loaded %d disambiguation entries from %s", len(data), path)

    def save(self, path: Path) -> None:
        """Save all entries to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d disambiguation entries to %s", len(self._entries), path)

    def add(
        self,
        surface: str,
        canonical_id: str,
        confidence: float = 0.85,
        context_overrides: Optional[dict[str, str]] = None,
        notes: str = "",
        overwrite: bool = False,
    ) -> None:
        """
        Add or update an entry.

        Args:
            surface: Surface form (case-insensitive)
            canonical_id: Default canonical entity ID
            confidence: Confidence for the default resolution
            context_overrides: {context_key: canonical_id} for context-dep resolution
            notes: Human-readable notes
            overwrite: If True, replace existing entry; if False, skip if exists
        """
        key = surface.lower().strip()
        if key in self._entries and not overwrite:
            return
        entry: dict = {"default": canonical_id, "confidence": confidence}
        if context_overrides:
            entry["context_overrides"] = context_overrides
        if notes:
            entry["notes"] = notes
        self._entries[key] = entry

    def resolve(
        self,
        surface: str,
        era: Optional[str] = None,
        book: Optional[str] = None,
    ) -> tuple[Optional[str], float]:
        """
        Resolve a surface form to a canonical entity ID.

        Context is checked in priority order:
        1. Exact era match in context_overrides
        2. Exact book match in context_overrides
        3. Partial era/book match (e.g., "First" matches "First Age")
        4. Default mapping

        Returns:
            (canonical_id, confidence) or (None, 0.0) if not found
        """
        key = surface.lower().strip()

        # Try with "the " prefix stripped
        entry = self._entries.get(key)
        if entry is None and key.startswith("the "):
            entry = self._entries.get(key[4:])
        if entry is None:
            return None, 0.0

        overrides = entry.get("context_overrides", {})
        base_confidence = float(entry.get("confidence", 0.85))

        # Check context overrides
        if overrides:
            # Build a combined context string for partial matching
            context_keys = []
            if era:
                context_keys.append(era)
            if book:
                context_keys.append(book)

            for ctx_key, ctx_value in overrides.items():
                ctx_key_lower = ctx_key.lower()
                for ck in context_keys:
                    if ctx_key_lower in ck.lower() or ck.lower() in ctx_key_lower:
                        # Context match — return override with slightly lower confidence
                        return ctx_value, base_confidence * 0.95

        # Return default
        default = entry.get("default")
        if default:
            return default, base_confidence

        return None, 0.0

    def has_entry(self, surface: str) -> bool:
        """Check if a surface form has an entry."""
        key = surface.lower().strip()
        if key in self._entries:
            return True
        if key.startswith("the ") and key[4:] in self._entries:
            return True
        return False

    def get_all_surfaces_for(self, canonical_id: str) -> list[str]:
        """Return all surface forms that map to a given canonical ID."""
        return [
            surface
            for surface, entry in self._entries.items()
            if entry.get("default") == canonical_id
            or canonical_id in entry.get("context_overrides", {}).values()
        ]

    def stats(self) -> dict:
        """Return stats about the dictionary."""
        with_overrides = sum(1 for e in self._entries.values() if "context_overrides" in e)
        return {
            "total_entries": len(self._entries),
            "with_context_overrides": with_overrides,
            "builtin_entries": len(_BUILTIN_ENTRIES),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, surface: str) -> bool:
        return self.has_entry(surface)
