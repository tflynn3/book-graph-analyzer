"""Shared spaCy model loading helpers for extraction modules."""

from __future__ import annotations

import subprocess
import sys

import spacy


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load a spaCy model, downloading it into the current interpreter if needed."""
    try:
        return spacy.load(model_name)
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )
        return spacy.load(model_name)
