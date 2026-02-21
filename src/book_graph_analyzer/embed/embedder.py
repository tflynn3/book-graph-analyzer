"""
Embedding Model Abstraction

Supports two providers:
- sentence-transformers (default, local, no extra services)
- ollama (if already running, uses nomic-embed-text)

The model is lazy-loaded on first call.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default sentence-transformers model: small (80MB), fast, good quality
_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"
# Ollama model name for embeddings
_DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


class Embedder:
    """
    Text → vector embedder with pluggable backend.

    Usage:
        embedder = Embedder()                           # auto-detect
        embedder = Embedder(provider='sentence-transformers')
        embedder = Embedder(provider='ollama', model='nomic-embed-text')

        vectors = embedder.embed(["text1", "text2"])
        vec = embedder.embed_one("single text")
    """

    def __init__(
        self,
        provider: str = "sentence-transformers",
        model: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.provider = provider
        self.model_name = model or (
            _DEFAULT_ST_MODEL if provider == "sentence-transformers"
            else _DEFAULT_OLLAMA_MODEL
        )
        self.batch_size = batch_size
        self._model = None  # Lazy-loaded

    def _load_model(self) -> None:
        """Load the embedding model (called on first use)."""
        if self._model is not None:
            return

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence-transformers model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Model loaded. Embedding dim: %d", self.embedding_dim)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from e
        elif self.provider == "ollama":
            # Ollama: we use the REST API directly (no extra library)
            import urllib.request
            import json as _json
            # Verify Ollama is reachable
            try:
                from book_graph_analyzer.config import get_settings
                settings = get_settings()
                base_url = settings.ollama_base_url
            except Exception:
                base_url = "http://localhost:11434"
            self._ollama_url = f"{base_url}/api/embeddings"
            self._model = "ollama"  # Sentinel
            logger.info("Using Ollama embeddings at %s model=%s", self._ollama_url, self.model_name)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Use 'sentence-transformers' or 'ollama'.")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension for the current model."""
        if self.provider == "sentence-transformers":
            if self._model is not None:
                return self._model.get_sentence_embedding_dimension()
            # Known dimensions for common models
            known = {
                "all-MiniLM-L6-v2": 384,
                "all-MiniLM-L12-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-MiniLM-L6-v2-quantized": 384,
            }
            return known.get(self.model_name, 384)
        elif self.provider == "ollama":
            return 768  # nomic-embed-text default

        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Returns list of float vectors, one per input text.
        Processes in batches for efficiency.
        """
        if not texts:
            return []

        self._load_model()

        if self.provider == "sentence-transformers":
            return self._embed_st(texts)
        elif self.provider == "ollama":
            return self._embed_ollama(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = self.embed([text])
        return results[0] if results else []

    def _embed_st(self, texts: list[str]) -> list[list[float]]:
        """Embed using sentence-transformers."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """Embed using Ollama API."""
        import json as _json
        import urllib.request
        import urllib.error

        all_embeddings = []
        for text in texts:
            payload = _json.dumps({
                "model": self.model_name,
                "prompt": text,
            }).encode("utf-8")
            req = urllib.request.Request(
                self._ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read())
                    all_embeddings.append(data["embedding"])
            except Exception as e:
                logger.error("Ollama embedding failed for text: %s... — %s", text[:50], e)
                # Return zero vector as fallback
                all_embeddings.append([0.0] * self.embedding_dim)
        return all_embeddings

    @classmethod
    def from_settings(cls) -> "Embedder":
        """Create an Embedder from application settings."""
        try:
            import os
            provider = os.environ.get("BGA_EMBEDDING_PROVIDER", "sentence-transformers")
            model = os.environ.get("BGA_EMBEDDING_MODEL", None)
            return cls(provider=provider, model=model)
        except Exception:
            return cls()  # Default
