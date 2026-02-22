"""LLM client abstraction.

Supports multiple backends:
- Ollama (local)
- Hugging Face Inference API (cloud)
- OpenAI API (cloud)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Optional

import httpx

from .config import get_settings


logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            provider: "ollama", "huggingface", or "openai" (default from config)
            model: Model name (default from config)
        """
        self.settings = get_settings()
        self.provider = (provider or self.settings.llm_provider).lower().strip()

        if self.provider not in {"ollama", "huggingface", "openai"}:
            raise ValueError(
                f"Unknown LLM provider '{self.provider}'. Expected one of: ollama, huggingface, openai"
            )

        if model:
            self.model = model
        elif self.provider == "huggingface":
            self.model = self.settings.hf_model
        elif self.provider == "openai":
            self.model = self.settings.openai_model
        else:
            self.model = self.settings.ollama_model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: float = 120.0,
    ) -> str:
        """Generate text from prompt.

        Returns generated text, or empty string on error.
        """
        if self.provider == "huggingface":
            return self._generate_hf(prompt, temperature, max_tokens, timeout)
        if self.provider == "openai":
            return self._generate_openai(prompt, temperature, max_tokens, timeout)
        return self._generate_ollama(prompt, temperature, max_tokens, timeout)

    @property
    def provider_label(self) -> str:
        """Human-friendly provider/model label for logs."""
        return f"{self.provider}:{self.model}"

    def _post_json_with_retry(
        self,
        *,
        url: str,
        headers: Optional[dict] = None,
        payload: Optional[dict] = None,
        timeout: float,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 12.0,
    ) -> Optional[httpx.Response]:
        """POST JSON with retry/backoff for transient failures (429/5xx/network)."""
        for attempt in range(1, max_attempts + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=timeout)

                if response.status_code == 200:
                    return response

                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if not retryable:
                    logger.warning(
                        "LLM API non-retryable error status=%s provider=%s body=%s",
                        response.status_code,
                        self.provider,
                        response.text[:200],
                    )
                    return response

                if attempt == max_attempts:
                    logger.warning(
                        "LLM API retry exhausted status=%s provider=%s",
                        response.status_code,
                        self.provider,
                    )
                    return response

                # Respect Retry-After if present
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(max_delay, max(0.1, float(retry_after)))
                    except ValueError:
                        delay = min(max_delay, base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
                else:
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))

                logger.info(
                    "Retrying LLM request provider=%s status=%s attempt=%d/%d delay=%.2fs",
                    self.provider,
                    response.status_code,
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == max_attempts:
                    logger.warning(
                        "LLM request failed after retries provider=%s error=%s",
                        self.provider,
                        e,
                    )
                    return None
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
                logger.info(
                    "Retrying LLM request after transport error provider=%s attempt=%d/%d delay=%.2fs error=%s",
                    self.provider,
                    attempt,
                    max_attempts,
                    delay,
                    e,
                )
                time.sleep(delay)

        return None

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Generate using Ollama."""
        response = self._post_json_with_retry(
            url=f"{self.settings.ollama_base_url}/api/generate",
            payload={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )
        if response and response.status_code == 200:
            return response.json().get("response", "").strip()
        return ""

    def _generate_hf(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Generate using Hugging Face Inference API (OpenAI-compatible)."""
        if not self.settings.hf_api_key:
            logger.warning("HF API key not set - falling back to Ollama")
            return self._generate_ollama(prompt, temperature, max_tokens, timeout)

        response = self._post_json_with_retry(
            url="https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.hf_api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if not response or response.status_code != 200:
            return ""

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "").strip()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        if isinstance(result, dict) and "generated_text" in result:
            return result.get("generated_text", "").strip()
        return ""

    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Generate using OpenAI Chat Completions API."""
        if not self.settings.openai_api_key:
            logger.warning("OpenAI API key not set - returning empty response")
            return ""

        base_url = (self.settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        response = self._post_json_with_retry(
            url=f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if not response or response.status_code != 200:
            return ""

        result = response.json()
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    def extract_json(self, response: str) -> list | dict | None:
        """Extract JSON from LLM response."""
        if not response:
            return None

        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r"\{[\s\S]*\}", response)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @property
    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        if self.provider == "huggingface":
            return bool(self.settings.hf_api_key)
        if self.provider == "openai":
            return bool(self.settings.openai_api_key)

        try:
            response = httpx.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False


# Convenience function
def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Get an LLM client instance."""
    return LLMClient(provider=provider)
