"""LLM client abstraction.

Supports multiple backends:
- Ollama (local)
- Hugging Face Inference API (cloud)
"""

import json
import re
from typing import Optional

import httpx

from .config import get_settings


class LLMClient:
    """Unified LLM client supporting multiple providers.
    
    Usage:
        client = LLMClient()  # Uses config defaults
        response = client.generate("What is the capital of France?")
        
        # Or specify provider
        client = LLMClient(provider="huggingface")
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM client.
        
        Args:
            provider: "ollama" or "huggingface" (default from config)
            model: Model name (default from config)
        """
        self.settings = get_settings()
        self.provider = provider or self.settings.llm_provider
        
        if model:
            self.model = model
        elif self.provider == "huggingface":
            self.model = self.settings.hf_model
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
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Generated text, or empty string on error
        """
        if self.provider == "huggingface":
            return self._generate_hf(prompt, temperature, max_tokens, timeout)
        else:
            return self._generate_ollama(prompt, temperature, max_tokens, timeout)
    
    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Generate using Ollama."""
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
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
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            
        except (httpx.RequestError, httpx.TimeoutException) as e:
            print(f"Ollama error: {e}")
        
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
            print("HF API key not set - falling back to Ollama")
            return self._generate_ollama(prompt, temperature, max_tokens, timeout)
        
        try:
            # HF Inference API - OpenAI-compatible chat endpoint (new router)
            api_url = "https://router.huggingface.co/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.settings.hf_api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            response = httpx.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            
            if response.status_code == 200:
                result = response.json()
                # OpenAI-compatible response format
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "").strip()
                # Legacy format fallback
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    return result.get("generated_text", "").strip()
            
            elif response.status_code == 503:
                # Model loading - wait and retry
                print(f"HF model loading, retrying...")
                import time
                time.sleep(20)
                return self._generate_hf(prompt, temperature, max_tokens, timeout)
            
            else:
                print(f"HF API error {response.status_code}: {response.text[:200]}")
                
        except (httpx.RequestError, httpx.TimeoutException) as e:
            print(f"HF API error: {e}")
        
        return ""
    
    def extract_json(self, response: str) -> list | dict | None:
        """Extract JSON from LLM response.
        
        Handles markdown code blocks and stray text.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON or None
        """
        if not response:
            return None
        
        # Try to extract from code block
        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)
        
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find array or object
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
        else:
            # Check Ollama
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
