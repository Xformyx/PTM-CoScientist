"""
Unified LLM Client for PTM-CoScientist.

Supports Ollama (local), OpenAI, and Gemini with auto-fallback.
Adapted from PTM-platform's workers/common/llm_client.py for compatibility.
"""

import logging
import time
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM interface supporting Ollama, OpenAI, and Gemini."""

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        ollama_url: str = "http://host.docker.internal:11434",
        openai_api_key: str = "",
        openai_model: str = "gpt-4.1-mini",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
    ):
        self.provider = provider
        self.ollama_url = ollama_url.rstrip("/")
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = "https://api.openai.com/v1"
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        self._model = model or "gemma3:27b"
        self._resolved_provider: Optional[str] = None

    def _resolve_provider(self) -> str:
        """Resolve which provider to use (auto mode tries in order)."""
        if self._resolved_provider:
            return self._resolved_provider

        if self.provider != "auto":
            self._resolved_provider = self.provider
            return self.provider

        # Auto: Ollama → OpenAI → Gemini
        if self._check_ollama():
            self._resolved_provider = "ollama"
        elif self.openai_api_key:
            self._resolved_provider = "openai"
        elif self.gemini_api_key:
            self._resolved_provider = "gemini"
        else:
            self._resolved_provider = "ollama"  # fallback

        logger.info(f"[LLM] Resolved provider: {self._resolved_provider}")
        return self._resolved_provider

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        provider = self._resolve_provider()
        if provider == "ollama":
            return self._check_ollama()
        elif provider == "openai":
            return bool(self.openai_api_key)
        elif provider == "gemini":
            return bool(self.gemini_api_key)
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text from the resolved LLM provider."""
        provider = self._resolve_provider()

        if provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, temperature, max_tokens)
        elif provider == "openai":
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)
        elif provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
        else:
            raise RuntimeError(f"Unknown LLM provider: {provider}")

    def _generate_ollama(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def _generate_openai(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={
                "model": self.openai_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _generate_gemini(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.gemini_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.gemini_api_key}"},
            json={
                "model": self.gemini_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
