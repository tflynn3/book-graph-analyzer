from types import SimpleNamespace

import httpx

from book_graph_analyzer import llm as llm_module
from book_graph_analyzer.llm import LLMClient


class _Resp:
    def __init__(self, status_code: int, data: dict | None = None, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data


def _settings(**overrides):
    base = dict(
        llm_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_base_url="https://api.openai.com/v1",
        hf_api_key="",
        hf_model="hf-model",
        ollama_model="llama3.1:8b",
        ollama_base_url="http://localhost:11434",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_openai_provider_selected_from_settings(monkeypatch):
    monkeypatch.setattr(llm_module, "get_settings", lambda: _settings())
    client = LLMClient()
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_openai_generate_parses_chat_completion(monkeypatch):
    monkeypatch.setattr(llm_module, "get_settings", lambda: _settings())

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "gpt-4o-mini"
        return _Resp(
            200,
            data={"choices": [{"message": {"content": '{"events": [], "relations": []}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = LLMClient()
    out = client.generate("prompt")
    assert '"events"' in out


def test_openai_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_module, "get_settings", lambda: _settings())

    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429, text="rate limited", headers={"Retry-After": "0"})
        return _Resp(200, data={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_: None)

    client = LLMClient()
    out = client.generate("prompt")
    assert out == "ok"
    assert calls["n"] == 2
