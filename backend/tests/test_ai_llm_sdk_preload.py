"""The LLM SDK's lazy imports must never run on the event loop.

``client.chat`` imports several hundred modules on first access. On the
event loop that froze the whole server for the duration (seen in a real
server run: every request, the progress stream included, stopped until the
import finished).
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from backend.app.ai import evaluator


class _Client:
    """Minimal chat client that records the thread of the request."""

    def __init__(self, calls: list) -> None:
        self._calls = calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self._calls.append(("request", threading.get_ident()))
        message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def test_first_request_loads_the_sdk_in_a_worker_thread_once(monkeypatch):
    calls: list = []

    def fake_preload() -> None:
        calls.append(("preload", threading.get_ident()))
        monkeypatch.setattr(evaluator, "_sdk_loaded", True)

    monkeypatch.setattr(evaluator, "_sdk_loaded", False)
    monkeypatch.setattr(evaluator, "preload_sdk", fake_preload)
    monkeypatch.setattr(evaluator, "get_shared_client", lambda: _Client(calls))

    async def run() -> int:
        request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        await evaluator._send(request, "test", None)
        await evaluator._send(request, "test", None)
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    kinds = [kind for kind, _ in calls]
    assert kinds == ["preload", "request", "request"]   # loaded once, before the first request
    assert calls[0][1] != loop_thread                   # not on the event loop
    assert calls[1][1] == loop_thread


def test_preload_sdk_imports_the_chat_resources(monkeypatch):
    import sys

    monkeypatch.setattr(evaluator, "_sdk_loaded", False)
    evaluator.preload_sdk()
    assert "openai.resources.chat" in sys.modules
    assert evaluator._sdk_loaded is True
