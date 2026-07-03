"""
Smoke-test for patch_all(): verifies that after patching the openai instrumentation
module, calling the patched _handle_request stamps overmind.prompt.* attributes on
the span.

Requires: opentelemetry-instrumentation-openai installed (dev dependency).
Skips cleanly if not available.
"""
import asyncio
import json
import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span

from opentelemetry.overmind import attrs
from opentelemetry.overmind.prompt import PromptString

pytest.importorskip("opentelemetry.instrumentation.openai", reason="openai instrumentation not installed")


def _make_span() -> Span:
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    return tracer.start_span("test-span")


def test_patch_openai_chat_stamps_attributes():
    from opentelemetry.overmind.patch import patch_all, _patched
    from opentelemetry.instrumentation.openai.shared import chat_wrappers

    # Reset patch state so we can apply fresh each test
    _patched.discard("openai")
    patch_all()
    assert "openai" in _patched

    span = _make_span()
    prompt = PromptString(id="p1", template="Hello {name}", kwargs={"name": "World"})
    kwargs = {"messages": [{"role": "user", "content": prompt}]}

    asyncio.run(chat_wrappers._handle_request(span, kwargs, instance=None))

    assert span.attributes.get(attrs.GEN_AI_PROMPT_HASH) == prompt.hash
    assert json.loads(span.attributes[attrs.GEN_AI_PROMPT_KWARGS]) == prompt.kwargs


def test_patch_idempotent():
    from opentelemetry.overmind.patch import patch_all, _patched

    # Calling patch_all() multiple times must not double-wrap.
    _patched.discard("openai")
    patch_all()
    from opentelemetry.instrumentation.openai.shared import chat_wrappers

    ref = chat_wrappers._handle_request
    patch_all()  # second call — already in _patched, no re-wrap
    assert chat_wrappers._handle_request is ref, "double-patching must not occur"
