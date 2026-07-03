"""
Monkeypatch upstream traceloop instrumentation packages to inject request_processor.

Call patch_all() once before enabling instrumentors (i.e. before overmind.init() calls
enable_tracing). Each provider is guarded by ImportError so the package works with
whatever subset of instrumentation packages is installed.
"""
import logging
from contextvars import ContextVar
from functools import wraps

# ponytail: thread-safe conduit for reference_output between the pre-call pop
# and the post-span _set_request_attributes hook (ContextVar = async-safe too)
_pending_reference_output: ContextVar[str | None] = ContextVar(
    "_pending_reference_output", default=None
)

logger = logging.getLogger(__name__)

_patched: set[str] = set()


def patch_all() -> None:
    _patch_openai()
    _patch_anthropic()
    _patch_google_genai()
    _patch_agno()


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _patch_openai() -> None:
    if "openai" in _patched:
        return
    try:
        from opentelemetry.instrumentation.openai.shared import (
            chat_wrappers,
            completion_wrappers,
            embeddings_wrappers,
        )
        from opentelemetry.instrumentation.openai.v1 import responses_wrappers
        from opentelemetry.overmind import attrs
        from opentelemetry.overmind.processor import request_processor

        # --- chat (async _handle_request) ---
        _orig_chat = chat_wrappers._handle_request

        @wraps(_orig_chat)
        async def _chat_handle_request(span, kwargs, instance):
            try:
                request_processor(span, kwargs, "openai.chat")
            except Exception:
                logger.debug("overmind patch: openai.chat request_processor failed", exc_info=True)
            return await _orig_chat(span, kwargs, instance)

        chat_wrappers._handle_request = _chat_handle_request

        # --- completion ---
        _orig_completion = completion_wrappers._handle_request

        @wraps(_orig_completion)
        def _completion_handle_request(span, kwargs, instance):
            try:
                request_processor(span, kwargs, "openai.completions")
            except Exception:
                logger.debug("overmind patch: openai.completions request_processor failed", exc_info=True)
            return _orig_completion(span, kwargs, instance)

        completion_wrappers._handle_request = _completion_handle_request

        # --- embeddings ---
        _orig_embeddings = embeddings_wrappers._handle_request

        @wraps(_orig_embeddings)
        def _embeddings_handle_request(span, kwargs, instance):
            try:
                request_processor(span, kwargs, "openai.embeddings")
            except Exception:
                logger.debug("overmind patch: openai.embeddings request_processor failed", exc_info=True)
            return _orig_embeddings(span, kwargs, instance)

        embeddings_wrappers._handle_request = _embeddings_handle_request

        # --- responses ---
        # Step 1: wrap responses_get_or_create_wrapper to pop reference_output BEFORE
        #         the OpenAI API call (it would reject unknown kwargs otherwise).
        _orig_responses_fn = responses_wrappers.responses_get_or_create_wrapper

        @wraps(_orig_responses_fn)
        def _responses_outer(tracer):
            inner = _orig_responses_fn(tracer)

            @wraps(inner)
            def _responses_pre(wrapped, instance, args, kwargs):
                reference_output = kwargs.pop("reference_output", None)
                _pending_reference_output.set(reference_output)
                return inner(wrapped, instance, args, kwargs)

            return _responses_pre

        responses_wrappers.responses_get_or_create_wrapper = _responses_outer

        # Step 2: hook _set_request_attributes (called inside every span-creation path)
        #         to stamp the span and call request_processor.
        _orig_set_req_attrs = responses_wrappers._set_request_attributes

        @wraps(_orig_set_req_attrs)
        def _responses_set_request_attributes(span, kwargs, instance):
            try:
                reference_output = _pending_reference_output.get()
                _pending_reference_output.set(None)
                if reference_output is not None:
                    span.set_attribute(attrs.GEN_AI_REFERENCE_OUTPUT, reference_output)
                request_processor(span, kwargs, "openai.responses")
            except Exception:
                logger.debug("overmind patch: openai.responses request_processor failed", exc_info=True)
            return _orig_set_req_attrs(span, kwargs, instance)

        responses_wrappers._set_request_attributes = _responses_set_request_attributes

        _patched.add("openai")
        logger.debug("overmind patch: openai patched")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _patch_anthropic() -> None:
    if "anthropic" in _patched:
        return
    try:
        import opentelemetry.instrumentation.anthropic as anthropic_mod
        from opentelemetry.overmind.processor import request_processor

        _orig_handle_input = anthropic_mod._handle_input

        @wraps(_orig_handle_input)
        def _handle_input(span, event_logger, kwargs):
            try:
                request_processor(span, kwargs, "anthropic.messages")
            except Exception:
                logger.debug("overmind patch: anthropic request_processor failed", exc_info=True)
            return _orig_handle_input(span, event_logger, kwargs)

        anthropic_mod._handle_input = _handle_input

        _orig_ahandle_input = anthropic_mod._ahandle_input

        @wraps(_orig_ahandle_input)
        async def _ahandle_input(span, event_logger, kwargs):
            try:
                request_processor(span, kwargs, "anthropic.messages")
            except Exception:
                logger.debug("overmind patch: anthropic async request_processor failed", exc_info=True)
            return await _orig_ahandle_input(span, event_logger, kwargs)

        anthropic_mod._ahandle_input = _ahandle_input

        _patched.add("anthropic")
        logger.debug("overmind patch: anthropic patched")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Google Generative AI
# ---------------------------------------------------------------------------

def _patch_google_genai() -> None:
    if "google_genai" in _patched:
        return
    try:
        import opentelemetry.instrumentation.google_generativeai as google_mod
        from opentelemetry.overmind.processor import request_processor

        _orig_handle_request = google_mod._handle_request

        @wraps(_orig_handle_request)
        def _handle_request(span, args, kwargs, llm_model, event_logger):
            try:
                request_processor(span, kwargs, "google.genai.responses")
            except Exception:
                logger.debug("overmind patch: google_genai request_processor failed", exc_info=True)
            return _orig_handle_request(span, args, kwargs, llm_model, event_logger)

        google_mod._handle_request = _handle_request

        _orig_handle_request_async = google_mod._handle_request_async

        @wraps(_orig_handle_request_async)
        async def _handle_request_async(span, args, kwargs, llm_model, event_logger):
            try:
                request_processor(span, kwargs, "google.genai.responses")
            except Exception:
                logger.debug("overmind patch: google_genai async request_processor failed", exc_info=True)
            return await _orig_handle_request_async(span, args, kwargs, llm_model, event_logger)

        google_mod._handle_request_async = _handle_request_async

        _patched.add("google_genai")
        logger.debug("overmind patch: google_genai patched")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Agno
# ---------------------------------------------------------------------------

def _patch_agno() -> None:
    if "agno" in _patched:
        return
    try:
        import agno.agent
        from opentelemetry import trace
        from opentelemetry.overmind.processor import request_processor
        import wrapt

        # ponytail: streaming path skipped — start_span (not start_as_current_span) is
        # used, so get_current_span() won't see the agno span. Upgrade: pass span explicitly
        # into the agno instrumentation or hook inside _AgentRunWrapper.

        @wrapt.decorator
        def _run_wrapper(wrapped, instance, args, kwargs):
            span = trace.get_current_span()
            if span.is_recording():
                try:
                    prompt_kw = {"prompt": args[0]} if args else {}
                    prompt_kw.update(kwargs)
                    request_processor(span, prompt_kw, "agno.agent.run")
                except Exception:
                    logger.debug("overmind patch: agno.run request_processor failed", exc_info=True)
            return wrapped(*args, **kwargs)

        @wrapt.decorator
        async def _arun_wrapper(wrapped, instance, args, kwargs):
            span = trace.get_current_span()
            if span.is_recording():
                try:
                    prompt_kw = {"prompt": args[0]} if args else {}
                    prompt_kw.update(kwargs)
                    request_processor(span, prompt_kw, "agno.agent.run")
                except Exception:
                    logger.debug("overmind patch: agno.arun request_processor failed", exc_info=True)
            return await wrapped(*args, **kwargs)

        agno.agent.Agent.run = _run_wrapper(agno.agent.Agent.run)
        agno.agent.Agent.arun = _arun_wrapper(agno.agent.Agent.arun)

        _patched.add("agno")
        logger.debug("overmind patch: agno patched")
    except ImportError:
        pass
