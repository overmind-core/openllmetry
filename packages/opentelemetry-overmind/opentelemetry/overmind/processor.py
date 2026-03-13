import json
from collections.abc import Mapping, Sequence

from opentelemetry.trace import Span
from opentelemetry.overmind.prompt import PromptString
from opentelemetry.overmind.attrs import OvermindAttributes


def _store_prompt_attributes(span: Span, content: PromptString | str) -> None:
    if isinstance(content, PromptString):
        span.set_attribute(OvermindAttributes.GEN_AI_PROMPT_ID, content.id)
        span.set_attribute(OvermindAttributes.GEN_AI_PROMPT_HASH, content.hash)
        span.set_attribute(OvermindAttributes.GEN_AI_PROMPT_KWARGS, json.dumps(content.kwargs))


MAX_PROMPT_COUNT = 1


def _walk_and_store_prompts(span: Span, value, counter: list[int]) -> None:
    """
    Recursively walk an arbitrary value (PromptString, str, list, dict, etc.)
    and store prompt attributes for any PromptString instances found.
    """
    if isinstance(value, PromptString):
        _store_prompt_attributes(span, value)
        counter[0] += 1
        return

    # Plain strings are not traced unless wrapped in PromptString
    if isinstance(value, str):
        return

    if isinstance(value, Mapping):
        for v in value.values():
            _walk_and_store_prompts(span, v, counter)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _walk_and_store_prompts(span, item, counter)
        return


def clean_kwargs(kwargs: dict) -> dict:
    """remove additional keys from kwargs, so that the default methods can be used, but returns a copy of the original args so tracing can work"""

    reference_output = kwargs.pop("reference_output", None)
    return {"reference_output": reference_output, **kwargs}


def request_processor(span: Span, kwargs: dict, type: str = "openai.chat") -> None:
    """
    Normalize prompt capture across providers.

    Supported types:
      - openai.chat
      - openai.responses
      - anthropic.messages
      - google.genai.responses
      - agno.agent.run
    """
    if reference_output := kwargs.pop("reference_output", None):
        span.set_attribute(OvermindAttributes.GEN_AI_REFERENCE_OUTPUT, reference_output)

    prompt_count = [0]

    if type == "openai.chat":
        # OpenAI chat: classic messages and newer content blocks
        for message in kwargs.get("messages", []):
            content = message.get("content")
            _walk_and_store_prompts(span, content, prompt_count)

        if prompt_count[0] > MAX_PROMPT_COUNT:
            raise ValueError("Too many prompts")

    elif type == "openai.response":
        # OpenAI Responses API: instructions + input (and potentially messages/system)
        if "instructions" in kwargs:
            _walk_and_store_prompts(span, kwargs["instructions"], prompt_count)
        if "input" in kwargs:
            _walk_and_store_prompts(span, kwargs["input"], prompt_count)

    elif type == "openai.embeddings":
        if "input" in kwargs:
            _store_prompt_attributes(span, kwargs["input"])

    elif type == "anthropic.messages":
        # Anthropic messages API (including beta and Bedrock)
        if "system" in kwargs:
            _walk_and_store_prompts(span, kwargs["system"], prompt_count)
        for message in kwargs.get("messages", []):
            _walk_and_store_prompts(span, message.get("content"), prompt_count)

    elif type == "google.genai.responses":
        # Google Generative AI / Gemini generate_content
        if "system_instruction" in kwargs:
            _walk_and_store_prompts(span, kwargs["system_instruction"], prompt_count)
        if "contents" in kwargs:
            _walk_and_store_prompts(span, kwargs["contents"], prompt_count)

    elif type == "agno.agent.run":
        # Agno Agent / Team run entrypoints
        for key in ("prompt", "system_prompt", "messages", "input"):
            if key in kwargs:
                _walk_and_store_prompts(span, kwargs[key], prompt_count)
