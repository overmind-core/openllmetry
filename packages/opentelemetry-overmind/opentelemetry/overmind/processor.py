import json
from opentelemetry.trace import Span
from opentelemetry.overmind.prompt import Prompt
from opentelemetry.overmind.attrs import OvermindAttributes


def request_processor(span: Span, kwargs: dict, type: str = "openai.chat") -> None:
    """
    types: openai.chat | openai.response | anthropic.response | etc.
    """
    if reference_output := kwargs.pop("reference_output", None):
        span.set_attribute(OvermindAttributes.GEN_AI_REFERENCE_OUTPUT, reference_output)

    if type == "openai.chat":
        messages = kwargs["messages"]
        for message in messages:
            content = message["content"]
            if isinstance(content, Prompt):
                span.set_attribute(OvermindAttributes.GEN_AI_PROMPT_HASH, content.hash)
                span.set_attribute(
                    OvermindAttributes.GEN_AI_PROMPT_KWARGS, json.dumps(content.kwargs)
                )
            message["content"] = str(content)

    if type == "openai.response":
        ...

    if type == "anthropic.response":
        ...
