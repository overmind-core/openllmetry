import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span

from opentelemetry.overmind.attrs import OvermindAttributes
from opentelemetry.overmind.processor import MAX_PROMPT_COUNT, request_processor
from opentelemetry.overmind.prompt import PromptString


def _make_span() -> Span:
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    return tracer.start_span("test-span")


def test_openai_chat_with_promptstring_and_blocks():
    span = _make_span()

    prompt1 = PromptString(id="id1", template="Hello {name}", kwargs={"name": "Alice"})
    prompt2 = PromptString(id="id2", template="Hi {name}", kwargs={"name": "Bob"})

    kwargs = {
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": prompt1},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt2},
                ],
            },
        ]
    }

    request_processor(span, kwargs, type="openai.chat")

    # Last seen prompt should win for attributes
    assert span.attributes[OvermindAttributes.GEN_AI_PROMPT_HASH] == prompt2.hash
    assert json.loads(
        span.attributes[OvermindAttributes.GEN_AI_PROMPT_KWARGS]
    ) == prompt2.kwargs


def test_openai_responses_instructions_and_input():
    span = _make_span()

    instructions = PromptString(
        id="id_instr", template="Follow {rule}", kwargs={"rule": "rules"}
    )
    ipt = PromptString(id="id_input", template="Say {x}", kwargs={"x": "hello"})

    kwargs = {
        "instructions": instructions,
        "input": ipt,
    }

    request_processor(span, kwargs, type="openai.responses")

    # Input should be the last processed prompt
    assert span.attributes[OvermindAttributes.GEN_AI_PROMPT_HASH] == ipt.hash
    assert json.loads(
        span.attributes[OvermindAttributes.GEN_AI_PROMPT_KWARGS]
    ) == ipt.kwargs


def test_anthropic_messages_system_and_messages():
    span = _make_span()

    system_prompt = PromptString(
        id="sys", template="System {mode}", kwargs={"mode": "strict"}
    )
    message_prompt = PromptString(
        id="msg", template="Hello {name}", kwargs={"name": "User"}
    )

    kwargs = {
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": message_prompt},
        ],
    }

    request_processor(span, kwargs, type="anthropic.messages")

    # Message prompt should be the last processed prompt
    assert span.attributes[OvermindAttributes.GEN_AI_PROMPT_HASH] == message_prompt.hash
    assert json.loads(
        span.attributes[OvermindAttributes.GEN_AI_PROMPT_KWARGS]
    ) == message_prompt.kwargs


def test_google_genai_responses_contents_and_system_instruction():
    span = _make_span()

    system_instruction = PromptString(
        id="sys", template="You are {role}", kwargs={"role": "assistant"}
    )
    content_prompt = PromptString(
        id="cnt", template="Tell me about {topic}", kwargs={"topic": "testing"}
    )

    kwargs = {
        "system_instruction": system_instruction,
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": content_prompt},
                ],
            }
        ],
    }

    request_processor(span, kwargs, type="google.genai.responses")

    # Content prompt should be the last processed prompt
    assert span.attributes[OvermindAttributes.GEN_AI_PROMPT_HASH] == content_prompt.hash
    assert json.loads(
        span.attributes[OvermindAttributes.GEN_AI_PROMPT_KWARGS]
    ) == content_prompt.kwargs


def test_agno_agent_run_prompt():
    span = _make_span()

    prompt = PromptString(
        id="agno", template="Agent task: {task}", kwargs={"task": "demo"}
    )

    kwargs = {
        "prompt": prompt,
    }

    request_processor(span, kwargs, type="agno.agent.run")

    assert span.attributes[OvermindAttributes.GEN_AI_PROMPT_HASH] == prompt.hash
    assert json.loads(
        span.attributes[OvermindAttributes.GEN_AI_PROMPT_KWARGS]
    ) == prompt.kwargs


def test_max_prompt_count_enforced():
    span = _make_span()

    prompts = [
        PromptString(id=str(i), template="T {i}", kwargs={"i": i}) for i in range(10)
    ]

    kwargs = {
        "messages": [{"role": "user", "content": p} for p in prompts],
    }

    # Reduce MAX_PROMPT_COUNT impact by ensuring we exceed it
    assert MAX_PROMPT_COUNT < len(prompts)

    try:
        request_processor(span, kwargs, type="openai.chat")
    except ValueError:
        # Expected behaviour when too many prompts are present
        return

    raise AssertionError("Expected ValueError when too many prompts are processed")

