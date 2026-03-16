class OvermindAttributes:
    # Prompt attributes
    GEN_AI_PROMPT_ID = "overmind.prompt.id"
    GEN_AI_PROMPT_HASH = "overmind.prompt.hash"
    GEN_AI_PROMPT_VERSION = "overmind.prompt.version"
    GEN_AI_PROMPT_KWARGS = "overmind.prompt.kwargs"
    GEN_AI_PROMPT_TEMPLATE = "overmind.prompt.template"
    GEN_AI_REFERENCE_OUTPUT = "overmind.reference.output"
    GEN_AI_REASONING_EFFORT = "overmind.reasoning.effort"

    # Request attributes
    GEN_AI_REQUEST_REASONING_EFFORT = "gen_ai.request.reasoning_effort"
    GEN_AI_REQUEST_MAX_COMPLETION_TOKENS = "gen_ai.request.max_completion_tokens"
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_REQUEST_RESPONSE_FORMAT = "gen_ai.request.response_format"
    GEN_AI_REQUEST_SERVICE_TIER = "gen_ai.request.service_tier"
    GEN_AI_REQUEST_PARALLEL_TOOL_CALLS = "gen_ai.request.parallel_tool_calls"
    GEN_AI_REQUEST_TOOL_CHOICE = "gen_ai.request.tool_choice"
    GEN_AI_REQUEST_LOGPROBS = "gen_ai.request.logprobs"
    GEN_AI_REQUEST_TOP_LOGPROBS = "gen_ai.request.top_logprobs"
    GEN_AI_REQUEST_MODALITIES = "gen_ai.request.modalities"

    # Response attributes
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_SYSTEM_FINGERPRINT = "gen_ai.response.system_fingerprint"
    GEN_AI_RESPONSE_SERVICE_TIER = "gen_ai.response.service_tier"

    # Usage detail attributes
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    GEN_AI_USAGE_AUDIO_TOKENS = "gen_ai.usage.audio_tokens"
    GEN_AI_USAGE_ACCEPTED_PREDICTION_TOKENS = "gen_ai.usage.accepted_prediction_tokens"
    GEN_AI_USAGE_REJECTED_PREDICTION_TOKENS = "gen_ai.usage.rejected_prediction_tokens"
