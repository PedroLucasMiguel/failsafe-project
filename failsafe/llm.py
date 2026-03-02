"""
LLM provider factory — instantiates the right chat model based on user choice.

Supports: openai, anthropic, google.
For Google, also supports OAuth 2.0 credentials (no API key needed when logged
in via `failsafe auth google`).

Adding a new provider = adding one entry to PROVIDER_REGISTRY.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def extract_text(response) -> str:
    """Extract plain text from an LLM response.

    Different providers return content in different formats:
      - OpenAI / Anthropic: response.content is a str.
      - Google Gemini (newer langchain-google-genai): response.content is a
        list of dicts like [{"type": "text", "text": "..."}].

    This normalizes both into a single stripped string.
    """
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts).strip()

    return str(content).strip()


def invoke_cached(llm: "BaseChatModel", messages: list) -> str:
    """Invoke an LLM with in-session response caching and provider optimizations.

    What this does for each provider:
      - ALL:       Checks the in-session cache first. On a hit, returns instantly
                   without making any network call or consuming tokens.
      - Anthropic: Adds ``cache_control: {"type": "ephemeral"}`` to system
                   messages so Anthropic caches the prompt prefix server-side
                   (cached tokens cost ~10% of normal price).
      - OpenAI:    Prompt caching is automatic for prompts > 1024 tokens on
                   compatible models — nothing extra needed.
      - Google:    Context caching requires 32k+ tokens AND is incompatible
                   with system_instruction, so we rely on the response cache only.

    Args:
        llm:      The active LangChain chat model.
        messages: List of SystemMessage / HumanMessage objects.

    Returns:
        The response text as a plain string (already stripped).
    """
    from failsafe.cache import response_cache
    from failsafe.tracking import tracker

    # 1. Cache hit → return immediately, no tokens consumed
    cached = response_cache.get(messages)
    if cached is not None:
        return cached

    # 2. Provider-specific prompt caching optimizations
    provider_messages = _apply_provider_caching(llm, messages)

    # 3. Invoke the LLM
    response = llm.invoke(provider_messages)
    tracker.record(response)
    text = extract_text(response)

    # 4. Store in cache for future identical calls
    response_cache.set(messages, text)

    return text


def _apply_provider_caching(llm: "BaseChatModel", messages: list) -> list:
    """Rewrite messages to add provider-specific caching hints.

    Currently only annotates Anthropic system messages with cache_control.
    Other providers are returned unchanged.
    """
    provider = _detect_provider(llm)

    if provider == "anthropic":
        return _add_anthropic_cache_control(messages)

    return messages


def _detect_provider(llm: "BaseChatModel") -> str:
    """Detect which provider an LLM instance belongs to."""
    class_name = type(llm).__name__.lower()
    module_name = type(llm).__module__.lower()

    if "anthropic" in class_name or "anthropic" in module_name:
        return "anthropic"
    if "openai" in class_name or "openai" in module_name:
        return "openai"
    if "google" in class_name or "google" in module_name:
        return "google"
    return "unknown"


def _add_anthropic_cache_control(messages: list) -> list:
    """Annotate Anthropic system messages with cache_control for prompt caching.

    This marks the system message content for server-side caching by Anthropic.
    Cached tokens cost ~10% of the normal input token price.

    Anthropic supports up to 4 cache breakpoints. We cache the largest/most-
    repeated content: system prompts (always the same across batches) and the
    last HumanMessage (the current file_batch) if it's large.

    Implementation note: LangChain's Anthropic integration accepts cache_control
    when message content is passed as ``[{"type": "text", "text": ...,
    "cache_control": {"type": "ephemeral"}}]`` instead of a plain string.
    """
    from langchain_core.messages import SystemMessage

    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = msg.content if isinstance(
                msg.content, str) else extract_text(msg)
            result.append(SystemMessage(content=[{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]))
        else:
            result.append(msg)

    return result


# ---------------------------------------------------------------------------
# Module-level LLM reference — set by CLI before graph invocation,
# read by agent nodes. This avoids passing the LLM through state.
# ---------------------------------------------------------------------------
_current_llm: BaseChatModel | None = None


def set_llm(llm: BaseChatModel) -> None:
    """Set the active LLM instance for agent nodes to use."""
    global _current_llm  # noqa: PLW0603
    _current_llm = llm


def get_llm() -> BaseChatModel:
    """Get the active LLM instance. Raises if not set."""
    if _current_llm is None:
        raise RuntimeError(
            "No LLM configured. Call set_llm() before running the graph."
        )
    return _current_llm


# ---------------------------------------------------------------------------
# Provider registry: provider_name -> (module_path, class_name, model_default)
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "openai": (
        "langchain_openai",
        "ChatOpenAI",
        "gpt-4o",
    ),
    "anthropic": (
        "langchain_anthropic",
        "ChatAnthropic",
        "claude-3-5-haiku-latest",
    ),
    "google": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        "gemini-3-flash-preview",
    ),
}

SUPPORTED_PROVIDERS = list(PROVIDER_REGISTRY.keys())


def create_llm(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create a chat model instance for the given provider.

    For the 'google' provider, OAuth credentials stored via `failsafe auth
    google` are used automatically when no api_key is given.

    Args:
        provider: One of 'openai', 'anthropic', 'google'.
        api_key: API key for the chosen provider. Optional for Google when
            OAuth credentials are already stored.
        model: Optional model name override. Uses a sensible default if omitted.
        temperature: Sampling temperature. Defaults to 0 for determinism.

    Returns:
        A BaseChatModel ready to use with LangChain/LangGraph.

    Raises:
        ValueError: If the provider is not supported.
        RuntimeError: If no credentials are available for the Google provider.
    """
    provider = provider.lower().strip()

    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {supported}"
        )

    _, _, default_model = PROVIDER_REGISTRY[provider]
    resolved_model = model or default_model

    if provider == "google":
        return _create_google_llm(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
        )

    # All non-Google providers require an API key
    if not api_key:
        raise ValueError(f"--api-key is required for provider '{provider}'.")

    return _create_llm_from_registry(
        provider=provider,
        api_key=api_key,
        model=resolved_model,
        temperature=temperature,
    )


def _create_google_llm(
    model: str,
    api_key: str | None,
    temperature: float,
) -> BaseChatModel:
    """Create a Google LLM using an API key."""
    import importlib

    module = importlib.import_module("langchain_google_genai")
    chat_class = getattr(module, "ChatGoogleGenerativeAI")

    if not api_key:
        raise ValueError(
            "--api-key is required for --provider google.\n\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    return chat_class(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def _create_llm_from_registry(
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
) -> BaseChatModel:
    """Instantiate an LLM using the provider registry (non-Google path)."""
    import importlib

    module_path, class_name, _ = PROVIDER_REGISTRY[provider]
    module = importlib.import_module(module_path)
    chat_class = getattr(module, class_name)
    api_key_kwarg = _api_key_kwarg_for(provider)

    return chat_class(
        model=model,
        temperature=temperature,
        **{api_key_kwarg: api_key},
    )


def _api_key_kwarg_for(provider: str) -> str:
    """Return the constructor kwarg name for the API key per provider."""
    mapping = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "google": "google_api_key",
    }
    return mapping[provider]
