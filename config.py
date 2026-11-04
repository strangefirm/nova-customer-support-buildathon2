"""Central configuration. All secrets come from the environment / .env, never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    embedding_model: str
    livekit_url: str  # wss://<project>.livekit.cloud
    livekit_api_key: str
    livekit_api_secret: str
    voice_agent_name: str  # LiveKit agent name (voice_agent.py)
    voice_stt_language: str  # "multi" (default: detect the language, any language), or one language such as "en", "hi", "ta" (more reliable on short audio)
    voice_stt: str  # LiveKit Inference model ids, billed to your LiveKit Cloud plan
    voice_llm: str  # LiveKit Inference model, used only when there is no Groq key
    groq_api_key: str  # when set, the voice conversation model runs on Groq
    voice_groq_model: str
    voice_groq_stt: str  # Groq speech-to-text model for the live voice agent
    voice_tts: str
    voice_tts_voice: str
    max_parallel: int  # tickets processed at the same time by the mailbox poller and the Telegram bot
    fast_model: str  # cheap model for triage, query analysis and the mechanical tool-calling agents (DeepSeek by default)
    fast_base_url: str
    fast_api_key: str  # empty = no fast model, everything uses the main model
    embeddings_mode: str  # auto | on | off
    vector_backend: str  # faiss | numpy
    team_mode: str  # roundrobin | selector
    tool_choice: str  # required | auto
    require_citations: bool
    guard_model: str  # empty = Llama Guard screen off
    guard_base_url: str
    guard_api_key: str
    guard_timeout: int
    max_complaint_chars: int
    rate_limit_per_min: int
    confidence_threshold: float
    kb_min_score: float
    llm_timeout: int
    run_timeout: int
    redact_pii: bool
    kb_dir: Path
    db_path: Path
    cache_dir: Path
    session_timeout_min: int
    max_failed_logins: int
    lockout_minutes: int
    reset_token_minutes: int
    reset_requests_per_hour: int
    app_base_url: str
    email_backend: str  # smtp | outbox | off
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    smtp_starttls: bool
    outbox_dir: Path
    mail_backend: str  # imap | folder | off
    imap_host: str
    imap_port: int
    imap_user: str
    imap_password: str
    imap_folder: str
    inbox_dir: Path
    mail_poll_seconds: int
    mail_auto: bool  # poll the mailbox in the background while the web app runs
    mail_max_per_poll: int
    stt_model: str  # speech-to-text model (OpenAI-compatible /audio/transcriptions)
    stt_base_url: str
    stt_api_key: str
    vision_model: str  # image-understanding chat model; empty = use LLM_MODEL
    vision_base_url: str
    vision_api_key: str
    media_enabled: bool
    max_audio_mb: int
    max_image_mb: int
    max_images: int
    telegram_token: str  # from @BotFather
    mail_triage: bool  # classify each email first: only support requests enter the agent pipeline

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def livekit_enabled(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def fast_enabled(self) -> bool:
        return bool(self.fast_api_key and self.fast_model)

    @property
    def embeddings_enabled(self) -> bool:
        if self.embeddings_mode == "off" or not self.has_key:
            return False
        if self.embeddings_mode == "on":
            return True
        # auto: only the default OpenAI endpoint is known to serve embeddings, unless the
        # user explicitly pointed EMBEDDING_MODEL at a compatible provider.
        return not self.base_url or bool(os.getenv("EMBEDDING_MODEL"))


def _default_kb_dir() -> Path:
    """Markdown KB if present, otherwise the PDF KB (so a fresh clone works without any .env setting)."""
    md, pdf = ROOT / "knowledge_base", ROOT / "knowledge_base_pdf"
    return md if md.exists() or not pdf.exists() else pdf


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip(),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        livekit_url=os.getenv("LIVEKIT_URL", "").strip(),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
        voice_agent_name=os.getenv("VOICE_AGENT_NAME", "novacart-support").strip(),
        voice_stt_language=os.getenv("VOICE_STT_LANGUAGE", "multi").strip(),
        voice_stt=os.getenv("VOICE_STT","assemblyai/universal-3-6-pro").strip(),
        voice_llm=os.getenv("VOICE_LLM", "google/gemma-4-31b-it").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        voice_groq_model=os.getenv("VOICE_GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        voice_groq_stt=os.getenv("VOICE_GROQ_STT", "whisper-large-v3-turbo").strip(),
        voice_tts=os.getenv("VOICE_TTS", "fishaudio/s2.1-pro").strip(),
        voice_tts_voice=os.getenv("VOICE_TTS_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a").strip(),
        max_parallel=max(1, _int("MAX_PARALLEL", 3)),
        fast_model=os.getenv("FAST_MODEL", "deepseek-chat").strip(),
        fast_base_url=os.getenv("FAST_BASE_URL", "https://api.deepseek.com").strip(),
        fast_api_key=(os.getenv("FAST_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")).strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embeddings_mode=os.getenv("EMBEDDINGS", "auto").strip().lower(),
        vector_backend=os.getenv("VECTOR_BACKEND", "faiss").strip().lower(),
        team_mode=os.getenv("TEAM_MODE", "roundrobin").strip().lower(),
        tool_choice=os.getenv("TOOL_CHOICE", "required").strip().lower(),
        require_citations=os.getenv("REQUIRE_CITATIONS", "true").strip().lower() != "false",
        guard_model=os.getenv("GUARD_MODEL", "").strip(),
        guard_base_url=os.getenv("GUARD_BASE_URL", "").strip(),
        guard_api_key=os.getenv("GUARD_API_KEY", "").strip(),
        guard_timeout=_int("GUARD_TIMEOUT", 15),
        max_complaint_chars=_int("MAX_COMPLAINT_CHARS", 2000),
        rate_limit_per_min=_int("RATE_LIMIT_PER_MIN", 10),
        confidence_threshold=_float("CONFIDENCE_THRESHOLD", 0.6),
        kb_min_score=_float("KB_MIN_SCORE", 0.08),
        llm_timeout=_int("LLM_TIMEOUT", 60),
        run_timeout=_int("RUN_TIMEOUT", 150),
        redact_pii=os.getenv("LOG_REDACT", "true").strip().lower() != "false",
        kb_dir=Path(os.getenv("KB_DIR") or _default_kb_dir()),
        db_path=Path(os.getenv("DB_PATH", ROOT / "data" / "support.db")),
        cache_dir=Path(os.getenv("CACHE_DIR", ROOT / ".cache")),
        session_timeout_min=_int("SESSION_TIMEOUT_MIN", 30),
        max_failed_logins=_int("MAX_FAILED_LOGINS", 5),
        lockout_minutes=_int("LOCKOUT_MINUTES", 15),
        reset_token_minutes=_int("RESET_TOKEN_MINUTES", 30),
        reset_requests_per_hour=_int("RESET_REQUESTS_PER_HOUR", 3),
        app_base_url=os.getenv("APP_BASE_URL", "http://localhost:8501").strip().rstrip("/"),
        email_backend=os.getenv("EMAIL_BACKEND", "smtp" if os.getenv("SMTP_HOST") else "outbox").strip().lower(),
        smtp_host=os.getenv("SMTP_HOST", "").strip(),
        smtp_port=_int("SMTP_PORT", 587),
        smtp_user=os.getenv("SMTP_USER", "").strip(),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_from=os.getenv("SMTP_FROM", "NovaCart Support <no-reply@novacart.example>").strip(),
        smtp_starttls=os.getenv("SMTP_STARTTLS", "true").strip().lower() != "false",
        outbox_dir=Path(os.getenv("OUTBOX_DIR", ROOT / "data" / "outbox")),
        mail_backend=os.getenv("MAIL_BACKEND", "imap" if os.getenv("IMAP_HOST") else "folder").strip().lower(),
        imap_host=os.getenv("IMAP_HOST", "").strip(),
        imap_port=_int("IMAP_PORT", 993),
        imap_user=os.getenv("IMAP_USER", "").strip(),
        imap_password=os.getenv("IMAP_PASSWORD", ""),
        imap_folder=os.getenv("IMAP_FOLDER", "INBOX").strip(),
        inbox_dir=Path(os.getenv("INBOX_DIR", ROOT / "data" / "inbox")),
        mail_poll_seconds=_int("MAIL_POLL_SECONDS", 60),
        mail_auto=os.getenv("MAIL_AUTO", "true").strip().lower() != "false",
        mail_max_per_poll=_int("MAIL_MAX_PER_POLL", 10),
        stt_model=(os.getenv("STT_MODEL") or ("whisper-large-v3-turbo" if os.getenv("GROQ_API_KEY") else "whisper-1")).strip(),
        stt_base_url=(os.getenv("STT_BASE_URL") or ("https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else os.getenv("LLM_BASE_URL", ""))).strip(),
        stt_api_key=(os.getenv("STT_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip(),
        vision_model=(os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")).strip(),
        vision_base_url=(os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL", "")).strip(),
        vision_api_key=(os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip(),
        media_enabled=os.getenv("MEDIA_INPUT", "true").strip().lower() != "false",
        max_audio_mb=_int("MAX_AUDIO_MB", 10),
        max_image_mb=_int("MAX_IMAGE_MB", 5),
        max_images=_int("MAX_IMAGES", 3),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        mail_triage=os.getenv("MAIL_TRIAGE", "true").strip().lower() != "false",
    )


def _client(model: str, api_key: str, base_url: str, timeout: int):
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    kwargs: dict = {"model": model, "api_key": api_key, "temperature": 0, "timeout": timeout, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
        # Non-OpenAI endpoints are unknown to AutoGen's model table, so declare capabilities.
        kwargs["model_info"] = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        }
    return OpenAIChatCompletionClient(**kwargs)


def build_model_client(settings: Settings | None = None):
    """Return a NEW chat-completion client.

    Deliberately not cached: the async HTTP pool inside the client is bound to the event loop it was first used in,
    and the Streamlit app runs every ticket in its own asyncio.run() loop. A client reused across loops fails with
    "RuntimeError: Event loop is closed". Callers own the client and must close it (team.run_ticket does).
    """
    s = settings or get_settings()
    if not s.has_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = _client(s.model, s.api_key, s.base_url, s.llm_timeout)
    if s.tool_choice == "required":
        from llm_client import RequiredToolClient

        return RequiredToolClient(client)
    return client


def build_fast_client(settings: Settings | None = None, fallback=None):
    """Client for the cheap, quick steps (triage, query analysis, the two tool-calling agents). Without a fast key it is just the main
    model. With one, a call that fails on the fast model is repeated on the main model (`fallback`, owned by the caller, or a new
    main client this one then owns). Same ownership rule as build_model_client: a NEW client the caller must close."""
    from llm_client import FallbackClient, RequiredToolClient

    s = settings or get_settings()
    if not s.fast_enabled:
        return fallback or build_model_client(s)
    primary = _client(s.fast_model, s.fast_api_key, s.fast_base_url, s.llm_timeout)
    if s.tool_choice == "required":
        primary = RequiredToolClient(primary)
    return FallbackClient(primary, fallback or build_model_client(s), own_fallback=fallback is None)
