"""Browser access to the voice agent: a short-lived LiveKit token that lets one signed-in visitor join one new room, and asks LiveKit to
send the NovaCart voice agent (voice_agent.py) into that room. The API secret stays on the server; the page only receives the token."""
from __future__ import annotations

import re
import secrets
from datetime import timedelta

from config import Settings

TOKEN_MINUTES = 10


def make_token(s: Settings, username: str, display_name: str = "") -> tuple[str, str]:
    """Return (token, room). Raises RuntimeError when LiveKit is not configured or the SDK is not installed."""
    if not s.livekit_enabled:
        raise RuntimeError("LiveKit is not configured")
    try:
        from livekit import api
    except ImportError as exc:
        raise RuntimeError("pip install -r requirements-voice.txt") from exc
    room = f"call-{secrets.token_hex(6)}"
    identity = f"web-{re.sub(r'[^A-Za-z0-9_.-]', '', username)[:40]}-{secrets.token_hex(3)}"
    token = (
        api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
        .with_identity(identity)
        .with_name(re.sub(r"[^\w .'-]", "", display_name)[:60] or "Customer")
        .with_attributes({"username": username})  # server-set: the voice agent saves the ticket under this account
        .with_ttl(timedelta(minutes=TOKEN_MINUTES))
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True))
        .with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=s.voice_agent_name)]))
        .to_jwt()
    )
    return token, room
