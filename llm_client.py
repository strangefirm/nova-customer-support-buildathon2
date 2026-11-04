"""Model-client wrapper that forces tool use.

Every agent in the desk has exactly one tool it must call (submit_analysis, search_kb,
submit_resolution, log_ticket). With the default tool_choice="auto", real models sometimes answer in
plain chat text instead; the result never reaches the shared ticket context and a perfectly good
ticket is escalated as "incomplete pipeline". AutoGen's AssistantAgent does not expose tool_choice,
so this wrapper sets "required" whenever tools are offered. Everything else is delegated untouched.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Mapping, Sequence

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelCapabilities, ModelInfo, RequestUsage
from autogen_core.tools import Tool, ToolSchema

log = logging.getLogger(__name__)


class RequiredToolClient(ChatCompletionClient):
    def __init__(self, inner: ChatCompletionClient):
        self._inner = inner

    @staticmethod
    def _choice(tools, tool_choice):
        return "required" if tools and tool_choice == "auto" else tool_choice

    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = [],
                     tool_choice: Tool | str = "auto", json_output: bool | type | None = None,
                     extra_create_args: Mapping[str, Any] = {}, cancellation_token: CancellationToken | None = None) -> CreateResult:
        return await self._inner.create(
            messages, tools=tools, tool_choice=self._choice(tools, tool_choice), json_output=json_output,
            extra_create_args=extra_create_args, cancellation_token=cancellation_token,
        )

    def create_stream(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = [],
                      tool_choice: Tool | str = "auto", json_output: bool | type | None = None,
                      extra_create_args: Mapping[str, Any] = {}, cancellation_token: CancellationToken | None = None
                      ) -> AsyncGenerator[str | CreateResult, None]:
        return self._inner.create_stream(
            messages, tools=tools, tool_choice=self._choice(tools, tool_choice), json_output=json_output,
            extra_create_args=extra_create_args, cancellation_token=cancellation_token,
        )

    async def close(self) -> None:
        await self._inner.close()

    def actual_usage(self) -> RequestUsage:
        return self._inner.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._inner.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._inner.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._inner.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return self._inner.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._inner.model_info


class FallbackClient(ChatCompletionClient):
    """Use `primary`; if a call to it fails for any reason (outage, rate limit, bad key, timeout), repeat that one call on `fallback`.
    Cancellation is never swallowed. `own_fallback` says whether close() must also close the fallback client."""

    def __init__(self, primary: ChatCompletionClient, fallback: ChatCompletionClient, own_fallback: bool = False, label: str = "fast model"):
        self._primary, self._fallback, self._own_fallback, self._label = primary, fallback, own_fallback, label

    def _warn(self, exc: Exception) -> None:
        log.warning("%s failed (%s); retrying this call on the main model", self._label, type(exc).__name__)

    async def create(self, messages: Sequence[LLMMessage], **kwargs: Any) -> CreateResult:
        try:
            return await self._primary.create(messages, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any provider failure falls back
            self._warn(exc)
            return await self._fallback.create(messages, **kwargs)

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        started = False
        try:
            async for item in self._primary.create_stream(messages, **kwargs):
                started = True
                yield item
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if started:  # part of an answer was already delivered: do not restart it
                raise
            self._warn(exc)
        async for item in self._fallback.create_stream(messages, **kwargs):
            yield item

    async def close(self) -> None:
        await self._primary.close()
        if self._own_fallback:
            await self._fallback.close()

    def actual_usage(self) -> RequestUsage:
        return self._primary.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._primary.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._primary.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._primary.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return self._primary.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._primary.model_info
