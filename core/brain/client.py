"""Synchronous OpenRouter boundary; call only from a worker thread.

The animation tick must never invoke this module: every method performs
blocking network I/O unless an offline transport is injected.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar
from urllib.request import Request, urlopen

from .prompts import FixedPromptBuilder, PromptBuilder
from .schema import (
    ObservationResponse,
    PlanResponse,
    ResponseSchemaError,
    parse_observation_response,
    parse_plan_response,
)

LOGGER = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RESPONSE_BYTES = 8 * 1024
REPAIR_INSTRUCTION = "Your prior response was invalid. Return corrected bare JSON only."
PRIVATE_PROVIDER = {"zdr": True, "data_collection": "deny", "allow_fallbacks": False}
CallType = Literal["converse", "observe", "narrate"]
T = TypeVar("T", PlanResponse, ObservationResponse)


@dataclass(frozen=True, slots=True)
class RecentExchange:
    human_say: str
    lamp_say: str

    def __post_init__(self) -> None:
        for name, value in (("human_say", self.human_say), ("lamp_say", self.lamp_say)):
            if not isinstance(value, str) or "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be single-line text")


@dataclass(frozen=True, slots=True)
class TransportResponse:
    content: str
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True, slots=True)
class CallMetrics:
    call_type: CallType
    attempt: int
    repaired: bool
    latency_ms: float
    model: str
    profile: str
    tokens_in: int
    tokens_out: int
    outcome: Literal["valid", "invalid", "transport_error"]


class ResponseTooLarge(RuntimeError):
    """The model's streamed content crossed the 8 KiB circuit breaker."""

    def __init__(self, partial_content: str) -> None:
        super().__init__("model response exceeded the transport guard")
        self.partial_content = partial_content


class OpenRouterTransportError(RuntimeError):
    """The endpoint returned an unusable streaming response."""


class BrainWarmError(RuntimeError):
    """Warm-up exhausted its repair attempt without a validated response."""


class CompletionTransport(Protocol):
    def complete(
        self,
        request: Mapping[str, object],
        api_key: str,
        max_response_bytes: int,
    ) -> TransportResponse: ...


class BrainClient(Protocol):
    def warm(self) -> None: ...

    def converse(
        self,
        transcript: str,
        compact_memory: str,
        recent: Sequence[RecentExchange],
    ) -> PlanResponse: ...

    def observe(
        self,
        jpeg: bytes,
        prior_canonical: Sequence[str],
    ) -> ObservationResponse: ...

    def narrate(self, missing: Sequence[str]) -> PlanResponse: ...


class StdlibOpenRouterTransport:
    """Stream OpenRouter SSE with no dependency beyond the standard library."""

    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s

    def complete(
        self,
        request: Mapping[str, object],
        api_key: str,
        max_response_bytes: int,
    ) -> TransportResponse:
        body = json.dumps(request, separators=(",", ":")).encode("utf-8")
        http_request = Request(
            OPENROUTER_URL,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        parts: list[str] = []
        byte_count = 0
        model: str | None = None
        tokens_in = tokens_out = 0
        with urlopen(http_request, timeout=self.timeout_s) as response:
            for raw_line in response:
                if not raw_line.startswith(b"data:"):
                    continue
                data = raw_line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise OpenRouterTransportError("invalid SSE event") from error
                if not isinstance(event, dict) or "error" in event:
                    raise OpenRouterTransportError("OpenRouter returned an error event")
                model = event.get("model") if isinstance(event.get("model"), str) else model
                usage = event.get("usage")
                if isinstance(usage, dict):
                    tokens_in = _token_count(usage.get("prompt_tokens"))
                    tokens_out = _token_count(usage.get("completion_tokens"))
                for choice in event.get("choices", ()):
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    fragment = delta.get("content") if isinstance(delta, dict) else None
                    if fragment is None:
                        continue
                    if not isinstance(fragment, str):
                        raise OpenRouterTransportError("non-text response content")
                    byte_count += len(fragment.encode("utf-8"))
                    parts.append(fragment)
                    if byte_count > max_response_bytes:
                        raise ResponseTooLarge("".join(parts))
        return TransportResponse("".join(parts), model, tokens_in, tokens_out)


class OpenRouterBrainClient:
    """Validated implementation of the three and only three model call types."""

    def __init__(
        self,
        *,
        profile: Literal["free", "private"] = "free",
        model: str | None = None,
        api_key: str | None = None,
        transport: CompletionTransport | None = None,
        prompts: PromptBuilder | None = None,
        metrics_callback: Callable[[CallMetrics], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if profile == "free":
            self.model = model or "openrouter/free"
            self.provider: dict[str, object] = {}
        elif profile == "private":
            if not isinstance(model, str) or not model.strip():
                raise ValueError("private profile requires an explicit model")
            self.model = model
            self.provider = dict(PRIVATE_PROVIDER)
        else:
            raise ValueError("profile must be 'free' or 'private'")
        secret = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError("OPENROUTER_API_KEY is required")
        self.profile = profile
        self._api_key = secret
        self._transport = transport or StdlibOpenRouterTransport()
        self._prompts = prompts or FixedPromptBuilder()
        self._metrics_callback = metrics_callback
        self._clock = clock
        self._warm_lock = threading.Lock()
        self._warmed = False

    def warm(self) -> None:
        """Warm once through the existing narrate call from a startup worker."""

        with self._warm_lock:
            if self._warmed:
                return
            self._narrate((), strict=True)
            self._warmed = True

    def converse(
        self,
        transcript: str,
        compact_memory: str,
        recent: Sequence[RecentExchange],
    ) -> PlanResponse:
        payload = self._prompts.converse_payload(transcript, compact_memory, recent)
        return self._run("converse", self._text_messages(payload), parse_plan_response)

    def observe(self, jpeg: bytes, prior_canonical: Sequence[str]) -> ObservationResponse:
        payload = self._prompts.observe_payload(jpeg, prior_canonical)
        labels = {"prior_canonical": payload["prior_canonical"]}
        content = [
            {"type": "text", "text": _compact_json(labels)},
            {"type": "image_url", "image_url": {"url": payload["jpeg_data_url"]}},
        ]
        return self._run("observe", self._messages(content), parse_observation_response)

    def narrate(self, missing: Sequence[str]) -> PlanResponse:
        return self._narrate(missing, strict=False)

    def _narrate(self, missing: Sequence[str], *, strict: bool) -> PlanResponse:
        payload = self._prompts.narrate_payload(missing)
        return self._run(
            "narrate", self._text_messages(payload), parse_plan_response, strict=strict
        )

    def _text_messages(self, payload: Mapping[str, object]) -> list[dict[str, object]]:
        return self._messages(_compact_json(payload))

    def _messages(self, user_content: object) -> list[dict[str, object]]:
        return [
            {"role": "system", "content": self._prompts.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _run(
        self,
        call_type: CallType,
        messages: list[dict[str, object]],
        parser: Callable[[str], T],
        *,
        strict: bool = False,
    ) -> T:
        for attempt in range(2):
            repair = [{"role": "user", "content": REPAIR_INSTRUCTION}] if attempt else []
            request = self._request(messages + repair)
            started = self._clock()
            try:
                response = self._transport.complete(request, self._api_key, MAX_RESPONSE_BYTES)
            except ResponseTooLarge as error:
                latency_ms = (self._clock() - started) * 1000.0
                self._log_invalid(call_type, error.partial_content, "response exceeded 8 KiB")
                self._emit_metrics(call_type, attempt, latency_ms, None, "invalid")
            except Exception as error:
                latency_ms = (self._clock() - started) * 1000.0
                LOGGER.warning("%s transport failed (%s)", call_type, type(error).__name__)
                self._emit_metrics(call_type, attempt, latency_ms, None, "transport_error")
            else:
                latency_ms = (self._clock() - started) * 1000.0
                try:
                    parsed = parser(response.content)
                except (ResponseSchemaError, TypeError, ValueError) as error:
                    self._log_invalid(call_type, response.content, str(error))
                    self._emit_metrics(call_type, attempt, latency_ms, response, "invalid")
                else:
                    self._emit_metrics(call_type, attempt, latency_ms, response, "valid")
                    return parsed
        if strict:
            raise BrainWarmError("OpenRouter warm-up failed after its repair attempt")
        if call_type == "observe":
            return ObservationResponse((), ())  # type: ignore[return-value]
        return PlanResponse("Oops—my thoughts got tangled! Can we try that again?", ())  # type: ignore[return-value]

    def _request(self, messages: list[dict[str, object]]) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": messages,
            "provider": dict(self.provider),
            "reasoning": {"effort": "none"},
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def _log_invalid(self, call_type: CallType, raw: str, reason: str) -> None:
        safe_raw = raw.replace(self._api_key, "[REDACTED]")
        LOGGER.warning("invalid %s response (%s); raw_response=%r", call_type, reason, safe_raw)

    def _emit_metrics(
        self,
        call_type: CallType,
        attempt: int,
        latency_ms: float,
        response: TransportResponse | None,
        outcome: Literal["valid", "invalid", "transport_error"],
    ) -> None:
        if self._metrics_callback is None:
            return
        metric = CallMetrics(
            call_type=call_type,
            attempt=attempt + 1,
            repaired=bool(attempt),
            latency_ms=latency_ms,
            model=(response.model or self.model) if response else self.model,
            profile=self.profile,
            tokens_in=response.tokens_in if response else 0,
            tokens_out=response.tokens_out if response else 0,
            outcome=outcome,
        )
        try:
            self._metrics_callback(metric)
        except Exception as error:
            LOGGER.warning("brain metrics callback failed (%s)", type(error).__name__)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _token_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = [
    "BrainClient",
    "BrainWarmError",
    "CallMetrics",
    "CompletionTransport",
    "MAX_RESPONSE_BYTES",
    "OPENROUTER_URL",
    "OpenRouterBrainClient",
    "PRIVATE_PROVIDER",
    "RecentExchange",
    "ResponseTooLarge",
    "StdlibOpenRouterTransport",
    "TransportResponse",
]
