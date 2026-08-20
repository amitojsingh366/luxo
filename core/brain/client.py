"""Synchronous OpenRouter boundary; call only from a worker thread.

The animation tick must never invoke this module: every method performs
blocking network I/O unless an offline transport is injected.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import FixedPromptBuilder, PromptBuilder
from .schema import (
    ActionOp,
    ObservationResponse,
    ObservedObject,
    PlanResponse,
    ResponseSchemaError,
    extract_json_object,
    parse_observation_response,
    parse_plan_response,
)

LOGGER = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RESPONSE_BYTES = 8 * 1024
PRIVATE_PROVIDER = {"zdr": True, "data_collection": "deny", "allow_fallbacks": False}
CallType = Literal["converse", "observe", "narrate", "scene_comment"]


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    """Validated facts plus quarantined dialogue from the same JPEG request.

    PRD §8.1.1 normally permits facts only from ``observe``. The owner-approved
    exception preserves a bounded accidental ``say`` only across that call's
    repair attempt. It is never part of the fact schema and carries no plan.
    """

    facts: ObservationResponse
    quarantined_say: str


ValidatedObservation = ObservationResponse | ObservationEnvelope
T = TypeVar("T", PlanResponse, ObservationResponse, ObservationEnvelope)

REPAIR_INSTRUCTIONS: Mapping[CallType, str] = {
    "converse": (
        "Your conversation response was invalid. Return one bare JSON object with exactly "
        "the top-level keys say and plan. Every plan item needs an op from gesture, "
        "look_at, light, sfx, scan, observe, posture, or wait. A gesture name such as "
        "perk_up belongs in name with op set to gesture; it is never itself an op. Do not "
        "copy the current transcript into say. For a current visual-detail question, say only "
        "a brief looking or checking acknowledgement and return exactly one observe action, "
        "with no scan. For an explicitly historical question, answer it from memory when the "
        "requested object fact is present; otherwise say that you do not know."
    ),
    "narrate": (
        "Your narration response was invalid. Return one bare JSON object with exactly "
        "the top-level keys say and plan. Every plan item needs an op from gesture, "
        "look_at, light, sfx, scan, observe, posture, or wait. A gesture name such as "
        "perk_up belongs in name with op set to gesture; it is never itself an op."
    ),
    "observe": (
        "Your observation response was invalid. Inspect the supplied image and return one "
        "bare JSON object with exactly the top-level keys present, known, and new. present must be "
        "an array containing only supplied prior_canonical labels that remain visible. new "
        "must describe every salient visible object not in prior_canonical using label, "
        "canonical, attributes, and bbox_norm. known must describe every present object with "
        "the same fields and current visible details. If prior_canonical is empty, present and "
        "known must be empty and visible objects must go in new. bbox_norm must contain "
        "decimal x, y, width, "
        "and height values between 0 and 1, not pixel or corner coordinates. Always include "
        "all three arrays. Do not return "
        "say, plan, dialogue, or prose."
    ),
    "scene_comment": (
        "Your scene comment was invalid. Return one bare JSON object with exactly say and "
        "plan. Ground say in a named fresh object. For spontaneous_hand, ask one concise, "
        "specific question about that object's purpose, feature, or use. Never ask what the "
        "person is working on, doing, holding, or what is in their hand. Do not observe again."
    ),
}


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

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class BrainWarmError(RuntimeError):
    """Warm-up exhausted its repair attempt without a validated response."""


class ObservationUnavailableError(RuntimeError):
    """An observation exhausted repair without producing validated facts."""


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
    ) -> ValidatedObservation: ...

    def narrate(self, missing: Sequence[str]) -> PlanResponse: ...

    def scene_comment(
        self,
        visual_intent: str,
        observation: ValidatedObservation,
        compact_memory: str,
    ) -> PlanResponse: ...


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
        try:
            response_context = urlopen(http_request, timeout=self.timeout_s)
        except HTTPError as error:
            retry_after = _retry_after_seconds(
                error.headers.get("Retry-After") if error.headers is not None else None
            )
            detail = f"OpenRouter returned HTTP {error.code}"
            payload = _read_error_payload(error)
            explanation = _openrouter_error_explanation(payload, api_key)
            if explanation:
                detail += f": {explanation}"
            if retry_after is not None:
                detail += f"; retry after {retry_after:g}s"
            raise OpenRouterTransportError(
                detail,
                status_code=error.code,
                retry_after_s=retry_after,
            ) from error
        except URLError as error:
            raise OpenRouterTransportError("OpenRouter connection failed") from error
        with response_context as response:
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
                if not isinstance(event, dict):
                    raise OpenRouterTransportError("OpenRouter returned an invalid SSE event")
                if "error" in event:
                    error_payload = event.get("error")
                    raw_code = error_payload.get("code") if isinstance(error_payload, dict) else None
                    status_code = raw_code if isinstance(raw_code, int) else None
                    detail = "OpenRouter returned a streaming error"
                    if status_code is not None:
                        detail += f" {status_code}"
                    explanation = _openrouter_error_explanation(event, api_key)
                    if explanation:
                        detail += f": {explanation}"
                    raise OpenRouterTransportError(detail, status_code=status_code)
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
    """Validated implementation of the four narrow model call types."""

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
        LOGGER.info("OpenRouter configured model=%s profile=%s", self.model, self.profile)

    def warm(self) -> None:
        """Warm once through the existing narrate call from a startup worker."""

        with self._warm_lock:
            if self._warmed:
                return
            if self.profile == "free":
                # OpenRouter is remote and a free request is quota, not a local
                # model warm-up. Readiness is exercised by the first real turn.
                self._warmed = True
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

        def parse_conversation(raw: str) -> PlanResponse:
            response = parse_plan_response(raw)
            if response.say.strip().casefold() == transcript.strip().casefold():
                raise ResponseSchemaError("conversation response echoed current transcript")
            if _requires_fresh_visual(transcript):
                if tuple(action.op.value for action in response.plan) != ("observe",):
                    raise ResponseSchemaError(
                        "current visual question requires exactly one observe action"
                    )
                acknowledgement = response.say.casefold()
                if not any(
                    phrase in acknowledgement
                    for phrase in ("look", "check", "let me see", "take a peek", "one moment")
                ):
                    raise ResponseSchemaError(
                        "current visual question requires a looking acknowledgement"
                    )
            return response

        return self._run("converse", self._text_messages(payload), parse_conversation)

    def observe(self, jpeg: bytes, prior_canonical: Sequence[str]) -> ValidatedObservation:
        payload = self._prompts.observe_payload(jpeg, prior_canonical)
        prior = payload["prior_canonical"]
        labels = {"prior_canonical": prior}
        content = [
            {"type": "text", "text": _compact_json(labels)},
            {"type": "image_url", "image_url": {"url": payload["jpeg_data_url"]}},
        ]

        quarantined_say: str | None = None

        def parse_observation(raw: str) -> ValidatedObservation:
            nonlocal quarantined_say
            if quarantined_say is None:
                quarantined_say = _dialogue_candidate(raw)
            response = parse_observation_response(raw, require_known=True)
            prior_set = frozenset(prior)
            if any(label not in prior_set for label in response.present):
                raise ResponseSchemaError(
                    "observation present contains a label outside prior_canonical"
                )
            if any(item.canonical not in prior_set for item in response.known):
                raise ResponseSchemaError(
                    "observation known contains an object outside prior_canonical"
                )
            if not {item.canonical for item in response.known} <= set(response.present):
                raise ResponseSchemaError(
                    "observation known must describe only present objects"
                )
            if any(item.canonical in prior_set for item in response.new):
                raise ResponseSchemaError(
                    "observation new repeats an object from prior_canonical"
                )
            if quarantined_say is not None and _grounded_scene_say(
                quarantined_say,
                response,
                allow_uncertainty=False,
            ):
                return ObservationEnvelope(response, quarantined_say)
            return response

        return self._run("observe", self._messages(content), parse_observation)

    def narrate(self, missing: Sequence[str]) -> PlanResponse:
        return self._narrate(missing, strict=False)

    def scene_comment(
        self,
        visual_intent: str,
        observation: ValidatedObservation,
        compact_memory: str,
    ) -> PlanResponse:
        facts = _observation_facts(observation)
        payload = self._prompts.scene_comment_payload(
            visual_intent,
            facts,
            compact_memory,
        )
        fresh_objects = payload["fresh_objects"]
        assert isinstance(fresh_objects, list)
        object_terms = tuple(
            str(item[key]).casefold()
            for item in fresh_objects
            if isinstance(item, Mapping)
            for key in ("canonical", "label")
            if isinstance(item.get(key), str)
        )
        attribute_terms = tuple(
            str(attribute).casefold()
            for item in fresh_objects
            if isinstance(item, Mapping) and isinstance(item.get("attributes"), list)
            for attribute in item["attributes"]
            if isinstance(attribute, str)
        )
        spontaneous = visual_intent == "spontaneous_hand"

        candidate = (
            observation.quarantined_say
            if isinstance(observation, ObservationEnvelope)
            else None
        )
        if candidate is not None and _grounded_scene_say(
            candidate, facts, allow_uncertainty=False
        ) and _candidate_matches_visual_intent(candidate, facts, visual_intent):
            # Perception dialogue is never trusted on its own. At this point a
            # separate repair has produced validated facts, memory has been
            # committed, and the candidate names one of those fresh objects.
            # Its model-authored plan is deliberately unavailable here.
            return PlanResponse(candidate, ())

        def parse_comment(raw: str) -> PlanResponse:
            response = parse_plan_response(raw)
            normalized = " ".join(response.say.casefold().replace("’", "'").split())
            if any(phrase in normalized for phrase in _GENERIC_SCENE_PHRASES):
                raise ResponseSchemaError("scene comment used a forbidden generic question")
            uncertainty = ("not sure", "can't tell", "cannot tell", "don't know")
            grounding_terms = object_terms if spontaneous else object_terms + attribute_terms
            grounded = any(term in normalized for term in grounding_terms)
            if grounding_terms and not grounded and not any(
                phrase in normalized for phrase in uncertainty
            ):
                raise ResponseSchemaError("scene comment is not grounded in a fresh object")
            if len(response.plan) > 3:
                raise ResponseSchemaError("scene comment plan must stay small")
            # The line is already grounded in the fresh, committed frame. A
            # lightweight model often appends another scan/observe out of habit;
            # those actions are redundant here and would create a visual loop.
            # Keep the useful line and locally remove only those forbidden ops
            # instead of paying a repair call that may rewrite it.
            safe_plan = tuple(
                action
                for action in response.plan
                if action.op not in (ActionOp.SCAN, ActionOp.OBSERVE)
            )
            return PlanResponse(response.say, safe_plan)

        return self._run("scene_comment", self._text_messages(payload), parse_comment)

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
        structured_output = True
        rate_limit: OpenRouterTransportError | None = None
        for attempt in range(2):
            repair = (
                [{"role": "user", "content": REPAIR_INSTRUCTIONS[call_type]}]
                if attempt
                else []
            )
            request = self._request(messages + repair, structured_output=structured_output)
            started = self._clock()
            try:
                response = self._transport.complete(request, self._api_key, MAX_RESPONSE_BYTES)
            except ResponseTooLarge as error:
                latency_ms = (self._clock() - started) * 1000.0
                self._log_invalid(call_type, error.partial_content, "response exceeded 8 KiB")
                self._emit_metrics(call_type, attempt, latency_ms, None, "invalid")
            except Exception as error:
                latency_ms = (self._clock() - started) * 1000.0
                if isinstance(error, OpenRouterTransportError):
                    LOGGER.warning("%s transport failed (%s)", call_type, error)
                else:
                    LOGGER.warning("%s transport failed (%s)", call_type, type(error).__name__)
                self._emit_metrics(call_type, attempt, latency_ms, None, "transport_error")
                if isinstance(error, OpenRouterTransportError):
                    if error.status_code == 429:
                        rate_limit = error
                        break
                    # A pinned endpoint may accept JSON text while rejecting
                    # response_format. The fixed prompt and parser still own
                    # the schema, so retry once without that optional hint.
                    structured_output = False
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
            raise ObservationUnavailableError(
                "OpenRouter observation failed after its repair attempt"
            )
        if rate_limit is not None:
            if call_type == "narrate":
                return PlanResponse("", ())  # type: ignore[return-value]
            wait = rate_limit.retry_after_s
            if wait is not None:
                seconds = max(1, int(math.ceil(wait)))
                say = f"OpenRouter is rate-limiting me—please try again in {seconds} seconds."
            else:
                say = "OpenRouter is rate-limiting me—please try again later."
            return PlanResponse(say, ())  # type: ignore[return-value]
        return PlanResponse("Oops—my thoughts got tangled! Can we try that again?", ())  # type: ignore[return-value]

    def _request(
        self,
        messages: list[dict[str, object]],
        *,
        structured_output: bool = True,
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "provider": dict(self.provider),
            "reasoning": {"effort": "none"},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if structured_output:
            request["response_format"] = {"type": "json_object"}
        return request

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


_GENERIC_SCENE_PHRASES = (
    "what are you working on",
    "what are you doing",
    "what are you holding",
    "what is in your hand",
    "what's in your hand",
    "anything in your hand",
)
_HUMAN_CANONICALS = frozenset(("person", "human", "man", "woman", "boy", "girl"))
_COLOR_WORDS = frozenset(
    (
        "black",
        "blue",
        "brown",
        "cyan",
        "gold",
        "gray",
        "green",
        "grey",
        "magenta",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "white",
        "yellow",
    )
)


def _dialogue_candidate(raw: str) -> str | None:
    """Extract only bounded dialogue from a dialogue-shaped observe failure."""

    try:
        root = extract_json_object(raw)
        if "say" not in root or "plan" not in root:
            return None
        return parse_plan_response(root).say or None
    except (ResponseSchemaError, TypeError, ValueError):
        return None


def _grounded_scene_say(
    say: str,
    observation: ObservationResponse,
    *,
    allow_uncertainty: bool,
) -> bool:
    """Approve a scene line only when fresh validated facts support it."""

    normalized = " ".join(say.casefold().replace("’", "'").split())
    if not normalized or any(
        phrase in normalized for phrase in _GENERIC_SCENE_PHRASES
    ):
        return False
    objects = (*observation.known, *observation.new)
    object_terms = tuple(
        term
        for item in objects
        for term in (item.canonical.casefold(), item.label.casefold())
        if term
    )
    if any(term in normalized for term in object_terms):
        return True
    return allow_uncertainty and any(
        phrase in normalized
        for phrase in ("not sure", "can't tell", "cannot tell", "don't know")
    )


def _candidate_matches_visual_intent(
    say: str,
    observation: ObservationResponse,
    visual_intent: str,
) -> bool:
    """Require the quarantined line to identify the intent's supported object."""

    objects = (*observation.known, *observation.new)
    if _is_hand_intent(visual_intent):
        held = tuple(item for item in objects if _is_explicitly_held(item.attributes))
        if held:
            focused = held
        else:
            non_person = tuple(
                item for item in objects if item.canonical not in _HUMAN_CANONICALS
            )
            if len(non_person) != 1:
                return False
            focused = non_person
    else:
        focused = objects

    normalized = " ".join(say.casefold().replace("’", "'").split())
    named = tuple(
        item
        for item in focused
        if any(
            term and term in normalized
            for term in (item.canonical.casefold(), item.label.casefold())
        )
    )
    if len(named) != 1:
        return False
    return not _contradicts_visible_colors(say, named[0])


def _is_hand_intent(visual_intent: str) -> bool:
    normalized = " ".join(visual_intent.casefold().replace("_", " ").split())
    return normalized == "spontaneous hand" or "holding" in normalized or "hand" in normalized


def _is_explicitly_held(attributes: Sequence[str]) -> bool:
    for attribute in attributes:
        normalized = " ".join(attribute.casefold().replace("_", " ").split())
        if "held" in normalized and "hand" in normalized:
            return True
        if normalized in ("held", "holding", "in hand", "in hands"):
            return True
    return False


def _contradicts_visible_colors(say: str, item: ObservedObject) -> bool:
    claimed = _colors_in(say)
    if not claimed:
        return False
    supported = _colors_in(" ".join((item.label, *item.attributes)))
    return not claimed <= supported


def _colors_in(value: str) -> frozenset[str]:
    colors = {word for word in re.findall(r"[a-z]+", value.casefold()) if word in _COLOR_WORDS}
    if "grey" in colors:
        colors.remove("grey")
        colors.add("gray")
    return frozenset(colors)


def _observation_facts(observation: ValidatedObservation) -> ObservationResponse:
    if isinstance(observation, ObservationEnvelope):
        return observation.facts
    if isinstance(observation, ObservationResponse):
        return observation
    raise TypeError("observation must contain validated facts")


def _requires_fresh_visual(transcript: str) -> bool:
    """Identify present visual questions without capturing historical recall."""

    normalized = " ".join(transcript.casefold().replace("’", "'").split())
    historical = (
        "remember",
        "yesterday",
        "earlier",
        "before",
        "last time",
        "did you see",
        "did it look",
        "what color was",
        "what colour was",
    )
    if any(phrase in normalized for phrase in historical):
        return False
    current_scene = (
        "look at this",
        "look at that",
        "take a look",
        "what do you see",
        "what can you see",
        "what is in my hand",
        "what's in my hand",
        "what am i holding",
        "what is this",
        "what's this",
        "what is that",
        "what's that",
    )
    if any(phrase in normalized for phrase in current_scene):
        return True
    visual_detail = (
        "what color",
        "what colour",
        "which color",
        "which colour",
        "what material",
        "what is it made of",
        "what does it look like",
    )
    return any(phrase in normalized for phrase in visual_detail)


def _retry_after_seconds(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0.0 else None


def _read_error_payload(error: HTTPError) -> object:
    """Read only a bounded OpenRouter error envelope, never a success body."""

    try:
        raw = error.read(MAX_RESPONSE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def _openrouter_error_explanation(payload: object, api_key: str) -> str:
    """Return bounded allow-listed diagnostics from an OpenRouter error."""

    if not isinstance(payload, dict):
        return ""
    error_payload = payload.get("error")
    if not isinstance(error_payload, dict):
        return ""

    fragments: list[str] = []
    message = _safe_error_text(error_payload.get("message"), api_key)
    if message:
        fragments.append(message)

    metadata = error_payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("error_type", "provider_code", "provider_name", "model_slug"):
            value = _safe_error_text(metadata.get(key), api_key)
            if value:
                fragments.append(f"{key}={value}")
    return "; ".join(fragments)


def _safe_error_text(value: object, api_key: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return ""
    text = " ".join(str(value).split()).replace(api_key, "[REDACTED]")
    return text[:240]


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
    "OpenRouterTransportError",
    "ObservationUnavailableError",
    "PRIVATE_PROVIDER",
    "RecentExchange",
    "ResponseTooLarge",
    "StdlibOpenRouterTransport",
    "TransportResponse",
]
