"""Synchronous coordination for explicit, discrete scene observations.

This is the only core-side path from a captured camera frame to the vision
model. It is deliberately blocking and must therefore be called from a worker
thread, never from an animation or behaviour tick.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from .client import BrainClient, ObservationOrigin
from .memory import MAX_SCENE_OBJECTS, SceneMemoryStore, SceneObject
from .schema import (
    ObservationPrior,
    ObservationResponse,
    ObservedObject,
    is_durable_scene_canonical,
)

LOGGER = logging.getLogger(__name__)
_SCENE_FOCUS_CANONICALS = frozenset({"desk", "table", "tabletop", "workbench"})


class ObservationError(RuntimeError):
    """An observation cannot advance from its current state."""


class ObservationBusyError(ObservationError):
    """A second observation was requested while one is pending."""


class ObservationRequestError(ObservationError):
    """A completion or cancellation does not match the pending request."""


class ObservationImageError(ValueError):
    """A capture is not exactly one complete JPEG image."""


class ObservationMemoryIntent(str, Enum):
    """Local authority for the durable side effect of a fresh observation."""

    TRANSIENT = "transient"
    FOCUS = "focus"
    SCENE = "scene"


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    """The exact origin and stable identities bound to one capture."""

    request_id: str
    prior: tuple[ObservationPrior, ...]
    origin: ObservationOrigin
    memory_intent: ObservationMemoryIntent


@dataclass(frozen=True, slots=True)
class _MemoryCandidate:
    visible_index: int | None
    record: SceneObject
    assigns_identity: bool = False


@dataclass(slots=True)
class _PendingObservation:
    request: ObservationRequest
    response: ObservationResponse | None = None
    observed_at: float | None = None
    jpeg_digest: bytes | None = None


MemoryPublisher = Callable[[tuple[SceneObject, ...]], None]


class ObservationCoordinator:
    """Serialize capture/model/memory work behind one retryable request.

    The coordinator retains no image bytes. If durable memory storage fails
    after a valid model response, only the response, observation time, and a
    one-way digest remain so retrying storage cannot duplicate the model call.
    """

    def __init__(
        self,
        brain: BrainClient,
        store: SceneMemoryStore,
        *,
        clock: Callable[[], float] = time.time,
        publish: MemoryPublisher | None = None,
    ) -> None:
        self._brain = brain
        self._store = store
        self._clock = clock
        self._publish = publish
        self._lock = threading.RLock()
        self._pending: _PendingObservation | None = None

    @property
    def pending(self) -> ObservationRequest | None:
        """Return an immutable snapshot of the current request, if any."""

        with self._lock:
            return self._pending.request if self._pending is not None else None

    @property
    def is_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def begin(
        self,
        request_id: str,
        prior: Sequence[ObservationPrior],
        origin: ObservationOrigin,
        memory_intent: ObservationMemoryIntent = ObservationMemoryIntent.TRANSIENT,
    ) -> ObservationRequest:
        """Adopt the plan blocker's exact id without invoking the model."""

        identifier = _request_id(request_id)
        prior_objects = _prior_objects(prior)
        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        if not isinstance(memory_intent, ObservationMemoryIntent):
            raise TypeError("memory_intent must be an ObservationMemoryIntent")
        with self._lock:
            if self._pending is not None:
                raise ObservationBusyError(
                    f"observation {self._pending.request.request_id} is already pending"
                )
            request = ObservationRequest(
                request_id=identifier,
                prior=prior_objects,
                origin=origin,
                memory_intent=memory_intent,
            )
            self._pending = _PendingObservation(request=request)
            return request

    def complete(self, request_id: str, jpeg: bytes) -> ObservationResponse:
        """Validate one capture, update memory durably, and release the block.

        Brain and store failures leave the exact request pending. A validated
        brain result is cached without its image so a storage retry does not
        send the frame to the model again.
        """

        frame = _complete_jpeg(jpeg)
        digest = hashlib.sha256(frame).digest()
        published: tuple[SceneObject, ...]

        with self._lock:
            pending = self._matching_pending(request_id)
            if pending.response is None:
                LOGGER.info(
                    "SCENE observe prior=%s",
                    json.dumps(
                        [
                            {"id": item.id, "canonical": item.canonical}
                            for item in pending.request.prior
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
                response = self._brain.observe(
                    frame,
                    pending.request.prior,
                    pending.request.origin,
                )
                if not isinstance(response, ObservationResponse):
                    raise TypeError("brain.observe must return validated observation facts")
                response = _durable_response(response, pending.request.prior)
                LOGGER.info("SCENE noticed=%s", _observation_log(response))
                observed_at = _timestamp(self._clock())
                pending.response = response
                pending.observed_at = observed_at
                pending.jpeg_digest = digest
            else:
                if pending.jpeg_digest != digest:
                    raise ObservationImageError(
                        "storage retry must use the original complete JPEG"
                    )
                response = pending.response
                observed_at = pending.observed_at

            if observed_at is None:
                raise AssertionError("validated observation is missing its timestamp")
            existing = self._store.load()
            candidates = _memory_candidates(
                existing,
                pending.request.prior,
                response,
                observed_at,
                pending.request.memory_intent,
            )
            published = tuple(
                self._store.update(tuple(candidate.record for candidate in candidates))
            )
            LOGGER.info("SCENE memory saved=%s", _memory_log(published))
            identity_candidates = tuple(
                candidate for candidate in candidates if candidate.assigns_identity
            )
            assigned_ids = {
                candidate.visible_index: published[index].id
                for index, candidate in enumerate(identity_candidates)
                if candidate.visible_index is not None
            }
            response = ObservationResponse(
                visible=tuple(
                    replace(item, match=assigned_ids.get(index, item.match))
                    for index, item in enumerate(response.visible)
                ),
                focus=response.focus,
                present_prior_ids=response.present_prior_ids,
                raw_saturated=response.raw_saturated,
            )
            self._pending = None

        if self._publish is not None:
            self._publish(published)
        return response

    def cancel(self, request_id: str) -> None:
        """Cancel only the exact pending request without touching memory."""

        with self._lock:
            self._matching_pending(request_id)
            self._pending = None

    def _matching_pending(self, request_id: str) -> _PendingObservation:
        identifier = _request_id(request_id)
        if self._pending is None:
            raise ObservationRequestError(
                f"observation {identifier} is stale; no request is pending"
            )
        expected = self._pending.request.request_id
        if identifier != expected:
            raise ObservationRequestError(
                f"observation {identifier} does not match pending {expected}"
            )
        return self._pending


def _request_id(value: object) -> str:
    """Validate an opaque executor-owned id without minting or interpreting it."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ObservationRequestError("request id must be a non-empty single-line string")
    return value


def _prior_objects(values: Sequence[ObservationPrior]) -> tuple[ObservationPrior, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("prior must be a sequence of ObservationPrior values")
    result: list[ObservationPrior] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, ObservationPrior):
            raise TypeError("prior must contain ObservationPrior values")
        if not is_durable_scene_canonical(value.canonical):
            LOGGER.warning("dropping non-durable observation prior %s", value.canonical)
            continue
        if value.id in seen:
            continue
        seen.add(value.id)
        result.append(value)
    return tuple(result)


def _complete_jpeg(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise ObservationImageError("capture must be immutable JPEG bytes")
    if len(value) < 4 or not value.startswith(b"\xff\xd8"):
        raise ObservationImageError("capture is missing the JPEG start marker")
    first_end = value.find(b"\xff\xd9", 2)
    if first_end < 0:
        raise ObservationImageError("capture is missing the JPEG end marker")
    if first_end != len(value) - 2:
        raise ObservationImageError("capture must contain one JPEG with no trailing data")
    return value


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("observation clock must return a finite timestamp")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("observation clock must return a finite nonnegative timestamp")
    return timestamp


def _memory_candidates(
    existing: tuple[SceneObject, ...],
    request_prior: tuple[ObservationPrior, ...],
    response: ObservationResponse,
    observed_at: float,
    memory_intent: ObservationMemoryIntent,
) -> tuple[_MemoryCandidate, ...]:
    """Convert model facts to store candidates without model-owned metadata."""

    request_ids = {item.id for item in request_prior}
    prior_by_id = {
        record.id: record for record in existing if record.id in request_ids
    }
    prior_by_canonical: dict[str, list[SceneObject]] = {}
    for record in prior_by_id.values():
        prior_by_canonical.setdefault(record.canonical, []).append(record)
    unmatched_counts: dict[str, int] = {}
    for item in response.visible:
        if item.match is None and not item.match_provided:
            unmatched_counts[item.canonical] = unmatched_counts.get(item.canonical, 0) + 1
    selected_indexes: frozenset[int]
    if memory_intent is ObservationMemoryIntent.SCENE:
        selected_indexes = frozenset(range(len(response.visible)))
    elif memory_intent is ObservationMemoryIntent.FOCUS and response.focus is not None:
        focused = response.visible[response.focus]
        selected_indexes = (
            frozenset(range(len(response.visible)))
            if focused.canonical in _SCENE_FOCUS_CANONICALS
            else frozenset({response.focus})
        )
    else:
        selected_indexes = frozenset()
    requested_candidates: list[_MemoryCandidate] = []
    refresh_candidates: list[_MemoryCandidate] = []
    claimed: set[str] = set()
    for index, item in enumerate(response.visible):
        if not isinstance(item, ObservedObject):
            raise TypeError("observation visible entries must be validated ObservedObject values")
        if not is_durable_scene_canonical(item.canonical):
            continue
        previous = None
        if (
            item.match is not None
            and item.match in request_ids
            and item.match not in claimed
        ):
            previous = prior_by_id.get(item.match)
        if (
            previous is None
            and item.match is None
            and not item.match_provided
            and unmatched_counts[item.canonical] == 1
        ):
            available = [
                record
                for record in prior_by_canonical.get(item.canonical, ())
                if record.id not in claimed
            ]
            if len(available) == 1:
                previous = available[0]
        if previous is not None:
            claimed.add(previous.id)
        if index in selected_indexes:
            requested_candidates.append(
                _MemoryCandidate(
                    index,
                    _new_candidate(item, index, observed_at, previous),
                    True,
                )
            )
        elif previous is not None:
            refresh_candidates.append(
                _MemoryCandidate(
                    index,
                    _refresh_candidate(item, previous, observed_at),
                )
            )
    presence = response.present_prior_ids or ()
    for object_id in presence:
        previous = prior_by_id.get(object_id)
        if previous is None or previous.id in claimed:
            continue
        claimed.add(previous.id)
        refresh_candidates.append(
            _MemoryCandidate(
                None,
                replace(previous, last_seen=observed_at, present=True),
            )
        )
    candidates = (*requested_candidates, *refresh_candidates)
    return tuple(candidates[:MAX_SCENE_OBJECTS])


def _durable_response(
    response: ObservationResponse,
    request_prior: tuple[ObservationPrior, ...],
) -> ObservationResponse:
    """Apply durable-memory eligibility even to directly constructed facts."""

    visible: list[ObservedObject] = []
    retained_indexes: dict[int, int] = {}
    for index, item in enumerate(response.visible):
        if not is_durable_scene_canonical(item.canonical):
            LOGGER.warning("dropping non-durable observed object %s", item.canonical)
            continue
        retained_indexes[index] = len(visible)
        visible.append(item)
    focus = None if response.focus is None else retained_indexes.get(response.focus)
    if response.present_prior_ids is None:
        presence = None
    else:
        eligible_ids = {item.id for item in request_prior}
        presence = tuple(
            object_id
            for object_id in response.present_prior_ids
            if object_id in eligible_ids
        )
    return ObservationResponse(
        visible=tuple(visible),
        focus=focus,
        present_prior_ids=presence,
        raw_saturated=response.raw_saturated,
    )


def _observation_log(response: ObservationResponse) -> str:
    return json.dumps(
        {
            "visible": [
                {
                    "match": item.match,
                    "label": item.label,
                    "canonical": item.canonical,
                    "attributes": list(item.attributes),
                    "bbox_norm": None if item.bbox_norm is None else list(item.bbox_norm),
                }
                for item in response.visible
            ],
            "focus": response.focus,
            "present_prior_ids": response.present_prior_ids,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _memory_log(records: tuple[SceneObject, ...]) -> str:
    return json.dumps(
        [
            {
                "id": record.id,
                "label": record.label,
                "canonical": record.canonical,
                "attributes": list(record.attributes),
                "bbox_norm": None if record.bbox_norm is None else list(record.bbox_norm),
                "first_seen": record.first_seen,
                "last_seen": record.last_seen,
                "present": record.present,
                "priority": record.priority,
                "requested_at": record.requested_at,
            }
            for record in records
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _new_candidate(
    item: ObservedObject,
    index: int,
    observed_at: float,
    previous: SceneObject | None,
) -> SceneObject:
    return SceneObject(
        id=previous.id if previous is not None else f"candidate_{index:03d}",
        label=item.label,
        canonical=item.canonical,
        attributes=tuple(
            dict.fromkeys(
                (*(previous.attributes if previous is not None else ()), *item.attributes)
            )
        ),
        bbox_norm=item.bbox_norm,
        first_seen=previous.first_seen if previous is not None else observed_at,
        last_seen=observed_at,
        present=True,
        priority="requested",
        requested_at=observed_at,
    )


def _refresh_candidate(
    item: ObservedObject,
    previous: SceneObject,
    observed_at: float,
) -> SceneObject:
    """Refresh presence/geometry without changing durable descriptive facts."""

    return replace(
        previous,
        bbox_norm=item.bbox_norm if item.bbox_norm is not None else previous.bbox_norm,
        last_seen=observed_at,
        present=True,
    )


__all__ = [
    "MemoryPublisher",
    "ObservationBusyError",
    "ObservationCoordinator",
    "ObservationError",
    "ObservationImageError",
    "ObservationMemoryIntent",
    "ObservationRequest",
    "ObservationRequestError",
]
