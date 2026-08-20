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

from .client import BrainClient, ObservationOrigin
from .memory import SceneMemoryStore, SceneObject
from .schema import (
    ObservationPrior,
    ObservationResponse,
    ObservedObject,
)

LOGGER = logging.getLogger(__name__)


class ObservationError(RuntimeError):
    """An observation cannot advance from its current state."""


class ObservationBusyError(ObservationError):
    """A second observation was requested while one is pending."""


class ObservationRequestError(ObservationError):
    """A completion or cancellation does not match the pending request."""


class ObservationImageError(ValueError):
    """A capture is not exactly one complete JPEG image."""


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    """The exact origin and stable identities bound to one capture."""

    request_id: str
    prior: tuple[ObservationPrior, ...]
    origin: ObservationOrigin


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
    ) -> ObservationRequest:
        """Adopt the plan blocker's exact id without invoking the model."""

        identifier = _request_id(request_id)
        prior_objects = _prior_objects(prior)
        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        with self._lock:
            if self._pending is not None:
                raise ObservationBusyError(
                    f"observation {self._pending.request.request_id} is already pending"
                )
            request = ObservationRequest(
                request_id=identifier,
                prior=prior_objects,
                origin=origin,
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
            )
            published = tuple(self._store.update(candidates))
            LOGGER.info("SCENE memory saved=%s", _memory_log(published))
            response = ObservationResponse(
                visible=tuple(
                    replace(item, match=published[index].id)
                    for index, item in enumerate(response.visible)
                ),
                focus=response.focus,
                present_prior_ids=response.present_prior_ids,
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
) -> tuple[SceneObject, ...]:
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
    candidates: list[SceneObject] = []
    claimed: set[str] = set()
    for index, item in enumerate(response.visible):
        if not isinstance(item, ObservedObject):
            raise TypeError("observation visible entries must be validated ObservedObject values")
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
        candidates.append(_new_candidate(item, index, observed_at, previous))
    return tuple(candidates)


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
        attributes=item.attributes,
        bbox_norm=item.bbox_norm,
        first_seen=previous.first_seen if previous is not None else observed_at,
        last_seen=observed_at,
        present=True,
    )


__all__ = [
    "MemoryPublisher",
    "ObservationBusyError",
    "ObservationCoordinator",
    "ObservationError",
    "ObservationImageError",
    "ObservationRequest",
    "ObservationRequestError",
]
