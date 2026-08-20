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

from .client import BrainClient
from .memory import SceneMemoryStore, SceneObject
from .schema import (
    ObservationResponse,
    ObservedObject,
    normalize_canonical_label,
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
    """The minimal capture request: an id and canonical labels only."""

    request_id: str
    prior_canonical: tuple[str, ...]


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
        self._next_request = 1
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

    def begin(self, prior_canonical: Sequence[str]) -> ObservationRequest:
        """Reserve one capture request without invoking the model."""

        labels = _canonical_labels(prior_canonical)
        with self._lock:
            if self._pending is not None:
                raise ObservationBusyError(
                    f"observation {self._pending.request.request_id} is already pending"
                )
            request = ObservationRequest(
                request_id=f"obs_{self._next_request}",
                prior_canonical=labels,
            )
            self._next_request += 1
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
                        pending.request.prior_canonical,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
                response = self._brain.observe(frame, pending.request.prior_canonical)
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
            candidates = _memory_candidates(existing, response, observed_at)
            published = tuple(self._store.update(candidates))
            LOGGER.info("SCENE memory saved=%s", _memory_log(published))
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
        if not isinstance(request_id, str) or not request_id:
            raise ObservationRequestError("request id must be a non-empty string")
        if self._pending is None:
            raise ObservationRequestError(
                f"observation {request_id} is stale; no request is pending"
            )
        expected = self._pending.request.request_id
        if request_id != expected:
            raise ObservationRequestError(
                f"observation {request_id} does not match pending {expected}"
            )
        return self._pending


def _canonical_labels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("prior canonical labels must be a sequence of strings")
    return tuple(dict.fromkeys(normalize_canonical_label(value) for value in values))


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
    response: ObservationResponse,
    observed_at: float,
) -> tuple[SceneObject, ...]:
    """Convert model facts to store candidates without model-owned metadata."""

    prior = {record.canonical: record for record in existing}
    present = frozenset(response.present)
    refreshed = {item.canonical: item for item in response.known}
    candidates: list[SceneObject] = []
    for record in existing:
        if record.canonical not in present:
            continue
        current = refreshed.get(record.canonical)
        if current is None:
            candidates.append(replace(record, last_seen=observed_at, present=True))
        else:
            candidates.append(_new_candidate(current, 0, observed_at, prior))
    claimed = {record.canonical for record in candidates}

    for index, item in enumerate(response.new):
        if not isinstance(item, ObservedObject):
            raise TypeError("observation new entries must be validated ObservedObject values")
        if item.canonical in claimed:
            continue
        claimed.add(item.canonical)
        candidates.append(_new_candidate(item, index, observed_at, prior))
    return tuple(candidates)


def _observation_log(response: ObservationResponse) -> str:
    return json.dumps(
        {
            "present": list(response.present),
            "known": [
                {
                    "label": item.label,
                    "canonical": item.canonical,
                    "attributes": list(item.attributes),
                    "bbox_norm": list(item.bbox_norm),
                }
                for item in response.known
            ],
            "new": [
                {
                    "label": item.label,
                    "canonical": item.canonical,
                    "attributes": list(item.attributes),
                    "bbox_norm": list(item.bbox_norm),
                }
                for item in response.new
            ],
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
                "bbox_norm": list(record.bbox_norm),
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
    prior: dict[str, SceneObject],
) -> SceneObject:
    previous = prior.get(item.canonical)
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
