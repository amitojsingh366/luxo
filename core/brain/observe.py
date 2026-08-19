"""Public discrete-observation coordination boundary.

Observation is the only path by which a camera frame may reach the model. It
blocks the semantic plan until the returned object facts update local memory.
Python set-difference for missing objects is intentionally outside this Phase 0
scaffold and must never be delegated to the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .schema import ObservationResponse


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    request_id: str
    prior_canonical: tuple[str, ...]


class ObservationCoordinator(Protocol):
    def begin(self, prior_canonical: Sequence[str]) -> ObservationRequest: ...

    def complete(self, request_id: str, jpeg: bytes) -> ObservationResponse: ...

    def cancel(self, request_id: str) -> None: ...


__all__ = ["ObservationCoordinator", "ObservationRequest"]
