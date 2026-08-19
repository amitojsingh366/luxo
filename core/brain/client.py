"""Public OpenRouter client boundary.

Network calls run on workers and publish completed results to the blackboard;
the animation tick never calls this interface. Concrete profile payloads,
repair behavior, and transport are assigned to ``brain-client``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .schema import ObservationResponse, PlanResponse


@dataclass(frozen=True, slots=True)
class RecentExchange:
    human_say: str
    lamp_say: str


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


__all__ = ["BrainClient", "RecentExchange"]
