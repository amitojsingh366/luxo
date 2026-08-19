"""Local missing-object comparison and minimal narration boundary.

The model performs perception and narration; this module alone performs the
set comparison required by the PRD. It is synchronous and must be called from a
worker thread because ``BrainClient.narrate`` may perform blocking network I/O.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from .client import BrainClient
from .schema import ObservationResponse, PlanResponse, normalize_canonical_label


@dataclass(frozen=True, slots=True)
class MissingComparison:
    """Immutable evidence for the Python-computed set difference."""

    baseline: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MissingNarration:
    """Validated narration paired with the local comparison that produced it."""

    response: PlanResponse
    comparison: MissingComparison


def compute_missing(
    baseline: Sequence[str],
    present: Sequence[str],
) -> MissingComparison:
    """Compute ``L0 - present`` locally in stable first-baseline order.

    Both inputs are normalized with the same canonical-label contract used by
    observation prompts and scene memory. Duplicates collapse after
    normalization. Labels reported as present but absent from the baseline are
    retained as local evidence but can never add an item to ``missing``.
    """

    normalized_baseline = _canonical_labels(baseline, "baseline")
    normalized_present = _canonical_labels(present, "present")
    present_set = frozenset(normalized_present)
    missing = tuple(
        label for label in normalized_baseline if label not in present_set
    )
    return MissingComparison(normalized_baseline, normalized_present, missing)


class MissingObjectCoordinator:
    """Serialize minimal narration calls without retaining observation data.

    The coordinator is stateless apart from its lock and client reference. A
    narration failure propagates unchanged and may be retried by invoking the
    method again; no frame, observation record, or partial response is cached.
    """

    def __init__(self, brain: BrainClient) -> None:
        if not callable(getattr(brain, "narrate", None)):
            raise TypeError("brain must provide narrate(missing)")
        self._brain = brain
        self._lock = threading.Lock()

    def compare_and_narrate(
        self,
        baseline: Sequence[str],
        observation: ObservationResponse,
    ) -> MissingNarration:
        """Compute locally, then call ``narrate`` once with only missing labels."""

        if not isinstance(observation, ObservationResponse):
            raise TypeError("observation must be a validated ObservationResponse")
        comparison = compute_missing(baseline, observation.present)

        # Brain clients need not be re-entrant. Serialization also makes the
        # one-call boundary deterministic when worker tasks race.
        with self._lock:
            response = self._brain.narrate(comparison.missing)
        if not isinstance(response, PlanResponse):
            raise TypeError("brain.narrate must return a validated PlanResponse")
        return MissingNarration(response=response, comparison=comparison)


def _canonical_labels(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence of canonical strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        try:
            label = normalize_canonical_label(value)
        except ValueError as error:
            raise ValueError(f"{field}[{index}] is not a canonical label") from error
        if label not in seen:
            seen.add(label)
            normalized.append(label)
    return tuple(normalized)


__all__ = [
    "MissingComparison",
    "MissingNarration",
    "MissingObjectCoordinator",
    "compute_missing",
]
