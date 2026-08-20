"""Pure local missing-object comparison.

The vision model reports visible facts. This module alone performs the set
comparison required by the PRD; cloud narration is owned by the generalized
observation resolver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import ObservationResponse, normalize_canonical_label


@dataclass(frozen=True, slots=True)
class MissingComparison:
    """Immutable evidence for the Python-computed set difference."""

    baseline: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]


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


def compute_observation_missing(
    baseline: Sequence[str],
    observation: ObservationResponse,
) -> MissingComparison:
    """Compare against every object the observation says is visible.

    Vision models sometimes repeat an already-known canonical in ``new``
    instead of ``present``. Scene memory correctly treats either location as
    visible, so missing-object narration must do the same or it will announce
    objects that are plainly still in frame.
    """

    if not isinstance(observation, ObservationResponse):
        raise TypeError("observation must be a validated ObservationResponse")
    visible = (
        observation.present
        + tuple(item.canonical for item in observation.known)
        + tuple(item.canonical for item in observation.new)
    )
    return compute_missing(baseline, visible)


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
    "compute_observation_missing",
    "compute_missing",
]
