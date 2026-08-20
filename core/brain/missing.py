"""Pure local missing-object comparison.

The vision model reports visible facts. This module alone performs the set
comparison required by the PRD; cloud narration is owned by the generalized
observation resolver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import ObservationPrior, ObservationResponse, normalize_canonical_label


@dataclass(frozen=True, slots=True)
class MissingComparison:
    """Immutable evidence for the Python-computed set difference."""

    baseline: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationMissingComparison:
    """Stable-id difference plus canonical labels safe for narration."""

    baseline_ids: tuple[str, ...]
    present_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    missing_labels: tuple[str, ...]


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
    baseline: Sequence[ObservationPrior],
    observation: ObservationResponse,
) -> ObservationMissingComparison:
    """Compare against every object the observation says is visible.

    The vision model reports one nearest-first visible list. Python alone
    compares that list with the stored baseline.
    """

    if not isinstance(observation, ObservationResponse):
        raise TypeError("observation must be a validated ObservationResponse")
    if isinstance(baseline, (str, bytes, bytearray)) or not isinstance(
        baseline, Sequence
    ):
        raise TypeError("baseline must be a sequence of ObservationPrior values")
    priors = tuple(baseline)
    if not all(isinstance(item, ObservationPrior) for item in priors):
        raise TypeError("baseline must contain ObservationPrior values")
    baseline_ids = tuple(dict.fromkeys(item.id for item in priors))
    present_ids = tuple(
        dict.fromkeys(
            item.match for item in observation.visible if item.match is not None
        )
    )
    present_set = frozenset(present_ids)
    missing_ids = tuple(item for item in baseline_ids if item not in present_set)
    by_id = {item.id: item.canonical for item in priors}
    return ObservationMissingComparison(
        baseline_ids,
        present_ids,
        missing_ids,
        tuple(by_id[item] for item in missing_ids),
    )


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
    "ObservationMissingComparison",
    "compute_observation_missing",
    "compute_missing",
]
