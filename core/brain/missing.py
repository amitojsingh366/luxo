"""Pure local missing-object comparison.

The vision model reports visible facts. This module alone performs the set
comparison required by the PRD; cloud narration is owned by the generalized
observation resolver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import (
    OBSERVATION_MAX_VISIBLE,
    ObservationPrior,
    ObservationResponse,
    normalize_canonical_label,
)


@dataclass(frozen=True, slots=True)
class MissingComparison:
    """Immutable evidence for the Python-computed set difference."""

    baseline: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationMissingComparison:
    """Stable-id difference plus canonical labels safe for narration."""

    baseline: tuple[ObservationPrior, ...]
    present_ids: tuple[str, ...]

    @property
    def baseline_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.baseline)

    @property
    def missing(self) -> tuple[ObservationPrior, ...]:
        present = frozenset(self.present_ids)
        return tuple(item for item in self.baseline if item.id not in present)

    @property
    def missing_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.missing)

    @property
    def missing_labels(self) -> tuple[str, ...]:
        return tuple(item.canonical for item in self.missing)


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
    baseline = tuple({item.id: item for item in priors}.values())
    baseline_ids = frozenset(item.id for item in baseline)
    retained = [item.match for item in observation.visible if item.match is not None]
    evidence = observation.present_prior_ids
    evidence_valid = evidence is not None and all(
        value in baseline_ids for value in evidence
    )
    if evidence_valid:
        retained.extend(evidence or ())
    elif len(observation.visible) == OBSERVATION_MAX_VISIBLE:
        # A saturated detail list may omit lower-priority objects that remain
        # visible. Without trustworthy presence evidence, silence is safer
        # than claiming that every omitted prior disappeared.
        retained.extend(item.id for item in baseline)
    return ObservationMissingComparison(baseline, tuple(dict.fromkeys(retained)))


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
