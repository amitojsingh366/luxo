"""Final safety boundary for articulated body commands.

Callers provide targets after all animation layers have already been summed.
This stage applies the mandatory spring, velocity, and SOFT-limit operations in
that order. Hard URDF limits deliberately do not appear in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

from ..blackboard import ClampCounts
from . import JOINT_NAMES, JointName, JointVector
from .springs import SpringBank


@dataclass(frozen=True, slots=True)
class JointConstraint:
    name: JointName
    soft_min: float
    soft_max: float
    max_velocity: float


JOINT_CONSTRAINTS: Final[tuple[JointConstraint, ...]] = (
    JointConstraint("base_yaw", -2.45, 2.45, 1.50),
    JointConstraint("shoulder_pitch", -0.65, 0.95, 0.95),
    JointConstraint("elbow_pitch", -1.70, 0.30, 1.15),
    JointConstraint("neck_yaw", -1.25, 1.25, 1.60),
    JointConstraint("head_pitch", -0.80, 0.60, 1.45),
)


@dataclass(frozen=True, slots=True)
class OutputSample:
    """One safe body command and its cumulative clamp telemetry."""

    positions: JointVector
    velocities: JointVector
    clamps: ClampCounts


class OutputStage:
    """Apply physical output constraints to already-summed joint targets."""

    def __init__(self, spring_bank: SpringBank) -> None:
        self._spring_bank = spring_bank
        self._previous = JointVector()
        self._clamp_counts = ClampCounts()
        self._spring_bank.reset(self._previous)

    @property
    def clamp_counts(self) -> ClampCounts:
        """Return cumulative per-joint clamp event counts since reset."""

        return self._clamp_counts

    def reset(self, positions: JointVector = JointVector()) -> None:
        """Reset spring and output history to a safe, stationary pose.

        Reset positions represent the last emitted command. They must already
        be within the soft envelope; accepting an unsafe history value would
        make the next sample unable to satisfy both output constraints.
        """

        values = _finite_values(positions, "reset positions")
        for constraint, value in zip(JOINT_CONSTRAINTS, values, strict=True):
            if not constraint.soft_min <= value <= constraint.soft_max:
                raise ValueError(
                    f"reset position for {constraint.name} is outside soft limits"
                )

        self._spring_bank.reset(positions)
        self._previous = positions
        self._clamp_counts = ClampCounts()

    def emit(self, summed_targets: JointVector, dt: float) -> OutputSample:
        """Advance springs and emit positions safe for the five-DOF body."""

        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")
        _finite_values(summed_targets, "summed targets")

        sprung = self._spring_bank.step(summed_targets, dt)
        sprung_values = _finite_values(sprung, "spring output")
        previous_values = _values(self._previous)

        emitted: list[float] = []
        velocity_events = 0
        limit_events = 0

        for constraint, previous, proposed in zip(
            JOINT_CONSTRAINTS, previous_values, sprung_values, strict=True
        ):
            max_delta = constraint.max_velocity * dt
            velocity_limited = _clamp(
                proposed, previous - max_delta, previous + max_delta
            )
            if velocity_limited != proposed:
                velocity_events += 1

            soft_limited = _clamp(
                velocity_limited, constraint.soft_min, constraint.soft_max
            )
            if soft_limited != velocity_limited:
                limit_events += 1
            emitted.append(soft_limited)

        positions = JointVector(*emitted)
        velocities = JointVector(
            *(
                (current - previous) / dt
                for current, previous in zip(emitted, previous_values, strict=True)
            )
        )
        self._previous = positions
        self._clamp_counts = ClampCounts(
            velocity=self._clamp_counts.velocity + velocity_events,
            limit=self._clamp_counts.limit + limit_events,
        )
        return OutputSample(positions, velocities, self._clamp_counts)


def _values(vector: JointVector) -> tuple[float, ...]:
    return tuple(getattr(vector, name) for name in JOINT_NAMES)


def _finite_values(vector: JointVector, label: str) -> tuple[float, ...]:
    values = _values(vector)
    try:
        valid = all(isfinite(value) for value in values)
    except TypeError as error:
        raise ValueError(f"{label} must contain only finite numbers") from error
    if not valid:
        raise ValueError(f"{label} must contain only finite numbers")
    return values


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


__all__ = [
    "JOINT_CONSTRAINTS",
    "JointConstraint",
    "OutputSample",
    "OutputStage",
]
