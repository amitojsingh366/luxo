"""Public final motion-output boundary.

The mandatory pipeline is sum -> spring -> velocity clamp -> SOFT-limit clamp.
Hard limits are never command targets. Every velocity and soft-limit clamp is
counted for telemetry; feature implementation belongs to ``output-stage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..blackboard import ClampCounts
from . import JointVector


@dataclass(frozen=True, slots=True)
class OutputSample:
    positions: JointVector
    velocities: JointVector
    clamps: ClampCounts


class OutputStage(Protocol):
    @property
    def clamp_counts(self) -> ClampCounts: ...

    def reset(self, positions: JointVector) -> None: ...

    def emit(self, summed_targets: JointVector, dt: float) -> OutputSample: ...


__all__ = ["OutputSample", "OutputStage"]
