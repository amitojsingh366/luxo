"""Public analytic look-at boundary.

The body faces world -x because the head joint carries a pi Z flip. Azimuth is
measured from -x toward +y, and positive head pitch looks down. There is no roll
DOF: curiosity must use whole-body lean/crane rather than a fake head tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..brain.schema import PostureName
from . import JointVector


@dataclass(frozen=True, slots=True)
class LookAtTarget:
    target: str
    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True, slots=True)
class LookAtSolution:
    offsets: JointVector
    requested_posture: PostureName | None = None


class LookAtSolver(Protocol):
    def reset(self) -> None: ...

    def solve(
        self,
        target: LookAtTarget,
        current: JointVector,
        dt: float,
    ) -> LookAtSolution: ...


__all__ = ["LookAtSolution", "LookAtSolver", "LookAtTarget"]
