"""Body-owned animation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias


JointName: TypeAlias = Literal[
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "neck_yaw",
    "head_pitch",
]

JOINT_NAMES: Final[tuple[JointName, ...]] = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "neck_yaw",
    "head_pitch",
)


@dataclass(frozen=True, slots=True)
class JointVector:
    base_yaw: float = 0.0
    shoulder_pitch: float = 0.0
    elbow_pitch: float = 0.0
    neck_yaw: float = 0.0
    head_pitch: float = 0.0


__all__ = ["JOINT_NAMES", "JointName", "JointVector"]
