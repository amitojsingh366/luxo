"""Public additive animation-layer boundary.

Later feature packets evaluate idle, gaze, gesture, light, and speech-bob in
that order. Layer sampling is pure CPU work and must never wait on I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..blackboard import BlackboardSnapshot
from . import JointVector


@dataclass(frozen=True, slots=True)
class LightCommand:
    intensity: float
    color_k: int
    pattern: str
    bloom: float


@dataclass(frozen=True, slots=True)
class LayerSample:
    joints: JointVector = JointVector()
    light: LightCommand | None = None


class AnimationLayer(Protocol):
    def sample(self, snapshot: BlackboardSnapshot, now: float) -> LayerSample: ...


class LayerMixer(Protocol):
    def sum(self, samples: Sequence[LayerSample]) -> LayerSample: ...


__all__ = ["AnimationLayer", "LayerMixer", "LayerSample", "LightCommand"]
