"""Public additive animation-layer boundary.

Later feature packets evaluate idle, gaze, gesture, light, and speech-bob in
that order. Layer sampling is pure CPU work and must never wait on I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, isfinite, sin, tau
from operator import index
from typing import Final, Protocol

from ..blackboard import BlackboardSnapshot
from . import JointVector


_IDLE_AMPLITUDE: Final = 0.012
_IDLE_FREQUENCY_HZ: Final = 0.22
_NOISE_AMPLITUDE: Final = 0.004
_MASK_64: Final = (1 << 64) - 1
_NECK_NOISE_SALT: Final = 0xA0761D6478BD642F
_HEAD_NOISE_SALT: Final = 0xE7037ED1A0B428DB


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


class IdleLayer:
    """Stateless breathing and micro-jitter driven by absolute time."""

    __slots__ = ("_seed",)

    def __init__(self, seed: int = 0) -> None:
        try:
            self._seed = index(seed) & _MASK_64
        except TypeError as error:
            raise ValueError("seed must be an integer") from error

    def sample(self, snapshot: BlackboardSnapshot, now: float) -> LayerSample:
        del snapshot
        checked_now = _finite(now, "now")
        shoulder = _IDLE_AMPLITUDE * sin(
            tau * _IDLE_FREQUENCY_HZ * checked_now
        )
        neck = _NOISE_AMPLITUDE * _gradient_noise(
            checked_now, self._seed ^ _NECK_NOISE_SALT
        )
        head = _NOISE_AMPLITUDE * _gradient_noise(
            checked_now, self._seed ^ _HEAD_NOISE_SALT
        )
        return LayerSample(joints=JointVector(0.0, shoulder, 0.0, neck, head))


class LayerMixer:
    """Combine caller-ordered additive body layers and independent light."""

    __slots__ = ()

    def sum(self, samples: Sequence[LayerSample]) -> LayerSample:
        base = shoulder = elbow = neck = head = 0.0
        light: LightCommand | None = None

        for sample in samples:
            joints = sample.joints
            base += _finite(joints.base_yaw, "layer base_yaw")
            shoulder += _finite(joints.shoulder_pitch, "layer shoulder_pitch")
            elbow += _finite(joints.elbow_pitch, "layer elbow_pitch")
            neck += _finite(joints.neck_yaw, "layer neck_yaw")
            head += _finite(joints.head_pitch, "layer head_pitch")
            if sample.light is not None:
                _validate_light(sample.light)
                light = sample.light

        if not (
            isfinite(base)
            and isfinite(shoulder)
            and isfinite(elbow)
            and isfinite(neck)
            and isfinite(head)
        ):
            raise ValueError("summed joints must be finite")
        return LayerSample(JointVector(base, shoulder, elbow, neck, head), light)


def _gradient_noise(position: float, seed: int) -> float:
    """Return continuous, normalized one-dimensional gradient noise."""

    left = floor(position)
    offset = position - left
    fade = offset * offset * offset * (offset * (offset * 6.0 - 15.0) + 10.0)
    left_slope = _gradient(left, seed) * offset
    right_slope = _gradient(left + 1, seed) * (offset - 1.0)
    return 2.0 * (left_slope + fade * (right_slope - left_slope))


def _gradient(lattice: int, seed: int) -> float:
    value = ((lattice & _MASK_64) ^ seed) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    value ^= value >> 31
    return 1.0 if value & 1 else -1.0


def _finite(value: float, label: str) -> float:
    try:
        checked = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not isfinite(checked):
        raise ValueError(f"{label} must be finite")
    return checked


def _validate_light(light: LightCommand) -> None:
    _finite(light.intensity, "layer light intensity")
    _finite(light.color_k, "layer light color_k")
    _finite(light.bloom, "layer light bloom")


__all__ = [
    "AnimationLayer",
    "IdleLayer",
    "LayerMixer",
    "LayerSample",
    "LightCommand",
]
