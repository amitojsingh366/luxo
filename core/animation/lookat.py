"""Stateful analytic aiming for Luxo's light emitter.

The URDF's head-frame pi flip makes world -x the body's front. Because yaw is
applied before that flip, positive input azimuth rotates the emitter from -x
toward -y; positive head pitch looks down. The target angles describe the
direction from ``light_emitter_link`` to the render camera or another world
target; ``camera_link`` is not the aiming frame.

This is deliberately not inverse kinematics. Base and neck share azimuth while
the neck leads and exponentially recentres. The pitch solution does include
the two upstream arm joints: after the pi flip, emitter elevation in the
positive-down convention is ``head - shoulder - elbow``. Unreachable elevation
asks the gesture layer for a whole-body posture. There is no roll joint to
author.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, pi
from typing import Final

from ..brain.schema import PostureName
from . import JointVector
from .poses import SOFT_LIMITS


NECK_LEAD_LIMIT_RAD: Final = 0.5
NECK_RECENTER_TIME_CONSTANT_S: Final = 0.9
_FULL_TURN: Final = 2.0 * pi
_HEAD_PITCH_MIN, _HEAD_PITCH_MAX = SOFT_LIMITS["head_pitch"]


@dataclass(frozen=True, slots=True)
class LookAtTarget:
    """A world-space emitter-axis target expressed in Luxo's conventions."""

    target: str
    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True, slots=True)
class LookAtSolution:
    """Additive gaze offsets plus an optional whole-body posture request."""

    offsets: JointVector
    requested_posture: PostureName | None = None


class LookAtSolver:
    """Split emitter azimuth between a leading neck and following base.

    A newly acquired target is unwrapped beside the current emitter heading.
    The neck takes up to 0.5 rad of the change, leaving the base at its current
    angle whenever that is sufficient. While the target remains unchanged,
    the neck lead decays toward zero with the specified 0.9 second time
    constant and the base absorbs exactly the same angle. This keeps
    ``base_yaw + neck_yaw`` on target without authoring base overshoot.
    """

    __slots__ = (
        "_lead_rad",
        "_raw_azimuth_rad",
        "_target",
        "_unwrapped_azimuth_rad",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget target-unwrapping and neck-lead history."""

        self._target: str | None = None
        self._raw_azimuth_rad = 0.0
        self._unwrapped_azimuth_rad = 0.0
        self._lead_rad = 0.0

    def solve(
        self,
        target: LookAtTarget,
        current: JointVector,
        dt: float,
    ) -> LookAtSolution:
        """Return gaze offsets relative to the supplied composed pose.

        ``dt`` advances recentering only for an already-held target. A target
        acquired on this call receives its full neck lead immediately; decay
        begins on the next call. Desired aiming angles are absolute internally,
        but the layer contract returns ``desired - current`` for additive
        composition. All inputs are validated before state changes.
        """

        (
            target_name,
            raw_azimuth,
            elevation,
            current_base,
            current_heading,
            arm_pitch,
        ) = _validated_inputs(target, current, dt)

        if self._target is None:
            unwrapped_azimuth = current_heading + _wrapped_delta(
                raw_azimuth, current_heading
            )
            lead = _clamp(
                unwrapped_azimuth - current_base,
                -NECK_LEAD_LIMIT_RAD,
                NECK_LEAD_LIMIT_RAD,
            )
        else:
            target_delta = _wrapped_delta(raw_azimuth, self._raw_azimuth_rad)
            unwrapped_azimuth = self._unwrapped_azimuth_rad + target_delta
            if target_name != self._target or target_delta != 0.0:
                # Preserve the prior absolute base target when the neck can
                # absorb a target change; the neck is the authored lead.
                prior_base = (
                    self._unwrapped_azimuth_rad - self._lead_rad
                )
                lead = _clamp(
                    unwrapped_azimuth - prior_base,
                    -NECK_LEAD_LIMIT_RAD,
                    NECK_LEAD_LIMIT_RAD,
                )
            else:
                lead = self._lead_rad * exp(
                    -float(dt) / NECK_RECENTER_TIME_CONSTANT_S
                )

        self._target = target_name
        self._raw_azimuth_rad = raw_azimuth
        self._unwrapped_azimuth_rad = unwrapped_azimuth
        self._lead_rad = lead

        head_pitch, posture = _elevation_solution(elevation, arm_pitch)
        return LookAtSolution(
            offsets=JointVector(
                base_yaw=unwrapped_azimuth - lead - current.base_yaw,
                neck_yaw=lead - current.neck_yaw,
                head_pitch=head_pitch - current.head_pitch,
            ),
            requested_posture=posture,
        )


def _validated_inputs(
    target: LookAtTarget,
    current: JointVector,
    dt: float,
) -> tuple[str, float, float, float, float, float]:
    if not isinstance(target, LookAtTarget):
        raise TypeError("target must be a LookAtTarget")
    if not isinstance(target.target, str) or not target.target:
        raise ValueError("target.target must be a non-empty string")
    if not isinstance(current, JointVector):
        raise TypeError("current must be a JointVector")

    raw_azimuth = _finite(target.azimuth_rad, "target.azimuth_rad")
    elevation = _finite(target.elevation_rad, "target.elevation_rad")
    checked_dt = _finite(dt, "dt")
    if checked_dt <= 0.0:
        raise ValueError("dt must be greater than zero")

    values = (
        _finite(current.base_yaw, "current.base_yaw"),
        _finite(current.shoulder_pitch, "current.shoulder_pitch"),
        _finite(current.elbow_pitch, "current.elbow_pitch"),
        _finite(current.neck_yaw, "current.neck_yaw"),
        _finite(current.head_pitch, "current.head_pitch"),
    )
    current_heading = values[0] + values[3]
    if not isfinite(current_heading):
        raise ValueError("current base and neck heading must be finite")
    arm_pitch = values[1] + values[2]
    if not isfinite(arm_pitch):
        raise ValueError("current shoulder and elbow pitch must be finite")
    return (
        target.target,
        raw_azimuth,
        elevation,
        values[0],
        current_heading,
        arm_pitch,
    )


def _finite(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _wrapped_delta(target: float, reference: float) -> float:
    """Return the deterministic shortest signed turn in [-pi, pi)."""

    return (target - reference + pi) % _FULL_TURN - pi


def _elevation_solution(
    elevation: float,
    arm_pitch: float,
) -> tuple[float, PostureName | None]:
    # URDF order is Ry(shoulder) * Ry(elbow) * Rz(pi) * Ry(head).
    # Therefore the emitter's positive-down elevation is head - shoulder -
    # elbow, and the absolute head target must carry the upstream arm pitch.
    required_head_pitch = elevation + arm_pitch
    if required_head_pitch > _HEAD_PITCH_MAX:
        return _HEAD_PITCH_MAX, PostureName.STOOP
    if required_head_pitch < _HEAD_PITCH_MIN:
        return _HEAD_PITCH_MIN, PostureName.CRANE
    return required_head_pitch, None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


__all__ = [
    "LookAtSolution",
    "LookAtSolver",
    "LookAtTarget",
    "NECK_LEAD_LIMIT_RAD",
    "NECK_RECENTER_TIME_CONSTANT_S",
]
