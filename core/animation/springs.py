"""Per-joint second-order spring-damper motion.

Springs only integrate summed targets. Velocity and SOFT-limit clamps belong
to the final output stage; hard limits are never command limits. The model has
five joints and no roll degree of freedom. Its head-frame pi flip (world -x is
front and positive head pitch is down) does not change scalar integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, isfinite, sin, sqrt
from typing import Final

from . import JOINT_NAMES, JointName, JointVector


@dataclass(frozen=True, slots=True)
class SpringState:
    position: float
    velocity: float


def _finite(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


class SpringDamper:
    """A constant-target analytic integrator for one second-order spring."""

    __slots__ = ("_omega", "_position", "_velocity", "_zeta")

    def __init__(
        self,
        omega: float,
        zeta: float,
        position: float = 0.0,
        velocity: float = 0.0,
    ) -> None:
        self._omega = _finite(omega, "omega")
        self._zeta = _finite(zeta, "zeta")
        if self._omega <= 0.0:
            raise ValueError("omega must be greater than zero")
        if self._zeta <= 0.0:
            raise ValueError("zeta must be greater than zero")
        self._position = 0.0
        self._velocity = 0.0
        self.reset(position, velocity)

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def zeta(self) -> float:
        return self._zeta

    @property
    def state(self) -> SpringState:
        return SpringState(self._position, self._velocity)

    def reset(self, position: float, velocity: float = 0.0) -> None:
        checked_position = _finite(position, "position")
        checked_velocity = _finite(velocity, "velocity")
        self._position = checked_position
        self._velocity = checked_velocity

    def step(self, target: float, dt: float) -> SpringState:
        checked_target = _finite(target, "target")
        checked_dt = _finite(dt, "dt")
        if checked_dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        self._advance(checked_target, checked_dt)
        return self.state

    def _advance(self, target: float, dt: float) -> None:
        """Advance without allocating; callers must validate both arguments."""
        omega = self._omega
        zeta = self._zeta
        offset = self._position - target
        velocity = self._velocity
        scaled_time = omega * dt

        # Any representable damping has fully settled by an infinite scaled
        # time. Handling it here also avoids passing infinity to sin/cos.
        if not isfinite(scaled_time):
            self._position = target
            self._velocity = 0.0
            return

        if zeta < 1.0:
            decay_rate = zeta * omega
            damped_omega = omega * sqrt(1.0 - zeta * zeta)
            decay = exp(-decay_rate * dt)
            if decay == 0.0:
                self._position = target
                self._velocity = 0.0
                return
            phase = damped_omega * dt
            cosine = cos(phase)
            sine = sin(phase)
            new_offset = decay * (
                offset * cosine
                + (velocity + decay_rate * offset) * sine / damped_omega
            )
            new_velocity = decay * (
                velocity * cosine
                - (decay_rate * velocity + omega * omega * offset)
                * sine
                / damped_omega
            )
        elif zeta == 1.0:
            decay = exp(-scaled_time)
            slope = velocity + omega * offset
            new_offset = decay * (offset + slope * dt)
            new_velocity = decay * (velocity - omega * slope * dt)
        else:
            root = sqrt(zeta * zeta - 1.0)
            slow_root = -omega / (zeta + root)
            fast_root = -omega * (zeta + root)
            slow_weight = (velocity - fast_root * offset) / (slow_root - fast_root)
            fast_weight = offset - slow_weight
            slow_term = slow_weight * exp(slow_root * dt)
            fast_term = fast_weight * exp(fast_root * dt)
            new_offset = slow_term + fast_term
            new_velocity = slow_root * slow_term + fast_root * fast_term

        position = target + new_offset
        if not isfinite(position) or not isfinite(new_velocity):
            raise ValueError("spring integration produced a non-finite state")
        self._position = position
        self._velocity = new_velocity


_PARAMETERS: Final[tuple[tuple[float, float], ...]] = (
    (9.0, 0.85),
    (6.5, 0.90),
    (7.5, 0.88),
    (14.0, 0.62),
    (13.0, 0.60),
)
_JOINT_INDEX: Final[dict[JointName, int]] = {
    joint: index for index, joint in enumerate(JOINT_NAMES)
}


class SpringBank:
    """The fixed PRD spring set in body joint order."""

    __slots__ = ("_springs",)

    def __init__(self, positions: JointVector | None = None) -> None:
        self._springs = tuple(SpringDamper(*values) for values in _PARAMETERS)
        if positions is not None:
            self.reset(positions)

    def spring_for(self, joint: JointName) -> SpringDamper:
        try:
            return self._springs[_JOINT_INDEX[joint]]
        except KeyError as error:
            raise ValueError(f"unknown joint: {joint!r}") from error

    def state_for(self, joint: JointName) -> SpringState:
        return self.spring_for(joint).state

    def reset(self, positions: JointVector) -> None:
        values = self._checked_values(positions, "positions")
        for spring, position in zip(self._springs, values, strict=True):
            spring.reset(position)

    def step(self, targets: JointVector, dt: float) -> JointVector:
        checked_dt = _finite(dt, "dt")
        if checked_dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        values = self._checked_values(targets, "targets")
        for spring, target in zip(self._springs, values, strict=True):
            spring._advance(target, checked_dt)
        return JointVector(
            self._springs[0]._position,
            self._springs[1]._position,
            self._springs[2]._position,
            self._springs[3]._position,
            self._springs[4]._position,
        )

    @staticmethod
    def _checked_values(vector: JointVector, label: str) -> tuple[float, ...]:
        return tuple(
            _finite(getattr(vector, joint), f"{label}.{joint}")
            for joint in JOINT_NAMES
        )


__all__ = ["SpringBank", "SpringDamper", "SpringState"]
