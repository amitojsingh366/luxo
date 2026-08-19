"""Public second-order spring interfaces.

The configured equation is ``xdd = omega^2(target-x) - 2*zeta*omega*xd``.
Integration belongs to the later ``springs`` packet; this scaffold stores no
global motion state and performs no timing work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from . import JointName, JointVector


@dataclass(frozen=True, slots=True)
class SpringState:
    position: float
    velocity: float


class SpringDamper(Protocol):
    @property
    def omega(self) -> float: ...

    @property
    def zeta(self) -> float: ...

    @property
    def state(self) -> SpringState: ...

    def reset(self, position: float, velocity: float = 0.0) -> None: ...

    def step(self, target: float, dt: float) -> SpringState: ...


class SpringBank(Protocol):
    def state_for(self, joint: JointName) -> SpringState: ...

    def reset(self, positions: JointVector) -> None: ...

    def step(self, targets: JointVector, dt: float) -> JointVector: ...


__all__ = ["SpringBank", "SpringDamper", "SpringState"]
