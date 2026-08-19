"""Public behavior-state-machine boundary.

The concrete transition logic belongs to the later ``fsm`` packet. The FSM is
the only component permitted to assign character intent from blackboard facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .blackboard import BlackboardSnapshot


class BehaviorState(str, Enum):
    BOOT = "BOOT"
    DORMANT = "DORMANT"
    NOTICING = "NOTICING"
    ENGAGED = "ENGAGED"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INSPECTING = "INSPECTING"
    ACTING = "ACTING"
    DISENGAGING = "DISENGAGING"


@dataclass(frozen=True, slots=True)
class Transition:
    t: float
    previous: BehaviorState
    current: BehaviorState
    reason: str


class BehaviorFSM(Protocol):
    @property
    def state(self) -> BehaviorState: ...

    def tick(
        self,
        snapshot: BlackboardSnapshot,
        now: float,
    ) -> Transition | None: ...


__all__ = ["BehaviorFSM", "BehaviorState", "Transition"]
