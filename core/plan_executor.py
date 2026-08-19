"""Public semantic-plan execution boundary.

Plans contain only the closed model verbs. This interface cannot accept joint
angles or animation timing: translating each verb into motion belongs to the
body-side animation modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .blackboard import BlackboardSnapshot
from .brain.schema import Action


class PlanExecutor(Protocol):
    @property
    def depth(self) -> int: ...

    @property
    def blocked_on_observation(self) -> bool: ...

    def submit(self, plan: Sequence[Action]) -> None: ...

    def clear(self) -> None: ...

    def tick(self, snapshot: BlackboardSnapshot, now: float) -> Action | None: ...

    def release_observation(self, request_id: str) -> None: ...


__all__ = ["PlanExecutor"]
