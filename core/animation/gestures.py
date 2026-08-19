"""Public additive gesture-controller boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..brain.schema import GestureName, PostureName
from . import JointVector


@dataclass(frozen=True, slots=True)
class GestureSample:
    offsets: JointVector
    complete: bool


class GestureController(Protocol):
    @property
    def active(self) -> GestureName | PostureName | None: ...

    def start(self, name: GestureName | PostureName, now: float) -> None: ...

    def cancel(self, now: float) -> None: ...

    def sample(self, now: float) -> GestureSample: ...


__all__ = ["GestureController", "GestureSample"]
