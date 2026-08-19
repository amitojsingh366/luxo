"""Public scene-memory records and persistence interface.

The concrete flat-JSON store belongs to the later ``memory-store`` packet. The
record is deliberately small and local: no embeddings or vector-search types
are part of this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SceneObject:
    id: str
    label: str
    canonical: str
    attributes: tuple[str, ...]
    bbox_norm: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    present: bool


class SceneMemoryStore(Protocol):
    @property
    def path(self) -> Path: ...

    def load(self) -> tuple[SceneObject, ...]: ...

    def save(self, objects: tuple[SceneObject, ...]) -> None: ...

    def update(self, objects: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]: ...

    def compact_line(self) -> str: ...


__all__ = ["SceneMemoryStore", "SceneObject"]
