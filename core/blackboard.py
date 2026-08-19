"""Thread-safe state exchange for the character core.

Worker threads publish completed facts and results here. In particular, sensor
records contain measurements only: they do not contain engagement decisions,
gesture requests, light choices, or any other intent owned by the FSM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from threading import RLock

from .brain.memory import SceneObject
from .brain.schema import Action


@dataclass(frozen=True, slots=True)
class GazeFact:
    t: float
    present: bool
    yaw_deg: float
    pitch_deg: float
    az: float
    el: float
    conf: float


@dataclass(frozen=True, slots=True)
class UtteranceFact:
    t: float
    text: str


@dataclass(frozen=True, slots=True)
class Goal:
    t: float
    text: str


@dataclass(frozen=True, slots=True)
class ClampCounts:
    velocity: int = 0
    limit: int = 0


@dataclass(frozen=True, slots=True)
class Telemetry:
    state: str = "BOOT"
    plan_depth: int = 0
    memory_count: int = 0
    last_latency_ms: float = 0.0
    clamps: ClampCounts = ClampCounts()


@dataclass(frozen=True, slots=True)
class BlackboardSnapshot:
    revision: int
    gaze: GazeFact
    utterance: UtteranceFact | None
    goal: Goal | None
    scene_memory: tuple[SceneObject, ...]
    plan_queue: tuple[Action, ...]
    telemetry: Telemetry


_ABSENT_GAZE = GazeFact(
    t=0.0,
    present=False,
    yaw_deg=0.0,
    pitch_deg=0.0,
    az=0.0,
    el=0.0,
    conf=0.0,
)


class Blackboard:
    """A locked mutable store that exposes only immutable typed snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 0
        self._gaze = _ABSENT_GAZE
        self._utterance: UtteranceFact | None = None
        self._goal: Goal | None = None
        self._scene_memory: tuple[SceneObject, ...] = ()
        self._plan_queue: tuple[Action, ...] = ()
        self._telemetry = Telemetry()

    @property
    def lock(self) -> RLock:
        """Expose the shared lock for an atomic multi-field transaction."""

        return self._lock

    def snapshot(self) -> BlackboardSnapshot:
        with self._lock:
            return BlackboardSnapshot(
                revision=self._revision,
                gaze=self._gaze,
                utterance=self._utterance,
                goal=self._goal,
                scene_memory=self._scene_memory,
                plan_queue=self._plan_queue,
                telemetry=self._telemetry,
            )

    def publish_gaze(self, fact: GazeFact) -> None:
        with self._lock:
            self._gaze = fact
            self._changed()

    def publish_utterance(self, fact: UtteranceFact | None) -> None:
        with self._lock:
            self._utterance = fact
            self._changed()

    def set_goal(self, goal: Goal | None) -> None:
        with self._lock:
            self._goal = goal
            self._changed()

    def set_scene_memory(self, objects: Iterable[SceneObject]) -> None:
        with self._lock:
            self._scene_memory = tuple(objects)
            self._changed()

    def set_plan(self, actions: Iterable[Action]) -> None:
        with self._lock:
            self._plan_queue = tuple(actions)
            self._changed()

    def pop_plan_action(self) -> Action | None:
        with self._lock:
            if not self._plan_queue:
                return None
            action = self._plan_queue[0]
            self._plan_queue = self._plan_queue[1:]
            self._changed()
            return action

    def set_telemetry(self, telemetry: Telemetry) -> None:
        with self._lock:
            self._telemetry = telemetry
            self._changed()

    def reset(self) -> None:
        """Reset volatile demo state while preserving no hidden references."""

        with self._lock:
            self._gaze = _ABSENT_GAZE
            self._utterance = None
            self._goal = None
            self._scene_memory = ()
            self._plan_queue = ()
            self._telemetry = Telemetry()
            self._changed()

    def _changed(self) -> None:
        self._revision += 1


SENSOR_FACT_FIELDS = {
    GazeFact: frozenset(field.name for field in fields(GazeFact)),
    UtteranceFact: frozenset(field.name for field in fields(UtteranceFact)),
}


__all__ = [
    "Blackboard",
    "BlackboardSnapshot",
    "ClampCounts",
    "GazeFact",
    "Goal",
    "SENSOR_FACT_FIELDS",
    "Telemetry",
    "UtteranceFact",
]
