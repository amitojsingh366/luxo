"""Thread-safe execution of validated semantic plans.

The executor deliberately preserves the model/body boundary: it sequences
validated :class:`Action` values but never interprets them as joint commands or
animation timing. Integration code dispatches each emitted semantic action to
the body module that owns its implementation.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from .blackboard import BlackboardSnapshot
from .brain.schema import Action, ActionOp

ObservationIdFactory = Callable[[int], str]
_OBSERVATION_ID = re.compile(r"obs_[1-9][0-9]*\Z")


class ObservationReleaseError(RuntimeError):
    """An observation completion does not match the active request."""


@dataclass(frozen=True, slots=True)
class ExecutorState:
    """Immutable executor state for FSM and telemetry integration."""

    queued_actions: tuple[Action, ...]
    active_blocker: Action | None
    pending_observation_id: str | None
    wait_deadline: float | None
    last_tick_time: float | None

    @property
    def depth(self) -> int:
        """Count actions still queued or actively blocking the plan."""

        return len(self.queued_actions) + int(self.active_blocker is not None)


class PlanExecutor:
    """FIFO executor for the schema-owned, closed semantic vocabulary.

    ``submit`` appends atomically. ``tick`` emits at most one action and never
    performs I/O. Wait and observation actions remain counted in ``depth``
    while active so the FSM cannot mistake a blocked plan for a drained one.
    """

    def __init__(
        self,
        observation_id_factory: ObservationIdFactory | None = None,
    ) -> None:
        if observation_id_factory is not None and not callable(observation_id_factory):
            raise TypeError("observation_id_factory must be callable")
        self._lock = RLock()
        self._queue: deque[Action] = deque()
        self._active_blocker: Action | None = None
        self._pending_observation_id: str | None = None
        self._wait_deadline: float | None = None
        self._last_tick_time: float | None = None
        self._next_observation_sequence = 1
        self._issued_observation_ids: set[str] = set()
        self._observation_id_factory = observation_id_factory or _default_observation_id

    @property
    def state(self) -> ExecutorState:
        """Return a consistent immutable view without exposing the deque."""

        with self._lock:
            return ExecutorState(
                queued_actions=tuple(self._queue),
                active_blocker=self._active_blocker,
                pending_observation_id=self._pending_observation_id,
                wait_deadline=self._wait_deadline,
                last_tick_time=self._last_tick_time,
            )

    @property
    def queued_actions(self) -> tuple[Action, ...]:
        return self.state.queued_actions

    @property
    def depth(self) -> int:
        return self.state.depth

    @property
    def blocked_on_observation(self) -> bool:
        with self._lock:
            return self._pending_observation_id is not None

    @property
    def blocked_on_wait(self) -> bool:
        with self._lock:
            return self._wait_deadline is not None

    @property
    def pending_observation_id(self) -> str | None:
        with self._lock:
            return self._pending_observation_id

    def submit(self, plan: Sequence[Action]) -> None:
        """Append a validated plan atomically, including while blocked."""

        if not isinstance(plan, Sequence):
            raise TypeError("plan must be a sequence of Action values")
        actions = tuple(plan)
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("plan must contain only validated Action values")
        with self._lock:
            self._queue.extend(actions)

    def clear(self) -> None:
        """Atomically cancel queued actions and either kind of blocker."""

        with self._lock:
            self._queue.clear()
            self._active_blocker = None
            self._pending_observation_id = None
            self._wait_deadline = None

    def tick(self, snapshot: BlackboardSnapshot, now: float) -> Action | None:
        """Advance one deterministic step, emitting at most one action.

        Time must be finite and non-decreasing. A rejected backward timestamp
        cannot alter or release a blocker. Reaching a wait deadline consumes
        that blocker on this tick; the next action is deliberately deferred to
        a later tick.
        """

        if not isinstance(snapshot, BlackboardSnapshot):
            raise TypeError("snapshot must be a BlackboardSnapshot")
        tick_time = _finite_time(now)

        with self._lock:
            if self._last_tick_time is not None and tick_time < self._last_tick_time:
                raise ValueError("now must be monotonic and cannot move backward")
            self._last_tick_time = tick_time

            if self._active_blocker is not None:
                if self._active_blocker.op is ActionOp.OBSERVE:
                    return None
                if tick_time < self._required_wait_deadline():
                    return None
                self._active_blocker = None
                self._wait_deadline = None
                return None

            if not self._queue:
                return None

            action = self._queue[0]
            if action.op is ActionOp.OBSERVE:
                request_id = self._make_observation_id()
                self._queue.popleft()
                self._active_blocker = action
                self._pending_observation_id = request_id
                return action

            if action.op is ActionOp.WAIT:
                duration_ms = action.ms
                if duration_ms is None:  # pragma: no cover - schema invariant
                    raise RuntimeError("validated wait action has no duration")
                duration = duration_ms / 1000.0
                deadline = tick_time + duration
                if not math.isfinite(deadline) or (duration > 0.0 and deadline <= tick_time):
                    raise ValueError("now is too large to represent the wait duration")
                self._queue.popleft()
                self._active_blocker = action
                self._wait_deadline = deadline
                return action
            self._queue.popleft()
            return action

    def release_observation(self, request_id: str) -> None:
        """Release exactly the pending observation, or fail without mutation."""

        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        with self._lock:
            pending = self._pending_observation_id
            if pending is None:
                raise ObservationReleaseError("no observation request is pending")
            if request_id != pending:
                raise ObservationReleaseError(
                    f"observation request {request_id!r} does not match pending {pending!r}"
                )
            self._pending_observation_id = None
            self._active_blocker = None

    def _required_wait_deadline(self) -> float:
        if self._wait_deadline is None:  # pragma: no cover - internal invariant
            raise RuntimeError("wait blocker has no deadline")
        return self._wait_deadline

    def _make_observation_id(self) -> str:
        sequence = self._next_observation_sequence
        request_id = self._observation_id_factory(sequence)
        if not isinstance(request_id, str) or not _OBSERVATION_ID.fullmatch(request_id):
            raise ValueError("observation id factory must return obs_<positive integer>")
        if request_id in self._issued_observation_ids:
            raise ValueError(f"observation id factory returned duplicate {request_id!r}")
        self._next_observation_sequence += 1
        self._issued_observation_ids.add(request_id)
        return request_id


def _default_observation_id(sequence: int) -> str:
    return f"obs_{sequence}"


def _finite_time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("now must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("now must be a finite number")
    return parsed


__all__ = [
    "ExecutorState",
    "ObservationIdFactory",
    "ObservationReleaseError",
    "PlanExecutor",
]
