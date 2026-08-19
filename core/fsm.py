"""Thread-safe 10 Hz behavior state machine for Luxo.

Workers post completed facts as closed :class:`BehaviorEvent` values. The FSM
consumes them at a tick boundary and assigns intent by changing state; it never
performs I/O, invokes a worker, or executes a plan.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import RLock

from .blackboard import BlackboardSnapshot, GazeFact


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


class BehaviorEvent(str, Enum):
    """Closed worker-to-FSM event vocabulary.

    ``BROWSER_READY`` means the browser sent ``hello`` and camera permission
    was granted. Result events carry no payload because their data belongs on
    the blackboard; they only announce that an atomic result is ready.
    """

    MODELS_WARM = "models_warm"
    BROWSER_READY = "browser_ready"
    VAD_START = "vad_start"
    TRANSCRIPT_READY = "transcript_ready"
    MODEL_RESPONSE = "model_response"
    MODEL_FALLBACK = "model_fallback"
    SPEECH_DONE = "speech_done"
    OBSERVE_START = "observe_start"
    OBSERVE_COMPLETE = "observe_complete"
    PLAN_DRAINED = "plan_drained"


@dataclass(frozen=True, slots=True)
class Transition:
    t: float
    previous: BehaviorState
    current: BehaviorState
    reason: str


@dataclass(frozen=True, slots=True)
class FSMStatus:
    """Immutable state suitable for telemetry and deterministic checks."""

    state: BehaviorState
    state_entered_at: float | None
    last_tick_at: float | None
    last_transition: Transition | None
    transition_count: int
    pending_events: int
    dropped_events: int
    models_warm: bool
    browser_ready: bool
    gaze_on: bool
    gaze_on_since: float | None
    gaze_off_since: float | None


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    sequence: int
    kind: BehaviorEvent
    posted_state: BehaviorState


_ENGAGED_STATES = frozenset(
    {
        BehaviorState.ENGAGED,
        BehaviorState.LISTENING,
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.INSPECTING,
        BehaviorState.ACTING,
    }
)


class BehaviorFSM:
    """Deterministic behavior FSM intended to be ticked at 10 Hz.

    Posting may happen from arbitrary worker threads. A tick consumes a stable
    FIFO batch and performs at most one transition. Events remember the state
    in which they were posted, preventing a late one-shot result from firing
    after the character has moved into an unrelated state.
    """

    GAZE_YAW_DEG = 25.0
    GAZE_PITCH_DEG = 20.0
    GAZE_MIN_CONFIDENCE = 0.5
    GAZE_STALE_S = 1.0
    ENGAGE_DWELL_S = 0.50
    NOTICE_HOLD_S = 0.30
    DISENGAGE_DWELL_S = 2.50
    DISENGAGE_DROOP_S = 1.50

    def __init__(self) -> None:
        self._lock = RLock()
        self._state = BehaviorState.BOOT
        self._state_entered_at: float | None = None
        self._last_tick_at: float | None = None
        self._last_transition: Transition | None = None
        self._transition_count = 0
        self._models_warm = False
        self._browser_ready = False
        self._gaze_on = False
        self._gaze_on_since: float | None = None
        self._gaze_off_since: float | None = None
        self._events: deque[_QueuedEvent] = deque()
        self._next_event_sequence = 0
        self._dropped_events = 0

    @property
    def state(self) -> BehaviorState:
        with self._lock:
            return self._state

    @property
    def last_transition(self) -> Transition | None:
        with self._lock:
            return self._last_transition

    @property
    def pending_event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def status(self) -> FSMStatus:
        with self._lock:
            return FSMStatus(
                state=self._state,
                state_entered_at=self._state_entered_at,
                last_tick_at=self._last_tick_at,
                last_transition=self._last_transition,
                transition_count=self._transition_count,
                pending_events=len(self._events),
                dropped_events=self._dropped_events,
                models_warm=self._models_warm,
                browser_ready=self._browser_ready,
                gaze_on=self._gaze_on,
                gaze_on_since=self._gaze_on_since,
                gaze_off_since=self._gaze_off_since,
            )

    def post_event(self, event: BehaviorEvent) -> int:
        """Queue one closed event and return its increasing sequence id."""

        if not isinstance(event, BehaviorEvent):
            raise TypeError("event must be a BehaviorEvent")
        with self._lock:
            sequence = self._next_event_sequence
            self._next_event_sequence += 1
            self._events.append(_QueuedEvent(sequence, event, self._state))
            return sequence

    def tick(
        self,
        snapshot: BlackboardSnapshot,
        now: float,
    ) -> Transition | None:
        """Consume facts and queued events without blocking worker I/O.

        Gaze-driven disengagement is evaluated before worker events. Thus a
        completed 2.50 s gaze-loss dwell wins if, for example, a model response
        arrives on the same tick. This enforces the PRD transition from any
        engaged state and deliberately discards the displaced event.
        """

        if not isinstance(snapshot, BlackboardSnapshot):
            raise TypeError("snapshot must be a BlackboardSnapshot")
        instant = _finite_time(now)

        with self._lock:
            if self._last_tick_at is not None and instant < self._last_tick_at:
                raise ValueError("now must not move backwards")

            event_batch = tuple(self._events)
            self._events.clear()
            self._gaze_on = gaze_is_on(snapshot.gaze, instant)
            self._update_gaze_dwell(instant)

            transition = self._gaze_transition(instant)
            if transition is not None:
                self._dropped_events += len(event_batch)
            else:
                transition = self._event_transition(event_batch, snapshot, instant)

            self._last_tick_at = instant
            if self._state_entered_at is None:
                self._state_entered_at = instant
            return transition

    def _update_gaze_dwell(self, now: float) -> None:
        if self._gaze_on:
            if self._gaze_on_since is None:
                self._gaze_on_since = now
            self._gaze_off_since = None
        else:
            if self._gaze_off_since is None:
                self._gaze_off_since = now
            self._gaze_on_since = None

    def _gaze_transition(self, now: float) -> Transition | None:
        if self._state is BehaviorState.DORMANT:
            if self._gaze_on and _at_least(
                _elapsed(now, self._gaze_on_since), self.ENGAGE_DWELL_S
            ):
                return self._transition(BehaviorState.NOTICING, now, "gaze_on_dwell")

        elif self._state is BehaviorState.NOTICING:
            if not self._gaze_on:
                return self._transition(
                    BehaviorState.DORMANT, now, "gaze_lost_during_notice"
                )
            if _at_least(
                _elapsed(now, self._state_entered_at), self.NOTICE_HOLD_S
            ):
                return self._transition(BehaviorState.ENGAGED, now, "notice_hold")

        elif self._state in _ENGAGED_STATES:
            if (
                not self._gaze_on
                and _at_least(
                    _elapsed(now, self._gaze_off_since), self.DISENGAGE_DWELL_S
                )
            ):
                return self._transition(
                    BehaviorState.DISENGAGING, now, "gaze_off_dwell"
                )

        elif self._state is BehaviorState.DISENGAGING:
            if self._gaze_on:
                return self._transition(
                    BehaviorState.ENGAGED, now, "gaze_returned_mid_droop"
                )
            if _at_least(
                _elapsed(now, self._state_entered_at), self.DISENGAGE_DROOP_S
            ):
                return self._transition(
                    BehaviorState.DORMANT, now, "disengage_droop_complete"
                )
        return None

    def _event_transition(
        self,
        events: tuple[_QueuedEvent, ...],
        snapshot: BlackboardSnapshot,
        now: float,
    ) -> Transition | None:
        for index, queued in enumerate(events):
            if queued.posted_state is not self._state:
                self._dropped_events += 1
                continue

            if self._state is BehaviorState.BOOT:
                if queued.kind is BehaviorEvent.MODELS_WARM:
                    self._models_warm = True
                    continue
                if queued.kind is BehaviorEvent.BROWSER_READY:
                    self._browser_ready = True
                    continue

            target_reason = self._target_for_event(queued.kind, snapshot)
            if target_reason is None:
                self._dropped_events += 1
                continue

            target, reason = target_reason
            self._dropped_events += len(events) - index - 1
            return self._transition(target, now, reason)

        if (
            self._state is BehaviorState.BOOT
            and self._models_warm
            and self._browser_ready
        ):
            return self._transition(BehaviorState.DORMANT, now, "startup_ready")
        return None

    def _target_for_event(
        self,
        event: BehaviorEvent,
        snapshot: BlackboardSnapshot,
    ) -> tuple[BehaviorState, str] | None:
        if self._state is BehaviorState.ENGAGED and event is BehaviorEvent.VAD_START:
            return BehaviorState.LISTENING, "vad_start"

        elif (
            self._state is BehaviorState.LISTENING
            and event is BehaviorEvent.TRANSCRIPT_READY
        ):
            return BehaviorState.THINKING, "transcript_ready"

        elif self._state is BehaviorState.THINKING:
            if event is BehaviorEvent.MODEL_RESPONSE:
                return BehaviorState.SPEAKING, "model_response"
            if event is BehaviorEvent.MODEL_FALLBACK:
                return BehaviorState.SPEAKING, "model_fallback"

        elif self._state is BehaviorState.SPEAKING and event is BehaviorEvent.SPEECH_DONE:
            if snapshot.plan_queue:
                return BehaviorState.ACTING, "speech_done_with_plan"
            return BehaviorState.ENGAGED, "speech_done_without_plan"

        elif self._state is BehaviorState.ACTING:
            if event is BehaviorEvent.OBSERVE_START:
                return BehaviorState.INSPECTING, "observe_start"
            if event is BehaviorEvent.PLAN_DRAINED and not snapshot.plan_queue:
                return BehaviorState.ENGAGED, "plan_drained"

        elif (
            self._state is BehaviorState.INSPECTING
            and event is BehaviorEvent.OBSERVE_COMPLETE
        ):
            return BehaviorState.ACTING, "observe_complete"
        return None

    def _transition(
        self,
        target: BehaviorState,
        now: float,
        reason: str,
    ) -> Transition:
        previous = self._state
        transition = Transition(now, previous, target, reason)
        self._state = target
        self._state_entered_at = now
        self._last_transition = transition
        self._transition_count += 1

        if target is BehaviorState.DORMANT:
            self._gaze_on_since = None
            self._gaze_off_since = None
        elif previous is BehaviorState.DISENGAGING and target is BehaviorState.ENGAGED:
            self._gaze_on_since = now
            self._gaze_off_since = None
        elif target is BehaviorState.DISENGAGING:
            self._gaze_off_since = None
        return transition


def gaze_is_on(fact: GazeFact, now: float) -> bool:
    """Return the PRD gaze predicate; malformed, future, or stale facts are off."""

    if not isinstance(fact, GazeFact) or fact.present is not True:
        return False
    instant = _finite_time(now)
    values = (fact.t, fact.yaw_deg, fact.pitch_deg, fact.conf)
    if not all(_is_finite_number(value) for value in values):
        return False
    age = instant - float(fact.t)
    return (
        0.0 <= age <= BehaviorFSM.GAZE_STALE_S
        and float(fact.conf) >= BehaviorFSM.GAZE_MIN_CONFIDENCE
        and abs(float(fact.yaw_deg)) <= BehaviorFSM.GAZE_YAW_DEG
        and abs(float(fact.pitch_deg)) <= BehaviorFSM.GAZE_PITCH_DEG
    )


def _finite_time(value: float) -> float:
    if not _is_finite_number(value):
        raise TypeError("now must be a finite real number")
    return float(value)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _elapsed(now: float, since: float | None) -> float:
    return -math.inf if since is None else now - since


def _at_least(elapsed: float, threshold: float) -> bool:
    return elapsed >= threshold or math.isclose(
        elapsed, threshold, rel_tol=0.0, abs_tol=1e-12
    )


__all__ = [
    "BehaviorEvent",
    "BehaviorFSM",
    "BehaviorState",
    "FSMStatus",
    "Transition",
    "gaze_is_on",
]
