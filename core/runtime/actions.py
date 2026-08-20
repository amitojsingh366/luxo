"""Route executed plan actions to the body and to outbound browser effects.

The owning runtime must tick exactly this order at 10 Hz::

    BehaviorFSM.tick() -> ConversationCoordinator.tick() -> ActionRouter.tick()

The router replaces the bare ``PlanExecutor.tick()`` at the end of that chain.
It owns the only path from the semantic plan to the body: one executor tick
emits at most one :class:`Action`, that action goes to
``AnimationDirector.apply_action``, and the director's drained effects become
outbound messages. Nothing here interprets a verb, invents timing, or touches a
joint; the director remains the sole authority on how a verb moves the lamp
(PRD 6.3).

The router is also the only caller of ``AnimationDirector.drain_effects``. The
director accumulates effects on its own 120 Hz tick, which this module never
drives, so a cue raised by the animation loop is dispatched on the next 10 Hz
router tick.

Blockers stay exactly where the executor put them. ``wait`` holds for its
duration and ``observe`` holds until a release. Finite gestures are body-owned
rather than executor-owned, so the router also withholds the next executor tick
while the director reports that a gesture is queued or animating. This keeps
multi-gesture plans sequential and prevents an empty queue from ending ACTING
before the final authored release.

The observation flow itself is not implemented here. ``capture_callback`` is
the seam: the router hands out one :class:`CaptureFrameMessage` per accepted
``observe`` and the owner of that callback round-trips the frame and calls
:meth:`ActionRouter.release_observation` with the same ``req_id``. The id is
the generation tag; ``PlanExecutor`` issues each one once, so a release that
arrives after a cancel is recognised as stale rather than believed.

Two independent guarantees keep a cancelled interaction from stranding half of
itself:

* A dormant or disengaging transition clears the plan in a ``finally``, so a
  director that rejects the transition cannot leave the plan blocker live.
* A ``CaptureRequest`` is dispatched only while its own id is the executor's
  pending observation, so a director capture can never outlive its blocker and
  reach the browser. This also covers ``ConversationCoordinator.disengage``,
  which clears the same executor directly without consulting the director.

Outbound callbacks must be quick, non-blocking enqueues and must never call
back into the router: they run while the router lock is held. Lock order is
router lock, then executor lock. The router never touches the blackboard; the
coordinator mirrors plan depth immediately before each router tick.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Final, get_args

from ..animation.director import AnimationDirector, CaptureRequest, SfxCue
from ..blackboard import BlackboardSnapshot
from ..brain.schema import Action, ActionOp, SfxName
from ..fsm import BehaviorState, Transition
from ..plan_executor import ObservationReleaseError, PlanExecutor
from ..protocol.messages import CaptureFrameMessage, CueMessage
from ..protocol.messages import SfxName as WireSfxName

LOGGER = logging.getLogger(__name__)

ROUTED_OPS: Final = frozenset(ActionOp)
"""The closed action vocabulary (PRD 6.2). Anything else is dropped."""

CUED_SFX: Final = frozenset(get_args(WireSfxName))
"""The closed sound vocabulary (PRD 9.2), taken from the wire schema."""

CANCELLING_STATES: Final = frozenset(
    {BehaviorState.DORMANT, BehaviorState.DISENGAGING}
)
"""States that must clear the plan, its blocker, and the body together."""

CueCallback = Callable[[CueMessage], None]
CaptureCallback = Callable[[CaptureFrameMessage], None]


@dataclass(frozen=True, slots=True)
class RouterStatus:
    """Immutable routing view carrying no action, effect, or plan payload."""

    routed: int
    dropped: int
    cues: int
    captures: int
    plan_depth: int
    pending_observation_id: str | None
    blocked_on_wait: bool
    blocked_on_observation: bool


class ActionRouter:
    """Drive one plan action per tick into the body and out to the browser.

    Every public method is safe to call from socket, worker, and tick threads.
    Counters are cumulative for the router's lifetime and exist for telemetry
    and checks; they never influence routing.
    """

    def __init__(
        self,
        *,
        director: AnimationDirector,
        plan_executor: PlanExecutor,
        cue_callback: CueCallback,
        capture_callback: CaptureCallback,
    ) -> None:
        if not isinstance(director, AnimationDirector):
            raise TypeError("director must be an AnimationDirector")
        if not isinstance(plan_executor, PlanExecutor):
            raise TypeError("plan_executor must be a PlanExecutor")
        if not callable(cue_callback) or not callable(capture_callback):
            raise TypeError("router callbacks must be callable")
        self._director = director
        self._plans = plan_executor
        self._cue_callback = cue_callback
        self._capture_callback = capture_callback
        self._lock = RLock()
        self._routed = 0
        self._dropped = 0
        self._cues = 0
        self._captures = 0

    @property
    def status(self) -> RouterStatus:
        with self._lock:
            state = self._plans.state
            return RouterStatus(
                routed=self._routed,
                dropped=self._dropped,
                cues=self._cues,
                captures=self._captures,
                plan_depth=state.depth,
                pending_observation_id=state.pending_observation_id,
                blocked_on_wait=state.wait_deadline is not None,
                blocked_on_observation=state.pending_observation_id is not None,
            )

    def tick(self, snapshot: BlackboardSnapshot, now: float) -> Action | None:
        """Route at most one executed action and dispatch drained effects.

        Returns the action that reached the body, or ``None`` when the plan is
        empty, blocked, or the emitted action was dropped as unroutable.
        Effects are dispatched even when the executor rejects the timestamp,
        so a caller error cannot silently strand a queued cue or capture.
        """

        with self._lock:
            try:
                if self._director.gesture_in_progress:
                    return None
                action = self._plans.tick(snapshot, now)
                return None if action is None else self._route_locked(action)
            finally:
                self._dispatch_locked()

    def apply_transition(self, transition: Transition) -> bool:
        """Apply one state beat, cancelling the plan on dormant or disengage.

        The plan clear runs in a ``finally`` so that a rejected transition
        cannot leave the queue blocked on an observation the director has
        already forgotten.
        """

        cancelling = (
            isinstance(transition, Transition)
            and transition.current in CANCELLING_STATES
        )
        with self._lock:
            try:
                return self._director.apply_transition(transition)
            finally:
                if cancelling:
                    self._cancel_locked(_reason(transition))
                self._dispatch_locked()

    def release_observation(self, request_id: str) -> bool:
        """Release exactly the blocker this observation belongs to.

        This is the seam the observation flow calls back on. A release that
        lost the race with a cancel is a normal outcome, not an error, so it
        reports false instead of raising.
        """

        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        with self._lock:
            try:
                self._plans.release_observation(request_id)
            except ObservationReleaseError as error:
                self._dropped += 1
                LOGGER.warning("ignoring stale observation release: %s", error)
                return False
            self._dispatch_locked()
            return True

    def cancel(self, reason: str = "cancelled") -> None:
        """Drop the plan and its blocker, and withhold an orphaned capture."""

        with self._lock:
            self._cancel_locked(reason)
            self._dispatch_locked()

    def reset(self) -> None:
        """Clear the plan and return the body to its canonical rest state."""

        with self._lock:
            self._cancel_locked("reset")
            self._director.reset()

    def _route_locked(self, action: Action) -> Action | None:
        op = getattr(action, "op", None)
        if not isinstance(action, Action) or not isinstance(op, ActionOp) or op not in ROUTED_OPS:
            self._dropped += 1
            LOGGER.warning("dropping action outside the closed vocabulary: %r", op)
            return None
        if op is ActionOp.OBSERVE:
            return self._route_observe_locked(action)
        self._director.apply_action(action)
        self._routed += 1
        return action

    def _route_observe_locked(self, action: Action) -> Action | None:
        request_id = self._plans.pending_observation_id
        if request_id is None:
            # A concurrent cancel won the race with this tick, so the capture
            # would have no blocker left to release.
            self._dropped += 1
            LOGGER.warning("dropping observe whose blocker was already cleared")
            return None
        self._director.apply_action(action, observation_id=request_id)
        self._routed += 1
        return action

    def _dispatch_locked(self) -> None:
        for effect in self._director.drain_effects():
            if isinstance(effect, SfxCue):
                self._dispatch_cue_locked(effect)
            elif isinstance(effect, CaptureRequest):
                self._dispatch_capture_locked(effect)
            else:  # pragma: no cover - the director owns the effect union
                self._dropped += 1
                LOGGER.warning("dropping unknown director effect: %r", effect)

    def _dispatch_cue_locked(self, cue: SfxCue) -> None:
        name = cue.name
        if not isinstance(name, SfxName) or name.value not in CUED_SFX:
            self._dropped += 1
            LOGGER.warning("dropping sfx outside the closed vocabulary: %r", name)
            return
        try:
            self._cue_callback(CueMessage(name.value))
        except Exception:
            # A lost cue is cosmetic: it must never interrupt the plan.
            self._dropped += 1
            LOGGER.exception("cue dispatch failed for %s", name.value)
            return
        self._cues += 1

    def _dispatch_capture_locked(self, capture: CaptureRequest) -> None:
        request_id = getattr(capture, "request_id", None)
        pending = self._plans.pending_observation_id
        if not isinstance(request_id, str) or request_id != pending:
            self._dropped += 1
            LOGGER.warning(
                "dropping capture %r with no matching blocker %r", request_id, pending
            )
            return
        try:
            self._capture_callback(CaptureFrameMessage(request_id))
        except Exception:
            # This frame will never come back, so the blocker it belongs to
            # would hold the queue forever. Cancel instead of stranding it.
            self._dropped += 1
            LOGGER.exception("capture dispatch failed for %s", request_id)
            self._cancel_locked("capture_dispatch_failed")
            return
        self._captures += 1

    def _cancel_locked(self, reason: str) -> None:
        """Clear the plan and its blocker; both clears are total."""

        self._plans.clear()
        LOGGER.info("plan routing cancelled: %s", reason)


def _reason(transition: Transition) -> str:
    reason = getattr(transition, "reason", None)
    return reason if isinstance(reason, str) and reason else "transition"


__all__ = [
    "ActionRouter",
    "CANCELLING_STATES",
    "CUED_SFX",
    "CaptureCallback",
    "CueCallback",
    "ROUTED_OPS",
    "RouterStatus",
]
