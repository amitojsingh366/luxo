"""Between-takes demo reset: one baseline, cleared in one behaviour tick.

PRD 13.1 asks for a hotkey that resets the FSM and clears memory *between*
takes, and says in the same breath that it is a dev tool and is never used
mid-take. That sentence fixes both halves of this module's design.

Because it is a dev tool, it is **not** a protocol command. No message type is
added, ``schema/messages.schema.json`` is untouched, and the browser is neither
asked for anything nor told anything: the reset is requested out of band, on
the process, and is applied entirely inside the core. The operator's terminal
already has the core's pid; the browser tab is left exactly as it is, still
connected, still holding its camera and microphone permission.

Because it must never fire mid-take, the request path is deliberately inert. A
request is one atomic append to a length-1 deque — no lock, no allocation of
consequence, no work — so it is safe from a POSIX signal handler running
between bytecodes on the main thread. Everything that actually changes state
happens in :meth:`DemoReset.drain`, which the owning runtime calls at the top
of its serialized 10 Hz beat, exactly as ``ConversationCoordinator`` drains
worker completions rather than applying them from the worker.

What "atomic" means here
------------------------

Not one global lock. The runtime deliberately has no such lock, and taking one
across the director, the router, the plan executor, and three staging
components would invert the documented order in ``app.py`` and deadlock against
the animation tick. Atomic here means *no participant can observe a half-reset
character*, which four separate properties deliver together:

* **One tick owns the whole sequence.** The behaviour tick is the only thread
  that advances a stage. Draining at the top of that tick means no other tick
  can interleave, and the tick that drains then runs to completion on the fresh
  baseline rather than on a mixture.
* **Each clear is covered by its own component's lock.** Every step below is a
  method the component already exposes for reconnect recovery, so each one is
  internally atomic under its own lock.
* **Generations make the in-flight work inert, not merely ignored.** Every
  staging component bumps a generation counter as it clears. A superseded STT,
  brain, speech, capture, or narration worker that completes afterwards finds
  its token stale and enqueues nothing; a superseded speech delivery raises out
  of its own send path instead of writing to the socket.
* **The FSM is reset first, which closes every inbound door.** ``on_vad_start``
  requires ENGAGED, ``accept_utterance`` requires LISTENING, ``on_tts_done``
  requires SPEAKING, and a capture is only armed from a plan. Once the state is
  DORMANT, a frame or a browser event arriving on the socket thread while the
  remaining steps run cannot re-arm anything that has already been cleared.

The step order is therefore load-bearing and is asserted by the checks: FSM,
conversation, narration, observation, body, blackboard, then the runtime's own
arming state. Each step is guarded independently and a failure is logged and
counted rather than allowed to abandon the steps after it, for the same reason
``app.py`` guards the steps of a cancelling transition: a director that rejects
a beat must not be able to strand the plan queue.

What survives, and why
----------------------

The whole point is to not pay warm-up cost between takes, so nothing here
touches whisper, Piper, the brain client, or ``WakeSequenceCoordinator``. That
last one matters more than it looks: the coordinator remembers that it has
already posted ``MODELS_WARM`` and ``BROWSER_READY``, and it never posts them
twice. A reset that returned the FSM to BOOT would therefore wait for two
events nobody will ever send again, and the character would be dead until the
process was restarted. DORMANT, with ``_models_warm`` and ``_browser_ready``
left standing, is the only baseline that is both truthful and reachable.

Browser readiness survives for the same reason: the reset publishes no message
at all, so the socket, the page, and its camera and microphone permission are
untouched. The one browser-derived fact that *is* cleared is the gaze record,
which is a volatile measurement the renderer republishes within a frame. The
lamp settles to rest and then notices the presenter again on the ordinary 0.50 s
dwell, which is exactly the beat a take wants to open on.

Scene memory
------------

PRD 8.2 memory is cleared on both sides: the blackboard mirror the behaviour
domain reads, and the durable JSON the store merges into on the next
``observe``. Clearing only the mirror would be worse than not clearing at all,
because ``SceneMemoryStore.update`` loads the file and merges, so every object
from the previous take would walk back into the next one.

The durable half is the single piece of this reset that does not run on the
tick, because a scene-memory write is ``fsync``-backed disk I/O and app.py's
contract puts that on a worker pool. It is submitted, not awaited. Ordering is
still safe: ``ObservationRuntime.reset`` runs first and cancels the pending
request through ``ObservationCoordinator``, whose lock is held for the whole of
``complete()`` — so by the time the reset's own step returns, any in-flight
scene-memory write has already finished, and the submitted clear can only land
after it. The residual window is narrow and documented: if a *new* observation
were to complete in the sub-millisecond gap before the clear lands, it would
merge against the old file. Nothing can be stranded by that; at worst one stale
label survives into the next take.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

from ..blackboard import Blackboard
from ..brain.memory import SceneMemoryStore
from ..fsm import BehaviorFSM, BehaviorState

LOGGER = logging.getLogger(__name__)

RESET_STATE: Final = BehaviorState.DORMANT
"""The resting baseline a reset returns to; never BOOT (see the module note)."""

DEFAULT_REASON: Final = "reset"

STEP_NAMES: Final = (
    "fsm",
    "conversation",
    "narration",
    "observation",
    "body",
    "blackboard",
    "runtime",
)
"""The fixed order of the reset, published so checks can pin it."""

_FSM_FIELDS: Final = (
    "_lock",
    "_state",
    "_state_entered_at",
    "_last_transition",
    "_gaze_on",
    "_gaze_on_since",
    "_gaze_off_since",
    "_events",
    "_dropped_events",
)
"""``BehaviorFSM`` internals this module writes directly.

``core/fsm.py`` exposes no path from an engaged state to DORMANT: every
transition it performs is earned by gaze dwell or by a closed worker event, and
the only route to DORMANT from an engaged state costs 2.50 s of gaze loss plus
a 1.50 s droop. Waiting four seconds is not a reset, and a reset that half
happens now and half happens later is precisely the stranded state this packet
exists to remove, so the baseline is written in one hold of the FSM's own lock.

Reconstructing the FSM instead is not an option: the conversation coordinator,
the observation runtime, and the wake sequence all captured the instance at
construction, and a replacement would leave them posting into an object nobody
reads. The names are checked at construction so that a future change to
``core/fsm.py`` fails loudly here instead of silently resetting nothing.
"""


@dataclass(frozen=True, slots=True)
class ResetReport:
    """Immutable record of one applied reset; carries no transcript or plan."""

    sequence: int
    t: float
    reason: str
    previous_state: BehaviorState
    failed_steps: tuple[str, ...]
    memory_clear_submitted: bool

    @property
    def clean(self) -> bool:
        return not self.failed_steps and self.memory_clear_submitted


@dataclass(frozen=True, slots=True)
class ResetStatus:
    """Immutable diagnostic view for telemetry and checks."""

    pending: bool
    applied: int
    failed_steps: tuple[str, ...]
    last_reason: str | None
    memory_clears_committed: int
    memory_clears_failed: int
    closed: bool


class DemoReset:
    """Coalesce out-of-band reset requests and apply one per behaviour tick.

    Every collaborator is injected, including the director lock, because the
    reset must take exactly the same lock the owning runtime takes around every
    entry into the lock-free animation body. Nothing is constructed here except
    the one worker that owns the durable scene-memory write.
    """

    def __init__(
        self,
        *,
        fsm: BehaviorFSM,
        blackboard: Blackboard,
        conversation: object,
        narration: object,
        observations: object,
        router: object,
        director_lock: threading.RLock,
        memory_store: SceneMemoryStore,
        on_cleared: Callable[[], None],
        executor: Executor | None = None,
    ) -> None:
        if not isinstance(fsm, BehaviorFSM):
            raise TypeError("fsm must be a BehaviorFSM")
        if not isinstance(blackboard, Blackboard):
            raise TypeError("blackboard must be a Blackboard")
        _require_fsm_fields(fsm)
        stages = {
            "conversation": conversation,
            "narration": narration,
            "observation": observations,
            "body": router,
        }
        for name, stage in stages.items():
            if not callable(getattr(stage, "reset", None)):
                raise TypeError(f"{name} must provide reset()")
        if not callable(on_cleared):
            raise TypeError("on_cleared must be callable")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must provide submit()")
        if not callable(getattr(memory_store, "save", None)):
            raise TypeError("memory_store must provide save()")

        self._fsm = fsm
        self._blackboard = blackboard
        self._conversation = conversation
        self._narration = narration
        self._observations = observations
        self._router = router
        self._director_lock = director_lock
        self._memory = memory_store
        self._on_cleared = on_cleared
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="luxo-reset"
        )
        self._owns_executor = executor is None

        # Length 1: a second request that arrives before the first is drained
        # is the same request. ``append`` and ``popleft`` are atomic, so the
        # producer side never takes a lock and is safe inside a signal handler.
        self._requests: deque[str] = deque(maxlen=1)
        self._lock = threading.RLock()
        self._applied = 0
        self._failed_steps: tuple[str, ...] = ()
        self._last_reason: str | None = None
        self._memory_clears_committed = 0
        self._memory_clears_failed = 0
        self._closed = False

    # ------------------------------------------------------------- requesting

    def request(self, reason: str = DEFAULT_REASON) -> None:
        """Ask for a reset. Signal-handler safe: one atomic append, no work.

        This must stay exactly this cheap. It may run on the main thread
        between bytecodes, where taking a non-reentrant lock or touching a
        component risks deadlocking against the very state it wants to clear.
        Validation, logging, and every mutation belong to :meth:`drain`.
        """

        self._requests.append(reason)

    @property
    def pending(self) -> bool:
        return bool(self._requests)

    @property
    def status(self) -> ResetStatus:
        with self._lock:
            return ResetStatus(
                pending=bool(self._requests),
                applied=self._applied,
                failed_steps=self._failed_steps,
                last_reason=self._last_reason,
                memory_clears_committed=self._memory_clears_committed,
                memory_clears_failed=self._memory_clears_failed,
                closed=self._closed,
            )

    # ---------------------------------------------------------------- applying

    def drain(self, now: float) -> ResetReport | None:
        """Apply at most one pending reset, or return ``None`` if none is.

        Call this only from the serialized behaviour tick, and before the tick
        does anything else, so the rest of the beat runs on the new baseline.
        A drain with nothing pending is a deque read and nothing more.
        """

        reason = self._take_request()
        if reason is None:
            return None
        instant = _finite_time(now)
        with self._lock:
            if self._closed:
                return None
            return self._apply_locked(instant, reason)

    def close(self) -> None:
        """Make the reset permanently inert and release an owned pool."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._requests.clear()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _take_request(self) -> str | None:
        try:
            return self._requests.popleft()
        except IndexError:
            return None

    def _apply_locked(self, now: float, reason: str) -> ResetReport:
        """Run the fixed sequence, guarding each step independently."""

        label = reason if isinstance(reason, str) and reason.strip() else DEFAULT_REASON
        previous = self._fsm.state
        steps: tuple[tuple[str, Callable[[], object]], ...] = (
            ("fsm", lambda: self._reset_fsm(now)),
            ("conversation", self._conversation.reset),  # type: ignore[attr-defined]
            ("narration", self._narration.reset),  # type: ignore[attr-defined]
            ("observation", self._observations.reset),  # type: ignore[attr-defined]
            ("body", self._reset_body),
            ("blackboard", self._blackboard.reset),
            ("runtime", self._on_cleared),
        )
        failed: list[str] = []
        for name, step in steps:
            try:
                step()
            except Exception:
                LOGGER.exception("demo reset step %s failed", name)
                failed.append(name)

        submitted = self._submit_memory_clear()
        self._applied += 1
        self._failed_steps = tuple(failed)
        self._last_reason = label
        report = ResetReport(
            sequence=self._applied,
            t=now,
            reason=label,
            previous_state=previous,
            failed_steps=tuple(failed),
            memory_clear_submitted=submitted,
        )
        if report.clean:
            LOGGER.info(
                "demo reset %d applied from %s (%s)",
                report.sequence,
                previous.value,
                label,
            )
        else:
            LOGGER.error(
                "demo reset %d applied from %s (%s) with failures %s",
                report.sequence,
                previous.value,
                label,
                ",".join(failed) or "memory",
            )
        return report

    def _reset_fsm(self, now: float) -> None:
        """Write the resting baseline in one hold of the FSM's own lock.

        Warm-model and browser-ready facts are preserved deliberately: nothing
        will ever post ``MODELS_WARM`` or ``BROWSER_READY`` again, so dropping
        them would leave the character permanently unable to leave BOOT. Queued
        events are counted as dropped rather than silently discarded, so the
        FSM's own status keeps telling the truth about them.
        """

        fsm = self._fsm
        with fsm._lock:  # noqa: SLF001 - see _FSM_FIELDS
            fsm._dropped_events += len(fsm._events)
            fsm._events.clear()
            fsm._state = RESET_STATE
            fsm._state_entered_at = now
            fsm._last_transition = None
            fsm._gaze_on = False
            fsm._gaze_on_since = None
            fsm._gaze_off_since = None

    def _reset_body(self) -> None:
        """Clear plan, blocker, pending effects, and body under the one lock.

        ``ActionRouter.reset`` clears the plan queue and then resets the
        director, which drops its pending cues and capture requests, forgets
        its scheduled notice and droop beats, and returns the animation runtime
        to rest. The director carries no lock of its own, so this takes the
        runtime's director lock exactly as every other entry into that body
        does, which is what keeps it serialized against the 120 Hz tick.
        """

        with self._director_lock:
            self._router.reset()  # type: ignore[attr-defined]

    def _submit_memory_clear(self) -> bool:
        try:
            future = self._executor.submit(self._clear_memory)
        except Exception as error:
            LOGGER.warning(
                "durable scene-memory clear was refused (%s)", type(error).__name__
            )
            self._memory_clears_failed += 1
            return False
        future.add_done_callback(self._memory_cleared)
        return True

    def _clear_memory(self) -> None:
        """Blocking scene-memory write; this never runs on a tick thread."""

        self._memory.save(())

    def _memory_cleared(self, future: Future[object]) -> None:
        try:
            error = future.exception()
        except BaseException as cancelled:  # noqa: BLE001 - a cancelled future
            error = cancelled
        with self._lock:
            if error is None:
                self._memory_clears_committed += 1
                return
            self._memory_clears_failed += 1
        LOGGER.warning(
            "durable scene-memory clear failed (%s); the mirror is still empty",
            type(error).__name__,
        )


def _require_fsm_fields(fsm: BehaviorFSM) -> None:
    missing = tuple(name for name in _FSM_FIELDS if not hasattr(fsm, name))
    if missing:
        raise AttributeError(
            "BehaviorFSM no longer carries "
            + ", ".join(missing)
            + "; the demo reset writes these directly and must be updated with it"
        )


def _finite_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("now must be a finite number")
    instant = float(value)
    if not math.isfinite(instant) or instant < 0.0:
        raise ValueError("now must be a finite nonnegative number")
    return instant


__all__ = [
    "DEFAULT_REASON",
    "DemoReset",
    "RESET_STATE",
    "STEP_NAMES",
    "ResetReport",
    "ResetStatus",
]
