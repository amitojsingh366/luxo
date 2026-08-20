"""Blocking ``observe`` staging: plan blocker to one frame to memory to speech.

The owning runtime must tick exactly this order at 10 Hz::

    BehaviorFSM.tick()
        -> ConversationCoordinator.tick()
        -> PlanExecutor.tick()
        -> ObservationRuntime.tick()

The observation runtime ticks last because the blocker it services is created
inside ``PlanExecutor.tick()``; ticking last lets one tick both raise and
service the blocker (PRD 6.4).

Blocking means the *plan* blocks, not the process. The plan queue is held from
the moment the ``observe`` op is emitted until a real frame has been through
the vision model and scene memory, while the tick thread itself never performs
network, disk, or model work (PRD 4.4). Capture round-trip and model work run
on a worker; the worker completion callback is allowed to do exactly one thing,
enqueue a generation-tagged completion, and every state change lands on the
serialized tick.

Wire mapping. A JPEG arrives as a bare ``0x02`` binary frame with no request id,
because PRD 10 gives binary frames a single type-prefix byte and no room for
one. That is deliberate and is not worked around here: a frame is mapped to the
one active blocker or to nothing at all. ``on_jpeg`` therefore rejects
unsolicited frames (no capture outstanding) and stale frames (a capture was
outstanding but was cancelled by reset, disengage, or a fault) instead of
feeding them to the model. The one case the wire cannot express is a frame from
a cancelled capture that lands while a *later* capture is outstanding: with no
id on the frame it is indistinguishable from the frame that was asked for, so
it is claimed by the active blocker. Rejection covers every case the protocol
makes decidable, and inventing a wire-level request id to cover the rest is out
of scope by design.

Identity. ``PlanExecutor.pending_observation_id`` is the sole request-id owner.
The runtime passes that exact value into ``ObservationCoordinator.begin`` and
re-verifies agreement before capture dispatch and before handing any frame to a
worker. A disagreement is a bug: it is logged, counted, and resolved by clearing
both sides together so neither is left stranded.

Resolution. The vision call returns facts only. ``missing`` is computed here in
Python against the baseline captured *before* the observation. The runtime then
always makes one generalized resolution call with the exact typed origin,
fresh facts, and complete diff, including when the diff is empty.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final

from ..brain.client import ObservationOrigin, RecentExchange
from ..brain.missing import compute_observation_missing
from ..brain.observe import (
    ObservationBusyError,
    ObservationCoordinator,
    ObservationRequestError,
)
from ..brain.schema import ObservationPrior, ObservationResponse, PlanResponse
from ..fsm import BehaviorEvent, BehaviorFSM, BehaviorState
from ..plan_executor import ObservationReleaseError, PlanExecutor
from ..protocol.messages import CaptureFrameMessage

LOGGER = logging.getLogger(__name__)

MAX_CAPTURE_ATTEMPTS: Final = 1
DEFAULT_WORKERS: Final = 1
INSPECTION_CUE_S: Final = 0.55
INSPECTION_FAILURE_SAY: Final = "I couldn't get a clear look—could you show me again?"

CaptureCallback = Callable[[CaptureFrameMessage], None]
ObservationResolver = Callable[
    [
        ObservationOrigin,
        ObservationResponse,
        tuple[ObservationPrior, ...],
        tuple[RecentExchange, ...],
    ],
    PlanResponse,
]
ResolutionCallback = Callable[[ObservationOrigin, PlanResponse], bool]
BaselineObjects = Callable[[], Sequence[ObservationPrior]]


class ObservationStage(str, Enum):
    """Position of the one in-flight observation, owned by the runtime."""

    IDLE = "idle"
    PREPARING = "preparing"
    REQUESTED = "requested"
    ANALYZING = "analyzing"
    RESOLVING = "resolving"


class FrameRejection(str, Enum):
    """Why an arriving JPEG was refused instead of reaching the model."""

    UNSOLICITED = "unsolicited_frame"
    STALE = "stale_frame"
    DISAGREEMENT = "id_disagreement"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ObservationStatus:
    """Immutable staging view carrying no image, label, or plan payload."""

    generation: int
    stage: ObservationStage
    request_id: str | None
    baseline_size: int
    capture_attempts: int
    captures_requested: int
    observations_completed: int
    resolutions: int
    rejected_frames: int
    faults: int
    last_missing: tuple[str, ...]
    last_rejection: FrameRejection | None
    last_error: str | None
    closed: bool


@dataclass(frozen=True, slots=True)
class _Completion:
    """One finished worker result, tagged with the generation that issued it."""

    generation: int
    kind: str
    future: Future[object]


# Engagement has ended, so any outstanding capture is abandoned rather than
# resumed. Matches the conversation coordinator's inactive set exactly.
_INACTIVE: Final = frozenset(
    {BehaviorState.BOOT, BehaviorState.DORMANT, BehaviorState.DISENGAGING}
)
_OUTSTANDING: Final = frozenset(
    {ObservationStage.REQUESTED, ObservationStage.ANALYZING}
)


class ObservationRuntime:
    """Tie one typed observation origin to one frame, memory, and resolution.

    Every public method is safe to call from socket, worker, and tick threads.
    The outbound callbacks must be quick, non-blocking enqueues and must never
    call back into the runtime: they are invoked while the runtime lock is held
    so that no reset or close can interleave between the generation check and
    the enqueue.

    ``baseline_objects`` is read on the tick and must therefore be cheap and
    non-blocking. Wire it to the blackboard's mirrored scene memory, never to a
    disk load.
    """

    def __init__(
        self,
        *,
        fsm: BehaviorFSM,
        plan_executor: PlanExecutor,
        observations: ObservationCoordinator,
        baseline_objects: BaselineObjects,
        capture_callback: CaptureCallback,
        resolver: ObservationResolver,
        resolution_callback: ResolutionCallback,
        clock: Callable[[], float] = time.monotonic,
        executor: Executor | None = None,
    ) -> None:
        callbacks = (
            baseline_objects,
            capture_callback,
            resolver,
            resolution_callback,
            clock,
        )
        if not all(callable(item) for item in callbacks):
            raise TypeError("observation runtime callbacks must be callable")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must provide submit()")
        self._fsm = fsm
        self._plans = plan_executor
        self._observations = observations
        self._baseline_objects = baseline_objects
        self._capture_callback = capture_callback
        self._resolver = resolver
        self._resolution_callback = resolution_callback
        self._clock = clock
        self._last_clock = _clock_value(clock())
        self._executor = executor or ThreadPoolExecutor(
            max_workers=DEFAULT_WORKERS, thread_name_prefix="luxo-observation"
        )
        self._owns_executor = executor is None
        self._lock = RLock()
        self._generation = 0
        self._closed = False
        self._stage = ObservationStage.IDLE
        self._completions: list[_Completion] = []
        self._future: Future[object] | None = None
        self._request_id: str | None = None
        self._baseline: tuple[ObservationPrior, ...] = ()
        self._attempts = 0
        self._capture_due_at: float | None = None
        self._inspecting = False
        self._stale_frame_expected = False
        self._captures_requested = 0
        self._observations_completed = 0
        self._resolutions = 0
        self._rejected_frames = 0
        self._faults = 0
        self._last_missing: tuple[str, ...] = ()
        self._last_rejection: FrameRejection | None = None
        self._last_error: str | None = None
        self._pending_origin: ObservationOrigin | None = None
        self._active_origin: ObservationOrigin | None = None
        self._origin_recent: tuple[RecentExchange, ...] = ()

    @property
    def status(self) -> ObservationStatus:
        with self._lock:
            return ObservationStatus(
                generation=self._generation,
                stage=self._stage,
                request_id=self._request_id,
                baseline_size=len(self._baseline),
                capture_attempts=self._attempts,
                captures_requested=self._captures_requested,
                observations_completed=self._observations_completed,
                resolutions=self._resolutions,
                rejected_frames=self._rejected_frames,
                faults=self._faults,
                last_missing=self._last_missing,
                last_rejection=self._last_rejection,
                last_error=self._last_error,
                closed=self._closed,
            )

    @property
    def stage(self) -> ObservationStage:
        with self._lock:
            return self._stage

    @property
    def awaiting_frame(self) -> bool:
        """Report whether exactly one capture is outstanding right now."""

        with self._lock:
            return self._stage is ObservationStage.REQUESTED

    def tick(self) -> None:
        """Drain completions and arm at most one capture; never blocks on I/O."""

        with self._lock:
            if self._closed:
                return
            state = self._fsm.state
            completions, self._completions = self._completions, []
            if state in _INACTIVE:
                if self._stage is not ObservationStage.IDLE:
                    self._invalidate_locked()
                return
            for completion in completions:
                self._apply_locked(completion)
            if (
                self._stage is ObservationStage.IDLE
                and self._plans.blocked_on_observation
            ):
                self._begin_locked()
            elif self._stage is ObservationStage.PREPARING:
                try:
                    now = self._now_locked()
                except ValueError as error:
                    self._fault_locked(str(error), announce_complete=True)
                else:
                    if self._fsm.state is BehaviorState.INSPECTING:
                        if self._capture_due_at is None:
                            self._capture_due_at = now + INSPECTION_CUE_S
                        elif (
                            now >= self._capture_due_at
                            and not self._dispatch_capture_locked()
                        ):
                            self._fault_locked(
                                "capture_dispatch_failed", announce_complete=True
                            )

    def on_jpeg(self, jpeg: bytes) -> bool:
        """Map one prefix-free ``0x02`` payload to the single active blocker.

        Returns whether the frame was accepted for analysis. The wire carries
        no request id, so a frame is only ever matched to the one outstanding
        capture; anything else is refused here and never reaches the model.
        """

        if not isinstance(jpeg, (bytes, bytearray, memoryview)):
            raise TypeError("capture payload must be binary frame bytes")
        frame = bytes(jpeg)
        with self._lock:
            if self._closed:
                return self._reject_locked(FrameRejection.CLOSED)
            if self._stage is not ObservationStage.REQUESTED:
                # No capture is outstanding. A frame is stale only if the
                # capture that would have claimed it was cancelled.
                return self._reject_locked(
                    FrameRejection.STALE
                    if self._stale_frame_expected
                    else FrameRejection.UNSOLICITED
                )
            if not self._agreed_locked():
                self._fault_locked("id_disagreement", announce_complete=True)
                return self._reject_locked(FrameRejection.DISAGREEMENT)

            generation = self._generation
            request_id = self._request_id
            self._stage = ObservationStage.ANALYZING
            try:
                future = self._executor.submit(self._analyze, request_id, frame)
            except Exception as error:
                self._fail_observation_locked(type(error).__name__)
                return False
            self._future = future
            future.add_done_callback(
                lambda done, token=generation: self._queue(token, "analyze", done)
            )
            return True

    def bind_origin(
        self,
        origin: ObservationOrigin,
        recent: Sequence[RecentExchange] = (),
    ) -> bool:
        """Bind one typed cloud-approved origin to the next observe blocker."""

        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        exchanges = tuple(recent)
        if not all(isinstance(item, RecentExchange) for item in exchanges):
            raise TypeError("recent must contain RecentExchange values")
        with self._lock:
            if (
                self._closed
                or self._stage is not ObservationStage.IDLE
                or self._pending_origin is not None
                or self._active_origin is not None
            ):
                return False
            self._pending_origin = origin
            self._origin_recent = exchanges[-3:]
            return True

    def cancel_origin(self) -> bool:
        """Forget an origin whose plan was cancelled before observation."""

        with self._lock:
            pending = self._pending_origin is not None or self._active_origin is not None
            self._pending_origin = None
            self._active_origin = None
            self._origin_recent = ()
            return pending

    def disengage(self) -> None:
        """Abandon the observation on gaze loss or an explicit runtime stop."""

        with self._lock:
            if not self._closed:
                self._invalidate_locked()

    def reset(self) -> None:
        """Abandon the observation and forget error state; use on reconnect."""

        with self._lock:
            if self._closed:
                return
            self._invalidate_locked()
            self._last_error = None
            self._last_rejection = None

    def close(self) -> None:
        """Make the runtime permanently inert and release an owned pool."""

        with self._lock:
            if self._closed:
                return
            self._invalidate_locked()
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _begin_locked(self) -> None:
        """Reserve one capture and prove both sides agree before dispatching."""

        pending_id = self._plans.pending_observation_id
        if pending_id is None:  # pragma: no cover - checked by the caller
            return
        if self._pending_origin is None:
            self._fault_locked("unbound_observation", announce_complete=False)
            return
        try:
            baseline = self._baseline_snapshot()
        except Exception as error:
            self._fault_locked(type(error).__name__, announce_complete=False)
            return

        try:
            request = self._observations.begin(
                pending_id,
                baseline,
                self._pending_origin,
            )
        except ObservationBusyError:
            existing = self._observations.pending
            if existing is None or existing.request_id != pending_id:
                # Some other request owns the coordinator. Refusing to steal it
                # would strand the plan, so both sides are cleared together.
                self._fault_locked("observation_busy", announce_complete=False)
                return
            # The pending request is exactly the blocker's own, so adopting it
            # preserves the one-capture invariant instead of breaking it.
            request = existing
        except Exception as error:
            self._fault_locked(type(error).__name__, announce_complete=False)
            return

        if request.request_id != pending_id:
            self._fault_locked("id_disagreement", announce_complete=False)
            return

        self._request_id = request.request_id
        self._baseline = request.prior
        self._active_origin = self._pending_origin
        self._pending_origin = None
        self._attempts = 0
        try:
            cue_started = self._now_locked()
        except ValueError as error:
            self._fault_locked(str(error), announce_complete=False)
            return
        already_inspecting = self._fsm.state is BehaviorState.INSPECTING
        self._stage = ObservationStage.PREPARING
        self._inspecting = True
        self._capture_due_at = (
            cue_started + INSPECTION_CUE_S if already_inspecting else None
        )
        # Dialogue observations enter INSPECTING directly from the typed cloud
        # result so the physical lean-in precedes this capture. Spontaneous
        # observations still begin in ACTING and need the ordinary event.
        if not already_inspecting:
            self._fsm.post_event(BehaviorEvent.OBSERVE_START)

    def _dispatch_capture_locked(self) -> bool:
        """Ask the browser for exactly one frame for the pending request."""

        request_id = self._request_id
        if request_id is None:  # pragma: no cover - internal invariant
            raise RuntimeError("cannot dispatch a capture without a request id")
        self._stage = ObservationStage.REQUESTED
        self._attempts += 1
        self._captures_requested += 1
        try:
            self._capture_callback(CaptureFrameMessage(request_id))
        except Exception as error:
            self._last_error = type(error).__name__
            return False
        return True

    def _queue(self, generation: int, kind: str, future: Future[object]) -> None:
        """Enqueue only: this may run inline on the registering thread."""

        with self._lock:
            if not self._closed and generation == self._generation:
                self._completions.append(_Completion(generation, kind, future))

    def _apply_locked(self, completion: _Completion) -> None:
        if completion.generation != self._generation:
            return
        expected = (
            ObservationStage.ANALYZING
            if completion.kind == "analyze"
            else ObservationStage.RESOLVING
        )
        if self._stage is not expected:
            self._last_error = "out_of_order_completion"
            return
        self._future = None
        if completion.kind == "analyze":
            self._apply_analyze_locked(completion.future)
        else:
            self._apply_resolve_locked(completion.future)

    def _apply_analyze_locked(self, future: Future[object]) -> None:
        """Commit fresh facts, compare locally, then resolve the exact origin."""

        try:
            response = future.result()
            if not isinstance(response, ObservationResponse):
                raise TypeError("observation must return validated observation facts")
        except Exception as error:
            self._fail_observation_locked(type(error).__name__)
            return

        self._observations_completed += 1
        comparison = compute_observation_missing(self._baseline, response)
        self._last_missing = comparison.missing_labels
        LOGGER.info(
            "SCENE comparison=%s",
            json.dumps(
                {
                    "baseline_ids": comparison.baseline_ids,
                    "visible_ids": comparison.present_ids,
                    "missing_ids": comparison.missing_ids,
                    "missing": comparison.missing_labels,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        origin = self._active_origin
        if origin is None:
            self._fault_locked("missing_active_origin", announce_complete=True)
            return
        generation = self._generation
        self._stage = ObservationStage.RESOLVING
        try:
            resolution_future = self._executor.submit(
                self._resolver,
                origin,
                response,
                comparison.missing,
                self._origin_recent,
            )
        except Exception as error:
            self._last_error = type(error).__name__
            self._fault_locked("resolver_submit_failed", announce_complete=True)
            return
        self._future = resolution_future
        resolution_future.add_done_callback(
            lambda done, token=generation: self._queue(token, "resolve", done)
        )

    def _apply_resolve_locked(self, future: Future[object]) -> None:
        """Stage one cloud resolution before releasing its observation blocker."""

        try:
            response = future.result()
            if not isinstance(response, PlanResponse):
                raise TypeError("resolver must return a validated PlanResponse")
        except Exception as error:
            self._last_error = type(error).__name__
            self._fault_locked("resolve_failed", announce_complete=True)
            return

        origin = self._active_origin
        if origin is None:
            self._fault_locked("missing_active_origin", announce_complete=True)
            return
        try:
            accepted = self._resolution_callback(origin, response)
        except Exception:
            LOGGER.exception("observation resolution callback failed")
            accepted = False
        if not accepted:
            self._fault_locked("resolution_speech_refused", announce_complete=True)
            return
        if response.plan:
            self._plans.submit(response.plan)
        request_id = self._request_id
        try:
            if request_id is None:
                raise ObservationReleaseError("observation completed without a request id")
            self._plans.release_observation(request_id)
        except ObservationReleaseError:
            self._fault_locked("id_disagreement", announce_complete=True)
            return
        self._resolutions += 1
        self._inspecting = False
        self._fsm.post_event(BehaviorEvent.OBSERVATION_RESPONSE)
        self._finish_locked()

    def _fail_observation_locked(self, reason: str) -> None:
        """Fail the one-frame observation without sampling a different scene."""

        self._last_error = reason
        origin = self._active_origin
        if origin is None:
            self._fault_locked("observe_failed", announce_complete=True)
            return
        try:
            accepted = self._resolution_callback(
                origin,
                PlanResponse(INSPECTION_FAILURE_SAY, ()),
            )
        except Exception:
            LOGGER.exception("observation failure speech callback failed")
            accepted = False
        if not accepted:
            self._fault_locked("observe_failed", announce_complete=True)
            return
        self._faults += 1
        self._release_blocker_locked()
        self._cancel_pending_locked()
        self._inspecting = False
        self._fsm.post_event(BehaviorEvent.OBSERVATION_RESPONSE)
        self._finish_locked()

    def _reject_locked(self, reason: FrameRejection) -> bool:
        """Refuse one frame and record why; the model is never given it."""

        if reason is FrameRejection.STALE:
            # One cancelled capture can strand at most the frame it asked for.
            self._stale_frame_expected = False
        self._rejected_frames += 1
        self._last_rejection = reason
        LOGGER.warning("rejected capture frame: %s", reason.value)
        return False

    def _agreed_locked(self) -> bool:
        """Require exact three-way agreement on the pending observation id."""

        request_id = self._request_id
        if request_id is None:
            return False
        pending = self._observations.pending
        if pending is None or pending.request_id != request_id:
            return False
        return self._plans.pending_observation_id == request_id

    def _baseline_snapshot(self) -> tuple[ObservationPrior, ...]:
        objects = self._baseline_objects()
        if isinstance(objects, (str, bytes, bytearray)) or not isinstance(
            objects, Sequence
        ):
            raise TypeError("baseline_objects must return a sequence of ObservationPrior values")
        result = tuple(objects)
        if not all(isinstance(item, ObservationPrior) for item in result):
            raise TypeError("baseline_objects must contain ObservationPrior values")
        return result

    def _now_locked(self) -> float:
        """Read one finite, nonnegative, monotonic injected-clock value."""

        now = _clock_value(self._clock())
        if now < self._last_clock:
            raise ValueError("clock_moved_backward")
        self._last_clock = now
        return now

    def _analyze(self, request_id: str, jpeg: bytes) -> ObservationResponse:
        """Blocking capture-to-memory work; this never runs on the tick."""

        return self._observations.complete(request_id, jpeg)

    def _announce_complete_locked(self) -> None:
        """Report that the inspection ended, never that memory changed."""

        if self._inspecting:
            self._inspecting = False
            self._fsm.post_event(BehaviorEvent.OBSERVE_COMPLETE)

    def _finish_locked(self) -> None:
        self._stage = ObservationStage.IDLE
        self._future = None
        self._request_id = None
        self._baseline = ()
        self._attempts = 0
        self._capture_due_at = None
        self._inspecting = False
        self._stale_frame_expected = False
        self._pending_origin = None
        self._active_origin = None
        self._origin_recent = ()

    def _invalidate_locked(self, *, announce_complete: bool = False) -> None:
        """Abandon the observation, clearing both sides in one lock hold."""

        self._generation += 1
        self._completions.clear()
        future, self._future = self._future, None
        if future is not None:
            future.cancel()
        if self._stage in _OUTSTANDING and self._attempts > 0:
            # A frame may already be in flight for the capture being cancelled.
            self._stale_frame_expected = True
        self._cancel_pending_locked()
        self._release_blocker_locked()
        if announce_complete:
            self._announce_complete_locked()
        self._stage = ObservationStage.IDLE
        self._request_id = None
        self._baseline = ()
        self._attempts = 0
        self._capture_due_at = None
        self._inspecting = False
        self._pending_origin = None
        self._active_origin = None
        self._origin_recent = ()

    def _fault_locked(self, reason: str, *, announce_complete: bool) -> None:
        LOGGER.error("observation runtime cleared both sides after %s", reason)
        self._last_error = reason
        self._faults += 1
        self._invalidate_locked(announce_complete=announce_complete)

    def _cancel_pending_locked(self) -> None:
        pending = self._observations.pending
        if pending is None:
            return
        try:
            self._observations.cancel(pending.request_id)
        except ObservationRequestError:  # pragma: no cover - lost a cancel race
            LOGGER.warning("pending observation vanished before cancellation")

    def _release_blocker_locked(self) -> None:
        pending_id = self._plans.pending_observation_id
        if pending_id is None:
            return
        try:
            self._plans.release_observation(pending_id)
        except ObservationReleaseError:  # pragma: no cover - lost a release race
            LOGGER.warning("observation blocker vanished before release")


def _clock_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clock_must_be_finite_nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("clock_must_be_finite_nonnegative")
    return result


__all__ = [
    "BaselineObjects",
    "CaptureCallback",
    "DEFAULT_WORKERS",
    "INSPECTION_CUE_S",
    "INSPECTION_FAILURE_SAY",
    "FrameRejection",
    "MAX_CAPTURE_ATTEMPTS",
    "ObservationResolver",
    "ResolutionCallback",
    "ObservationRuntime",
    "ObservationStage",
    "ObservationStatus",
]
