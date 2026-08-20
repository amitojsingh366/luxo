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

Identity. ``PlanExecutor.pending_observation_id`` and
``ObservationCoordinator.pending`` mint their ids independently, so the runtime
treats their agreement as an invariant to be checked, never assumed. It is
re-verified before every capture dispatch and again before any frame is handed
to a worker. A disagreement is a bug: it is logged, counted, and resolved by
clearing both sides together so neither is left stranded.

Comparison. The model performs perception (``observe``) and narration
(``narrate``) only. ``missing`` is computed here in Python by
:func:`core.brain.missing.compute_missing` against the baseline captured
*before* the observation, and the model is never asked to compare (PRD 8.3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final

from ..brain.missing import (
    MissingComparison,
    MissingNarration,
    MissingObjectCoordinator,
    compute_observation_missing,
)
from ..brain.observe import (
    ObservationBusyError,
    ObservationCoordinator,
    ObservationRequestError,
)
from ..brain.client import ObservationEnvelope, ValidatedObservation
from ..brain.schema import ObservationResponse, PlanResponse, normalize_canonical_label
from ..fsm import BehaviorEvent, BehaviorFSM, BehaviorState
from ..plan_executor import ObservationReleaseError, PlanExecutor
from ..protocol.messages import CaptureFrameMessage

LOGGER = logging.getLogger(__name__)

MAX_CAPTURE_ATTEMPTS: Final = 3
DEFAULT_WORKERS: Final = 1

CaptureCallback = Callable[[CaptureFrameMessage], None]
NarrationCallback = Callable[[MissingNarration], None]
SceneCommenter = Callable[[str, ValidatedObservation], PlanResponse]
SceneCommentCallback = Callable[[PlanResponse], None]
BaselineLabels = Callable[[], Sequence[str]]
NarratePolicy = Callable[[MissingComparison], bool | Sequence[str] | None]


class ObservationStage(str, Enum):
    """Position of the one in-flight observation, owned by the runtime."""

    IDLE = "idle"
    REQUESTED = "requested"
    ANALYZING = "analyzing"
    NARRATING = "narrating"
    COMMENTING = "commenting"


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
    narrations: int
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


def _narrate_when_missing(comparison: MissingComparison) -> bool:
    """Narrate only a real absence; a plain inspection has nothing to report."""

    return bool(comparison.missing)


def _selected_comparison(
    comparison: MissingComparison,
    decision: bool | Sequence[str] | None,
) -> MissingComparison | None:
    """Turn policy output into a safe subset; ``None`` means stay silent."""

    if decision is True:
        selected = comparison.missing
    elif decision is False or decision is None:
        return None
    else:
        if isinstance(decision, (str, bytes, bytearray)) or not isinstance(
            decision, Sequence
        ):
            raise TypeError("narration policy must return bool, labels, or None")
        selected = tuple(
            dict.fromkeys(normalize_canonical_label(label) for label in decision)
        )
        if not set(selected).issubset(comparison.missing):
            raise ValueError("narration policy selected an object that is not missing")
    return MissingComparison(comparison.baseline, comparison.present, selected)


class ObservationRuntime:
    """Tie one ``observe`` op to one captured frame, memory, and narration.

    Every public method is safe to call from socket, worker, and tick threads.
    The outbound callbacks must be quick, non-blocking enqueues and must never
    call back into the runtime: they are invoked while the runtime lock is held
    so that no reset or close can interleave between the generation check and
    the enqueue.

    ``baseline_labels`` is read on the tick and must therefore be cheap and
    non-blocking. Wire it to the blackboard's mirrored scene memory, never to a
    disk load.
    """

    def __init__(
        self,
        *,
        fsm: BehaviorFSM,
        plan_executor: PlanExecutor,
        observations: ObservationCoordinator,
        missing: MissingObjectCoordinator,
        baseline_labels: BaselineLabels,
        capture_callback: CaptureCallback,
        narration_callback: NarrationCallback,
        scene_commenter: SceneCommenter | None = None,
        scene_comment_callback: SceneCommentCallback | None = None,
        should_narrate: NarratePolicy | None = None,
        executor: Executor | None = None,
    ) -> None:
        callbacks = (baseline_labels, capture_callback, narration_callback)
        if not all(callable(item) for item in callbacks):
            raise TypeError("observation runtime callbacks must be callable")
        if should_narrate is not None and not callable(should_narrate):
            raise TypeError("should_narrate must be callable")
        if (scene_commenter is None) != (scene_comment_callback is None):
            raise TypeError("scene comment worker and callback must be provided together")
        if scene_commenter is not None and not callable(scene_commenter):
            raise TypeError("scene_commenter must be callable")
        if scene_comment_callback is not None and not callable(scene_comment_callback):
            raise TypeError("scene_comment_callback must be callable")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must provide submit()")
        self._fsm = fsm
        self._plans = plan_executor
        self._observations = observations
        self._missing = missing
        self._baseline_labels = baseline_labels
        self._capture_callback = capture_callback
        self._narration_callback = narration_callback
        self._scene_commenter = scene_commenter
        self._scene_comment_callback = scene_comment_callback
        self._should_narrate = should_narrate or _narrate_when_missing
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
        self._baseline: tuple[str, ...] = ()
        self._attempts = 0
        self._inspecting = False
        self._stale_frame_expected = False
        self._captures_requested = 0
        self._observations_completed = 0
        self._narrations = 0
        self._rejected_frames = 0
        self._faults = 0
        self._last_missing: tuple[str, ...] = ()
        self._last_rejection: FrameRejection | None = None
        self._last_error: str | None = None
        self._visual_intent: str | None = None

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
                narrations=self._narrations,
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
                # A refused worker leaves the capture unanalyzed, so it takes
                # the same bounded re-capture path as a failed analysis.
                self._retry_or_fault_locked(type(error).__name__)
                return False
            self._future = future
            future.add_done_callback(
                lambda done, token=generation: self._queue(token, "analyze", done)
            )
            return True

    def request_visual_followup(self, intent: str) -> bool:
        """Attach one dialogue intent to the next observation.

        The intent contains no image or telemetry. It is held until a complete
        JPEG has passed through perception and memory, then consumed by the
        scene-comment worker. A second request is refused rather than replacing
        the observation it belongs to.
        """

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("visual intent must be non-empty text")
        normalized = " ".join(intent.split())
        with self._lock:
            if (
                self._closed
                or self._stage is not ObservationStage.IDLE
                or self._visual_intent is not None
                or self._scene_commenter is None
            ):
                return False
            self._visual_intent = normalized
            return True

    def cancel_visual_followup(self) -> bool:
        """Forget an intent whose plan was cancelled; report whether one existed."""

        with self._lock:
            pending = self._visual_intent is not None
            self._visual_intent = None
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
        try:
            baseline = self._baseline_snapshot()
        except Exception as error:
            self._fault_locked(type(error).__name__, announce_complete=False)
            return

        try:
            request = self._observations.begin(baseline)
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
        self._baseline = request.prior_canonical
        self._attempts = 0
        self._stage = ObservationStage.REQUESTED
        self._inspecting = True
        self._fsm.post_event(BehaviorEvent.OBSERVE_START)
        if not self._dispatch_capture_locked():
            self._fault_locked("capture_dispatch_failed", announce_complete=True)

    def _dispatch_capture_locked(self) -> bool:
        """Ask the browser for exactly one frame for the pending request."""

        request_id = self._request_id
        if request_id is None:  # pragma: no cover - internal invariant
            raise RuntimeError("cannot dispatch a capture without a request id")
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
            else (
                ObservationStage.NARRATING
                if completion.kind == "narrate"
                else ObservationStage.COMMENTING
            )
        )
        if self._stage is not expected:
            self._last_error = "out_of_order_completion"
            return
        self._future = None
        if completion.kind == "analyze":
            self._apply_analyze_locked(completion.future)
        elif completion.kind == "narrate":
            self._apply_narrate_locked(completion.future)
        else:
            self._apply_comment_locked(completion.future)

    def _apply_analyze_locked(self, future: Future[object]) -> None:
        """Release the blocker, then compare locally before any narration."""

        try:
            response = future.result()
            if not isinstance(response, (ObservationResponse, ObservationEnvelope)):
                raise TypeError("observation must return validated observation facts")
        except Exception as error:
            self._retry_or_fault_locked(type(error).__name__)
            return

        request_id = self._request_id
        try:
            if request_id is None:
                raise ObservationReleaseError("observation completed without a request id")
            self._plans.release_observation(request_id)
        except ObservationReleaseError:
            # Memory already changed, so the surviving blocker cannot simply be
            # ignored; both sides are cleared and the inspection is ended.
            self._fault_locked("id_disagreement", announce_complete=True)
            return

        self._observations_completed += 1
        self._announce_complete_locked()
        facts = response.facts if isinstance(response, ObservationEnvelope) else response
        comparison = compute_observation_missing(self._baseline, facts)
        self._last_missing = comparison.missing
        LOGGER.info(
            "SCENE comparison=%s",
            json.dumps(
                {
                    "baseline": comparison.baseline,
                    "visible": comparison.present,
                    "missing": comparison.missing,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        if self._visual_intent is not None:
            self._begin_comment_locked(response)
            return
        selected = _selected_comparison(comparison, self._should_narrate(comparison))
        if selected is None:
            self._finish_locked()
            return

        generation = self._generation
        self._stage = ObservationStage.NARRATING
        try:
            narration_future = self._executor.submit(self._narrate, selected)
        except Exception as error:
            self._last_error = type(error).__name__
            self._finish_locked()
            return
        self._future = narration_future
        narration_future.add_done_callback(
            lambda done, token=generation: self._queue(token, "narrate", done)
        )

    def _begin_comment_locked(self, response: ValidatedObservation) -> None:
        """Run the approved fourth call only after fresh facts are committed."""

        intent = self._visual_intent
        commenter = self._scene_commenter
        if intent is None or commenter is None:  # pragma: no cover - guarded caller
            self._finish_locked()
            return
        generation = self._generation
        self._stage = ObservationStage.COMMENTING
        try:
            comment_future = self._executor.submit(commenter, intent, response)
        except Exception as error:
            self._last_error = type(error).__name__
            self._finish_locked()
            return
        self._future = comment_future
        comment_future.add_done_callback(
            lambda done, token=generation: self._queue(token, "comment", done)
        )

    def _apply_narrate_locked(self, future: Future[object]) -> None:
        """Route one narration to speech and append its plan to the queue."""

        try:
            narration = future.result()
            if not isinstance(narration, MissingNarration):
                raise TypeError("narration must return a validated MissingNarration")
        except Exception as error:
            # The blocker is already released, so a failed narration costs the
            # line and nothing else; the rest of the plan still runs.
            self._last_error = type(error).__name__
            self._finish_locked()
            return

        self._narrations += 1
        self._last_missing = narration.comparison.missing
        if narration.response.plan:
            self._plans.submit(narration.response.plan)
        try:
            self._narration_callback(narration)
        except Exception:
            LOGGER.exception("narration callback failed")
            self._last_error = "narration_callback_failed"
        self._finish_locked()

    def _apply_comment_locked(self, future: Future[object]) -> None:
        """Route a grounded follow-up through the normal plan and speech owners."""

        try:
            response = future.result()
            if not isinstance(response, PlanResponse):
                raise TypeError("scene comment must return a validated PlanResponse")
        except Exception as error:
            self._last_error = type(error).__name__
            self._finish_locked()
            return

        if response.plan:
            self._plans.submit(response.plan)
        callback = self._scene_comment_callback
        if callback is not None:
            try:
                callback(response)
            except Exception:
                LOGGER.exception("scene comment callback failed")
                self._last_error = "scene_comment_callback_failed"
        self._finish_locked()

    def _retry_or_fault_locked(self, reason: str) -> None:
        """Re-capture the same request, or clear both sides once exhausted.

        No image bytes are retained anywhere, so a failed analysis is retried
        by asking for a new frame rather than by replaying the old one. The
        coordinator keeps the exact request pending after a failure, which is
        what lets the retry keep the same id on both sides.
        """

        self._last_error = reason
        if self._attempts >= MAX_CAPTURE_ATTEMPTS or not self._agreed_locked():
            self._fault_locked("observe_failed", announce_complete=True)
            return
        self._stage = ObservationStage.REQUESTED
        if not self._dispatch_capture_locked():
            self._fault_locked("capture_dispatch_failed", announce_complete=True)

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

    def _baseline_snapshot(self) -> tuple[str, ...]:
        labels = self._baseline_labels()
        if isinstance(labels, (str, bytes, bytearray)) or not isinstance(
            labels, Sequence
        ):
            raise TypeError("baseline_labels must return a sequence of canonical labels")
        return tuple(labels)

    def _analyze(self, request_id: str, jpeg: bytes) -> ValidatedObservation:
        """Blocking capture-to-memory work; this never runs on the tick."""

        return self._observations.complete(request_id, jpeg)

    def _narrate(self, comparison: MissingComparison) -> MissingNarration:
        """Call ``narrate`` with only the locally selected missing labels."""

        return self._missing.narrate_comparison(comparison)

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
        self._inspecting = False
        self._stale_frame_expected = False
        self._visual_intent = None

    def _invalidate_locked(self, *, announce_complete: bool = False) -> None:
        """Abandon the observation, clearing both sides in one lock hold."""

        self._generation += 1
        self._completions.clear()
        future, self._future = self._future, None
        if future is not None:
            future.cancel()
        if self._stage in _OUTSTANDING:
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
        self._inspecting = False
        self._visual_intent = None

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


__all__ = [
    "BaselineLabels",
    "CaptureCallback",
    "DEFAULT_WORKERS",
    "FrameRejection",
    "MAX_CAPTURE_ATTEMPTS",
    "NarrationCallback",
    "NarratePolicy",
    "SceneCommentCallback",
    "SceneCommenter",
    "ObservationRuntime",
    "ObservationStage",
    "ObservationStatus",
]
