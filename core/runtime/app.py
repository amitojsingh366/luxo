"""Assemble the Luxo character: one process, three rates, one owner per fact.

Every other module in this package is complete and unit-tested in isolation.
This module is the only place where they meet, so it is also the only place
where an ownership question can be answered. The answers are recorded here
because a wiring decision that lives only in code is a decision nobody can
audit.

Rate separation
---------------

Three domains run in one process (PRD 4.4):

* **10 Hz behaviour** — :meth:`LuxoApp.tick_behavior`. Owns intent.
* **120 Hz animation** — :meth:`LuxoApp.tick_animation`. Owns motion and
  performs no I/O of any kind (see below).
* **60 Hz ``body_state``** — published from every second animation tick, so the
  body snapshot is always exactly one animation sample old and never
  interpolated (PRD 10.2).

Blocking work — whisper, Piper, OpenRouter, scene-memory persistence, latency
CSV appends — runs only on worker pools. Workers deliver results by setting
blackboard fields and by enqueueing generation-tagged completions that the
serialized ticks drain.

Clock domains
-------------

The FSM, VAD, gaze, plan, conversation, and observation domains all use
``time.time()``, because the browser stamps ``gaze``, ``vad``, and ``tts_done``
with ``Date.now() / 1000`` and those timestamps are compared directly against
core timestamps.

The animation domain is a **fixed-step timeline anchored to the epoch**:
``t_n = anchor + n * FIXED_DT`` where ``anchor = time.time()`` at start.
``time.monotonic()` is used only to decide *when* to run the next step, never
as a value. This is not a stylistic choice. ``AnimationDirector`` schedules its
notice-freeze and droop beats as ``transition.t + delta``, where ``transition.t``
comes from the FSM, and then compares those deadlines against its own tick time.
A monotonic animation timeline would put those two on different origins and the
NOTICING regard beat and the DISENGAGING droop gesture would never fire.
``AnimationRuntime`` was written for this: its 1 microsecond step tolerance is
documented against "contemporary epoch values". Anchoring to the epoch and
advancing by exact multiples of ``FIXED_DT`` keeps consecutive deltas within
about 1.3e-7 s of ``FIXED_DT`` for hours, well inside that tolerance.

The 10 Hz tick order
--------------------

``interactions.py`` and ``observations.py`` each document their own position.
Combined, and with ``actions.py`` line 141 replacing the bare executor tick::

    BehaviorFSM.tick()
        -> ConversationCoordinator.tick()
        -> ActionRouter.tick()          # ticks PlanExecutor internally
        -> ObservationRuntime.tick()

``PlanExecutor.tick()`` is never called from here: ``ActionRouter.tick()`` is
its only caller, exactly once per behaviour tick. The observation runtime ticks
last so that one tick can both raise a blocker and begin servicing it.

Two steps in that chain are conditional, and both conditions are stated with the
constant that carries them: the router is skipped in ``PLAN_HELD_STATES``, and
the observation runtime is skipped while its blocker is still waiting to be
armed (see ``OBSERVATION_ARM_TIMEOUT_S`` below). Neither skip can strand
anything, and both are pinned by checks.

Capture ownership — the decision this module exists to make
-----------------------------------------------------------

``ActionRouter`` takes a ``capture_callback`` and dispatches a
``CaptureFrameMessage`` for each accepted ``observe``; it also exposes
``release_observation``. ``ObservationRuntime`` *also* takes a
``capture_callback`` and independently releases the executor's blocker. Wired
naively that is two JPEGs for one ``observe`` (violating the PRD 5.3
one-frame-per-observe guarantee) or two releases of one blocker.

**ObservationRuntime owns the capture lifecycle and is the only component that
emits ``capture_frame`` to the browser.** The deciding reason is retry
ownership: ``ObservationRuntime._retry_or_fault_locked`` re-requests a frame
through its own ``capture_callback`` after a failed analysis or a refused
worker, up to ``MAX_CAPTURE_ATTEMPTS``. ``ActionRouter`` emits once per accepted
``observe`` and has no re-request path at all. If the router owned emission, the
first frame would arrive and every retry would be swallowed, leaving the plan
blocked on a frame that was never asked for again. Staleness handling
(``on_jpeg`` rejecting unsolicited and stale frames) lives with the same
component for the same reason.

**The router's ``capture_callback`` therefore feeds the observation runtime, not
the protocol.** It is the *arming* signal: :meth:`LuxoApp._on_router_capture`
records the request id and never touches the socket. ``ObservationRuntime.tick``
is withheld while the runtime is idle, a blocker exists, and that arming signal
has not yet named the blocker's id. This preserves a real body-owned beat: when
a plan is ``scan`` then ``observe``, ``AnimationDirector`` deliberately holds its
``CaptureRequest`` until the sweep finishes, so the lamp looks *again* and then
captures, rather than capturing at the start of the sweep (PRD 6.4, 13 beat 11).
Comparing against the executor's current pending id makes a stale arming signal
harmless: ``PlanExecutor`` issues every id exactly once, so an id from a
cancelled observation can never match a later one.

A watchdog bounds that deferral. If a blocker stays un-armed for
``OBSERVATION_ARM_TIMEOUT_S`` the runtime is ticked anyway and the delay is
logged, so a director fault can never deadlock the plan queue.

**Exactly one release path touches the executor.**
``ActionRouter.release_observation`` is never called by this module and is not
wired to anything. ``ObservationRuntime`` performs the only
``PlanExecutor.release_observation`` calls, on successful analysis and on
invalidation. The other two ways a blocker disappears are total *clears*
(``PlanExecutor.clear``), not releases: ``ActionRouter.apply_transition`` clears
in a ``finally`` on a dormant or disengaging beat, and
``ConversationCoordinator`` clears when it invalidates. ``clear`` is idempotent
and cannot raise ``ObservationReleaseError``, and the tick order runs the
clearing components before the observation runtime, so by the time
``ObservationRuntime._release_blocker_locked`` looks there is no pending id left
to release.

Cancellation is atomic across all three
---------------------------------------

:meth:`LuxoApp._cancel_engagement` runs the router transition, the
conversation disengage, and the observation disengage as independent guarded
steps. A failure in any one is logged and the rest still run, so a director that
rejects a transition can never leave the plan queue, the speech staging, or the
observation blocker live on its own.

Outbound ownership
------------------

``ProtocolServer`` frames ``0x03`` and is the only thing that does. Inbound,
``0x01`` utterance PCM goes to ``ConversationCoordinator.accept_utterance`` and
``0x02`` goes to ``ObservationRuntime.on_jpeg`` as the prefix-free body.

Locking
-------

``AnimationDirector`` and ``AnimationRuntime`` carry no lock of their own: they
were written as single-threaded bodies. Here the behaviour thread mutates them
(through the router's ``apply_action`` and ``apply_transition``) while the
animation thread ticks them, so this module supplies the missing serialization
with one director lock held around every entry into that body. Lock order is
always::

    director lock -> ActionRouter lock -> PlanExecutor lock -> app state lock

The observation runtime's lock is only ever taken *before* the app state lock,
never after, so an inbound ``capture_frame`` dispatch and a behaviour-tick arm
check cannot deadlock against each other.

The animation tick and I/O
--------------------------

:meth:`tick_animation` calls only: a blackboard snapshot, two speech-envelope
lookups, the director's fixed step, and ``publish_body_state``. It never
serializes a message, never touches the event loop, never opens a file, never
submits a worker job, and never calls a model.

Being precise about the locks it does take, because "no I/O" is not the same as
"no waiting":

* the **director lock**, held for the fixed step, contended only by the 10 Hz
  behaviour tick doing pure in-memory routing;
* the **blackboard lock**, for one snapshot;
* the **app state lock**, for counters and the cached state name;
* the **conversation and narration locks**, for ``speaking`` and
  ``speech_amplitude``. Both read two fields and return; the speech worker holds
  the conversation lock only across a chunk enqueue, never across Piper
  synthesis, which happens outside it.

Every one of those is a bounded in-memory critical section. It never takes the
router, FSM, observation, or plan-executor lock at all. Telemetry it needs from
the behaviour domain (state name, arousal, last latency) is cached as a plain
attribute by the domain that owns it, so the 120 Hz tick never waits on the
10 Hz one to read it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Protocol

from ..animation.director import AnimationDirector
from ..animation.lookat import LookAtTarget
from ..animation.runtime import FIXED_DT, TICK_HZ, AnimationDiscontinuityError, AnimationRuntime, AnimationSample
from ..animation.poses import PoseLibrary, load_pose_library
from ..blackboard import Blackboard, BlackboardSnapshot, GazeFact, Telemetry
from ..brain.client import BrainClient, CallMetrics
from ..brain.memory import SceneMemoryStore
from ..brain.missing import MissingNarration, MissingObjectCoordinator
from ..brain.observe import ObservationCoordinator
from ..brain.schema import Action
from ..config import FrozenConfig, load_config
from ..fsm import BehaviorEvent, BehaviorFSM, BehaviorState, Transition
from ..instrumentation import InteractionCSVLogger, InteractionTimeline, Milestone, TimelineError
from ..plan_executor import PlanExecutor
from ..protocol.messages import (
    AudioState,
    BinaryFrame,
    BinaryFrameType,
    BodyStateMessage,
    BrowserToCoreMessage,
    CaptureFrameMessage,
    ClampCounts as WireClampCounts,
    CoreToBrowserMessage,
    CueMessage,
    ErrorMessage,
    GazeMessage,
    HelloMessage,
    JointsState,
    LightState,
    SpeakBeginMessage,
    SpeakEndMessage,
    TelemetryGaze,
    TelemetryState,
    TtsDoneMessage,
    VadMessage,
)
from ..speech.stt import SpeechToText
from ..speech.tts import DEFAULT_CHUNK_BYTES, ENVELOPE_HZ
from ..speech.tts import SAMPLE_HZ as TTS_SAMPLE_HZ
from ..speech.tts import SpeechAudio, TextToSpeech
from ..wake_sequence import WakeSequenceCoordinator
from .actions import CANCELLING_STATES, ActionRouter
from .interactions import ConversationCoordinator
from .observations import ObservationRuntime, ObservationStage

LOGGER = logging.getLogger(__name__)

BEHAVIOR_HZ: Final = 10.0
ANIMATION_HZ: Final = float(TICK_HZ)
BODY_STATE_HZ: Final = 60.0
BODY_STATE_EVERY_N_TICKS: Final = 2
"""120 Hz animation divided by 2 is the 60 Hz body_state rate of PRD 10.2."""

OBSERVATION_ARM_TIMEOUT_S: Final = 2.0
"""Upper bound on how long a blocker may wait for the director's capture beat."""

PLAN_HELD_STATES: Final = frozenset(
    {BehaviorState.LISTENING, BehaviorState.THINKING, BehaviorState.SPEAKING}
)
"""States in which the plan queue waits instead of advancing.

PRD 6.1 lists ``SPEAKING -> ACTING | plan queue non-empty after speech`` and
``ACTING -> INSPECTING | observe op reached``. Both are unreachable if the
router drains the plan as soon as the brain submits it: a three-action plan
empties in 0.3 s of behaviour ticks while a two-second utterance is still
playing, so ``SPEECH_DONE`` would always find an empty queue and take
``speech_done_without_plan``, and an ``observe`` would fire mid-sentence in a
state where the FSM does not handle ``OBSERVE_START`` at all. Holding the plan
across the three staging states is what keeps the model's plan an accompaniment
to the utterance rather than a race against it.

The router is still ticked in every other state, including ``BOOT`` for the
waking sequence and ``INSPECTING`` where the executor is blocked but the
director's effects must still drain. The one cost is that an effect raised on
the animation tick during a held state waits for the next unheld tick; the only
producer of such an effect is a notice-freeze beat that outlived its own state,
which is already degenerate.
"""

SCENE_LOOK_ELEVATION_RAD: Final = 0.15
"""Slightly downward, matching the director's own scan elevation."""

MAX_TICK_BACKSTEP_S: Final = 1.0
"""A wall-clock step back larger than this re-anchors instead of being clamped."""

AROUSAL_BY_STATE: Final = {
    BehaviorState.BOOT: 0.05,
    BehaviorState.DORMANT: 0.15,
    BehaviorState.NOTICING: 0.45,
    BehaviorState.ENGAGED: 0.70,
    BehaviorState.LISTENING: 0.75,
    BehaviorState.THINKING: 0.60,
    BehaviorState.SPEAKING: 0.85,
    BehaviorState.INSPECTING: 0.80,
    BehaviorState.ACTING: 0.75,
    BehaviorState.DISENGAGING: 0.25,
}
"""App-owned arousal channel for the browser's music density (PRD 9.3).

No other module computes arousal and the PRD does not fix the numbers, so this
table is a judgment call made here rather than an omission. It is a
presentation projection of the state the FSM already owns; it never feeds back
into behaviour.
"""


class ProtocolPort(Protocol):
    """The only three transport calls the character core makes outbound."""

    def publish_body_state(self, state: BodyStateMessage) -> None: ...

    def publish_event(self, message: CoreToBrowserMessage) -> None: ...

    def publish_tts_pcm(self, payload: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraGeometry:
    """Frame geometry reported by the browser's ``hello`` (PRD 10.1)."""

    width: int = 640
    height: int = 480
    hfov_deg: float = 60.0

    @property
    def vfov_rad(self) -> float:
        hfov = math.radians(self.hfov_deg)
        aspect = self.width / self.height
        return 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)


def target_angles(
    centre_x: float, centre_y: float, camera: CameraGeometry
) -> tuple[float, float]:
    """Map a normalized frame point to emitter azimuth and elevation.

    Byte-for-byte the same projection the renderer applies to the face centroid
    (``renderer/src/sensors/gaze.ts``: image-right is positive azimuth,
    image-down is positive elevation, which is the direction Luxo's flipped
    head pitch calls down). Reusing it means an ``obj:<id>`` target and a
    ``person`` target arrive in the solver on one convention.
    """

    hfov = math.radians(camera.hfov_deg)
    x = min(1.0, max(0.0, centre_x))
    y = min(1.0, max(0.0, centre_y))
    azimuth = math.atan(2.0 * (x - 0.5) * math.tan(hfov / 2.0))
    elevation = math.atan(2.0 * (y - 0.5) * math.tan(camera.vfov_rad / 2.0))
    return azimuth, elevation


@dataclass(frozen=True, slots=True)
class AppStatus:
    """Immutable diagnostic view; carries no transcript, image, or plan text."""

    state: BehaviorState
    behavior_ticks: int
    animation_ticks: int
    body_states: int
    plan_depth: int
    memory_count: int
    hellos: int
    capture_frames_sent: int
    router_capture_signals: int
    observation_arm_timeouts: int
    narrations_spoken: int
    last_latency_ms: float
    running: bool


class NarrationSpeaker:
    """Speak one observation narration over the PRD 8.4 wire contract.

    ``ConversationCoordinator`` owns dialogue speech but exposes no public
    "speak this text" entry point, and ``interactions.py`` is not owned by this
    packet, so the observation runtime's ``narration_callback`` cannot be routed
    into it. This class is the interim owner of that hop: it reuses the same
    ``TextToSpeech`` instance, the same ``speak_begin``/PCM/``speak_end``
    sequence, and the same browser ``tts_done`` authority, and it publishes its
    own ``speaking`` flag and envelope so ``body_state`` stays truthful while a
    narration plays.

    It deliberately does not retry. ``ObservationRuntime`` has already released
    the plan blocker by the time a narration is dispatched, so a failed line
    costs the line and nothing else.
    """

    def __init__(
        self,
        *,
        tts: TextToSpeech,
        speak_callback: Callable[[SpeakBeginMessage | SpeakEndMessage], None],
        pcm_callback: Callable[[bytes], None],
        clock: Callable[[], float] = time.time,
        executor: Executor | None = None,
    ) -> None:
        if not all(callable(item) for item in (speak_callback, pcm_callback, clock)):
            raise TypeError("narration callbacks must be callable")
        self._tts = tts
        self._speak_callback = speak_callback
        self._pcm_callback = pcm_callback
        self._clock = clock
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="luxo-narration"
        )
        self._owns_executor = executor is None
        self._lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._speaking = False
        self._awaiting_done = False
        self._envelope: tuple[float, ...] = ()
        self._started_at: float | None = None
        self._spoken = 0
        self._failures = 0
        self._last_error: str | None = None

    @property
    def speaking(self) -> bool:
        """True from ``speak_begin`` until the browser reports ``tts_done``.

        Deliberately identical in shape to ``ConversationCoordinator.speaking``:
        the flag tracks audio the browser is still playing, not bytes the core
        is still writing, so ``body_state.audio.speaking`` stays true for the
        whole utterance and the music keeps ducking under it.
        """

        with self._lock:
            return self._speaking

    @property
    def awaiting_done(self) -> bool:
        with self._lock:
            return self._awaiting_done

    @property
    def spoken(self) -> int:
        with self._lock:
            return self._spoken

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def speech_amplitude(self, now: float) -> float:
        """Look up the 50 Hz envelope; pure and safe on the animation tick."""

        with self._lock:
            started, envelope = self._started_at, self._envelope
        if started is None or now < started:
            return 0.0
        index = int((now - started) * ENVELOPE_HZ)
        return envelope[index] if index < len(envelope) else 0.0

    def speak(self, text: str) -> bool:
        """Queue one narration line; returns whether a worker was armed."""

        if not isinstance(text, str):
            raise TypeError("narration text must be a string")
        line = " ".join(text.split())
        if not line:
            return False
        with self._lock:
            if self._closed or self._speaking or self._awaiting_done:
                return False
            self._generation += 1
            generation = self._generation
            try:
                self._executor.submit(self._deliver, generation, line)
            except Exception as error:
                self._last_error = type(error).__name__
                self._failures += 1
                return False
            return True

    def on_tts_done(self, t: float) -> bool:
        """Accept the browser as the only authority for narration completion."""

        with self._lock:
            if self._closed or not self._awaiting_done:
                return False
            started = self._started_at
            if started is not None and t < started:
                return False
            self._clear_locked()
            return True

    def disengage(self) -> None:
        """Abandon narration audio on gaze loss or an explicit runtime stop."""

        with self._lock:
            if self._closed:
                return
            self._generation += 1
            self._clear_locked()

    def reset(self) -> None:
        self.disengage()
        with self._lock:
            self._last_error = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            self._clear_locked()
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _clear_locked(self) -> None:
        self._speaking = False
        self._awaiting_done = False
        self._envelope = ()
        self._started_at = None

    def _deliver(self, generation: int, text: str) -> None:
        """Synthesize and stream one line; this never runs on a tick thread."""

        try:
            audio = self._synthesize(text)
            chunks = self._chunks(audio)
        except Exception as error:
            with self._lock:
                if generation == self._generation:
                    self._last_error = type(error).__name__
                    self._failures += 1
            LOGGER.warning("narration synthesis failed (%s)", type(error).__name__)
            return

        try:
            with self._lock:
                if self._closed or generation != self._generation:
                    return
                self._envelope = tuple(audio.envelope)
                self._started_at = self._clock()
                self._speaking = True
                self._awaiting_done = True
                self._speak_callback(SpeakBeginMessage(ENVELOPE_HZ))
            for chunk in chunks:
                with self._lock:
                    if self._closed or generation != self._generation:
                        return
                    self._pcm_callback(chunk)
            with self._lock:
                if self._closed or generation != self._generation:
                    return
                # ``speaking`` stays true: the browser is only now starting to
                # play what was streamed. Only tts_done, disengage, or close
                # clears it, so completion is never fabricated here.
                self._speak_callback(SpeakEndMessage())
                self._spoken += 1
        except Exception as error:
            with self._lock:
                if generation != self._generation:
                    return
                self._last_error = type(error).__name__
                self._failures += 1
                if self._awaiting_done:
                    # Audio already reached the browser. Send one best-effort
                    # speak_end and then wait, exactly as the conversation
                    # coordinator does after a partial delivery.
                    try:
                        self._speak_callback(SpeakEndMessage())
                    except Exception:
                        LOGGER.debug("speak_end after a partial narration failed")
                else:
                    self._clear_locked()
            LOGGER.warning("narration delivery failed (%s)", type(error).__name__)

    def _synthesize(self, text: str) -> SpeechAudio:
        audio = self._tts.synthesize(text)
        if not isinstance(audio, SpeechAudio):
            raise TypeError("TTS must return SpeechAudio")
        if audio.sample_hz != TTS_SAMPLE_HZ or audio.envelope_hz != ENVELOPE_HZ:
            raise ValueError("TTS returned an incompatible audio contract")
        return audio

    def _chunks(self, audio: SpeechAudio) -> tuple[bytes, ...]:
        chunks = tuple(self._tts.chunks(audio, max_bytes=DEFAULT_CHUNK_BYTES))
        if not chunks or b"".join(chunks) != audio.pcm_int16:
            raise ValueError("TTS chunks do not reconstruct the utterance")
        for chunk in chunks:
            if not chunk or len(chunk) % 2 or len(chunk) > DEFAULT_CHUNK_BYTES:
                raise ValueError("TTS chunk is not even-sized PCM of at most 8 KiB")
        return chunks


class LatencyRecorder:
    """Turn coordinator milestones into PRD 11.1 CSV rows without blocking.

    ``ConversationCoordinator`` invokes ``milestone_callback`` while holding its
    own lock, and one of the six milestones is emitted from the speech worker
    mid-delivery. Marking a milestone is therefore an in-memory append, and the
    row commit — which appends and ``fsync``s a file — is handed to a worker.

    Token counts and the resolved model id come from ``BrainClient``'s metrics
    callback. When a transport never reported usage the row is still written,
    using the configured model and profile with zero tokens, and counted, so a
    missing usage field degrades the row rather than losing the interaction.
    """

    def __init__(
        self,
        logger: InteractionCSVLogger,
        *,
        model: str,
        profile: str,
        executor: Executor | None = None,
    ) -> None:
        self._logger = logger
        self._model = model
        self._profile = profile
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="luxo-latency"
        )
        self._owns_executor = executor is None
        self._lock = threading.RLock()
        self._timeline: InteractionTimeline | None = None
        self._usage: tuple[str, int, int] | None = None
        self._closed = False
        self.last_latency_ms = 0.0
        self.rows_committed = 0
        self.rows_dropped = 0
        self.rows_without_usage = 0

    def on_call_metrics(self, metrics: CallMetrics) -> None:
        """Capture only the four CSV-permitted fields from a converse call."""

        if getattr(metrics, "call_type", None) != "converse":
            return
        with self._lock:
            self._usage = (metrics.model, metrics.tokens_in, metrics.tokens_out)

    def on_milestone(self, milestone: Milestone, t: float) -> None:
        """Record one milestone. Fast, allocation-light, never touches disk."""

        with self._lock:
            if self._closed:
                return
            if milestone is Milestone.VAD_END:
                self._timeline = self._logger.new_interaction()
                self._usage = None
            timeline = self._timeline
            if timeline is None:
                return
            try:
                timeline.mark(milestone, t)
            except TimelineError as error:
                # A dropped interaction is normal: STT can fail, gaze can be
                # lost mid-turn. Abandon the row rather than writing a lie.
                LOGGER.debug("latency timeline abandoned: %s", error)
                self._timeline = None
                self.rows_dropped += 1
                return
            if milestone is not Milestone.FIRST_AUDIO_CHUNK:
                return
            usage = self._usage
            self._timeline = None
            self._usage = None
        self._commit(timeline, usage)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._timeline = None
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _commit(
        self, timeline: InteractionTimeline, usage: tuple[str, int, int] | None
    ) -> None:
        model, tokens_in, tokens_out = usage or (self._model, 0, 0)
        if usage is None:
            with self._lock:
                self.rows_without_usage += 1
        try:
            timeline.set_model_usage(
                model=model,
                profile=self._profile,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        except (TimelineError, ValueError) as error:
            LOGGER.warning("latency row rejected its model usage: %s", error)
            with self._lock:
                self.rows_dropped += 1
            return
        try:
            self._executor.submit(self._write, timeline)
        except Exception as error:
            LOGGER.warning("latency commit was refused (%s)", type(error).__name__)
            with self._lock:
                self.rows_dropped += 1

    def _write(self, timeline: InteractionTimeline) -> None:
        """Blocking CSV append; this only ever runs on the latency worker."""

        record = self._logger.commit(timeline)
        with self._lock:
            if record is None:
                self.rows_dropped += 1
                return
            self.rows_committed += 1
            self.last_latency_ms = record.stage_ms_end_to_end


class LuxoApp:
    """One assembled character: config, mind, body, transport, and clocks.

    Every collaborator is injectable so the whole assembly can be driven
    offline with fakes and a driven clock. ``tick_behavior`` and
    ``tick_animation`` are public for exactly that reason: the threads started
    by :meth:`start` do nothing except pace those two calls.
    """

    def __init__(
        self,
        *,
        protocol: ProtocolPort,
        stt: SpeechToText,
        tts: TextToSpeech,
        brain: BrainClient,
        memory_store: SceneMemoryStore,
        config: FrozenConfig | None = None,
        poses: PoseLibrary | None = None,
        latency_logger: InteractionCSVLogger | None = None,
        latency_recorder: LatencyRecorder | None = None,
        brain_model: str = "unknown",
        brain_profile: str = "free",
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        conversation_executor: Executor | None = None,
        observation_executor: Executor | None = None,
        narration_executor: Executor | None = None,
        warm_executor: Executor | None = None,
        latency_executor: Executor | None = None,
        idle_seed: int = 0,
    ) -> None:
        self._config = config if config is not None else load_config()
        _check_rates(self._config)
        self._protocol = protocol
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._memory = memory_store

        self._blackboard = Blackboard()
        self._fsm = BehaviorFSM()
        self._plans = PlanExecutor()

        pose_library = poses if poses is not None else load_pose_library()
        self._animation = AnimationRuntime(pose_library, idle_seed=idle_seed)
        self._director = AnimationDirector(self._animation, self._resolve_look_target)

        self._router = ActionRouter(
            director=self._director,
            plan_executor=self._plans,
            cue_callback=self._send_cue,
            # Deliberately NOT the protocol: this arms the observation runtime.
            capture_callback=self._on_router_capture,
        )

        # A caller that needs token counts must build the recorder itself and
        # hand the same object to the brain as its metrics callback, because
        # ``BrainClient`` takes that callback at construction and the brain is
        # built before the app. Passing only a logger is still supported; those
        # rows carry the configured model id and zero tokens.
        if latency_recorder is not None:
            self._latency: LatencyRecorder | None = latency_recorder
        elif latency_logger is not None:
            self._latency = LatencyRecorder(
                latency_logger,
                model=brain_model,
                profile=brain_profile,
                executor=latency_executor,
            )
        else:
            self._latency = None

        self._conversation = ConversationCoordinator(
            blackboard=self._blackboard,
            fsm=self._fsm,
            plan_executor=self._plans,
            stt=stt,
            brain=brain,
            tts=tts,
            compact_memory=self._memory.compact_line,
            speak_callback=self._send_speak,
            pcm_callback=self._protocol.publish_tts_pcm,
            milestone_callback=None if self._latency is None else self._latency.on_milestone,
            clock=clock,
            sleep=sleep,
            executor=conversation_executor,
        )

        self._observation_coordinator = ObservationCoordinator(
            brain,
            self._memory,
            clock=clock,
            publish=self._blackboard.set_scene_memory,
        )
        self._observations = ObservationRuntime(
            fsm=self._fsm,
            plan_executor=self._plans,
            observations=self._observation_coordinator,
            missing=MissingObjectCoordinator(brain),
            baseline_labels=self._baseline_labels,
            # The single owner of outbound capture_frame.
            capture_callback=self._send_capture_frame,
            narration_callback=self._on_narration,
            executor=observation_executor,
        )

        self._narration = NarrationSpeaker(
            tts=tts,
            speak_callback=self._send_speak,
            pcm_callback=self._protocol.publish_tts_pcm,
            clock=clock,
            executor=narration_executor,
        )

        self._wake = WakeSequenceCoordinator(
            stt=stt,
            tts=tts,
            brain=brain,
            emit_action=self._emit_wake_action,
            post_event=self._fsm.post_event,
            executor=warm_executor,
        )

        # Serializes the lock-free animation body against the behaviour thread.
        self._director_lock = threading.RLock()
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self._prepared = False
        self._started = False

        self._animation_anchor: float | None = None
        self._animation_step = 0
        self._animation_ticks = 0
        self._behavior_ticks = 0
        self._body_states = 0
        self._last_behavior_time: float | None = None
        self._last_sample: AnimationSample | None = None

        self._capture_armed_id: str | None = None
        self._pending_blocker_id: str | None = None
        self._pending_blocker_since: float | None = None
        self._router_capture_signals = 0
        self._capture_frames_sent = 0
        self._observation_arm_timeouts = 0
        self._cue_failures = 0

        self._camera = CameraGeometry()
        self._hellos = 0
        self._camera_ready = False

        # Read by the animation thread; written by the behaviour thread. Plain
        # attribute assignment keeps the 120 Hz tick off every other lock.
        self._state_name = BehaviorState.BOOT
        self._arousal = AROUSAL_BY_STATE[BehaviorState.BOOT]

    # ---------------------------------------------------------------- status

    @property
    def blackboard(self) -> Blackboard:
        return self._blackboard

    @property
    def fsm(self) -> BehaviorFSM:
        return self._fsm

    @property
    def plans(self) -> PlanExecutor:
        return self._plans

    @property
    def router(self) -> ActionRouter:
        return self._router

    @property
    def conversation(self) -> ConversationCoordinator:
        return self._conversation

    @property
    def observations(self) -> ObservationRuntime:
        return self._observations

    @property
    def narration(self) -> NarrationSpeaker:
        return self._narration

    @property
    def wake(self) -> WakeSequenceCoordinator:
        return self._wake

    @property
    def director(self) -> AnimationDirector:
        return self._director

    @property
    def latency_recorder(self) -> LatencyRecorder | None:
        return self._latency

    def status(self) -> AppStatus:
        snapshot = self._blackboard.snapshot()
        state = self._fsm.state
        narrations = self._narration.spoken
        with self._lock:
            return AppStatus(
                state=state,
                behavior_ticks=self._behavior_ticks,
                animation_ticks=self._animation_ticks,
                body_states=self._body_states,
                plan_depth=len(snapshot.plan_queue),
                memory_count=len(snapshot.scene_memory),
                hellos=self._hellos,
                capture_frames_sent=self._capture_frames_sent,
                router_capture_signals=self._router_capture_signals,
                observation_arm_timeouts=self._observation_arm_timeouts,
                narrations_spoken=narrations,
                last_latency_ms=0.0 if self._latency is None else self._latency.last_latency_ms,
                running=self._started and not self._stopping.is_set(),
            )

    # ------------------------------------------------------------- lifecycle

    def prepare(self) -> None:
        """Do everything :meth:`start` does except own a thread.

        Splitting this out is what makes the whole assembly driveable offline:
        a caller runs ``prepare`` and then paces :meth:`tick_behavior` and
        :meth:`tick_animation` itself, with no thread and no real clock.

        Model warm-up is submitted to its own pool and reports back through the
        FSM, so this returns immediately; the lamp performs the PRD 10.4 waking
        sequence while whisper and Piper are still loading.
        """

        with self._lock:
            if self._prepared:
                return
            self._prepared = True
            self._animation_anchor = self._clock()
            self._animation_step = 0
        self._load_memory()
        self._wake.warm()

    def start(self) -> None:
        """Prepare the character, then start the behaviour and animation threads."""

        with self._lock:
            if self._started:
                return
            self._started = True
        self.prepare()
        self._threads = [
            threading.Thread(target=self._behavior_loop, name="luxo-behavior", daemon=True),
            threading.Thread(target=self._animation_loop, name="luxo-animation", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        LOGGER.info("character core running: 10 Hz behaviour, %d Hz animation", TICK_HZ)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the tick threads, then make every stage permanently inert."""

        self._stopping.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = []
        for name, close in (
            ("wake", self._wake.close),
            ("conversation", self._conversation.close),
            ("observations", self._observations.close),
            ("narration", self._narration.close),
        ):
            try:
                close()
            except Exception:
                LOGGER.exception("shutting down %s failed", name)
        if self._latency is not None:
            self._latency.close()
        with self._lock:
            self._started = False
        LOGGER.info("character core stopped")

    def _load_memory(self) -> None:
        """One startup disk read; scene memory is mirrored for cheap lookups."""

        try:
            objects = self._memory.load()
        except Exception:
            LOGGER.exception("scene memory could not be loaded; starting empty")
            return
        self._blackboard.set_scene_memory(objects)

    # ------------------------------------------------------- protocol inbound

    def on_message(self, message: BrowserToCoreMessage) -> None:
        """Handle one browser text frame. Runs on the socket thread."""

        try:
            if isinstance(message, HelloMessage):
                self._on_hello(message)
            elif isinstance(message, GazeMessage):
                self._on_gaze(message)
            elif isinstance(message, VadMessage):
                self._on_vad(message)
            elif isinstance(message, TtsDoneMessage):
                self._on_tts_done(message)
            elif isinstance(message, ErrorMessage):
                LOGGER.warning("browser reported %s: %s", message.where, message.detail)
        except Exception:
            LOGGER.exception("handling %r failed", getattr(message, "type", message))

    def on_binary(self, frame: BinaryFrame) -> None:
        """Route one browser binary frame. Runs on the socket thread.

        Both branches hand off to a worker immediately, so a large utterance or
        a JPEG never occupies the socket thread and never reaches a tick.
        """

        try:
            if frame.kind is BinaryFrameType.UTTERANCE_PCM:
                self._conversation.accept_utterance(frame.payload)
            elif frame.kind is BinaryFrameType.CAPTURE_JPEG:
                self._observations.on_jpeg(frame.payload)
            else:  # pragma: no cover - 0x03 is outbound only and never parsed here
                LOGGER.warning("ignoring inbound binary prefix %r", frame.kind)
        except Exception:
            LOGGER.exception("handling binary frame %r failed", frame.kind)

    def _on_hello(self, message: HelloMessage) -> None:
        camera = message.camera
        with self._lock:
            self._hellos += 1
            reconnect = self._hellos > 1
            self._camera = CameraGeometry(camera.w, camera.h, camera.hfov_deg)
        if reconnect:
            # A new browser will never send tts_done for audio the previous one
            # was playing, and will never return a frame the previous one was
            # asked for. Clear both so a reconnect cannot strand either.
            self._conversation.reset()
            self._narration.reset()
            self._observations.reset()
            LOGGER.info("browser reconnected; speech and observation staging reset")
        self._wake.mark_browser_hello()

    def _on_gaze(self, message: GazeMessage) -> None:
        self._blackboard.publish_gaze(
            GazeFact(
                t=message.t,
                present=message.present,
                yaw_deg=message.yaw_deg,
                pitch_deg=message.pitch_deg,
                az=message.az,
                el=message.el,
                conf=message.conf,
            )
        )
        # The renderer refuses to start the gaze sensor until the camera track
        # is live, so any gaze frame at all is proof of camera permission.
        with self._lock:
            already_ready = self._camera_ready
            self._camera_ready = True
        if not already_ready:
            self._wake.mark_camera_ready()

    def _on_vad(self, message: VadMessage) -> None:
        if self._narration.speaking or self._narration.awaiting_done:
            # PRD 16: no barge-in. The browser also suppresses VAD during
            # playback, but narration audio is not coordinator speech, so the
            # core enforces the same rule on its own side.
            LOGGER.debug("ignoring VAD start during narration playback")
            return
        self._conversation.on_vad_start(message.t)

    def _on_tts_done(self, message: TtsDoneMessage) -> None:
        if self._narration.awaiting_done:
            self._narration.on_tts_done(message.t)
            return
        self._conversation.on_tts_done(message.t)

    # ------------------------------------------------------ protocol outbound

    def _send_cue(self, message: CueMessage) -> None:
        """Best-effort SFX cue. A lost cue is cosmetic and never stops a plan."""

        try:
            self._protocol.publish_event(message)
        except Exception as error:
            with self._lock:
                self._cue_failures += 1
            LOGGER.debug("cue %s not delivered (%s)", message.sfx, type(error).__name__)

    def _send_capture_frame(self, message: CaptureFrameMessage) -> None:
        """The one and only outbound ``capture_frame``.

        Failure is deliberately allowed to propagate: ``ObservationRuntime``
        treats a refused dispatch as a fault and clears both sides, which is
        correct, because a frame that was never requested will never arrive.
        """

        self._protocol.publish_event(message)
        with self._lock:
            self._capture_frames_sent += 1

    def _send_speak(self, message: SpeakBeginMessage | SpeakEndMessage) -> None:
        self._protocol.publish_event(message)

    # ------------------------------------------------------- wiring callbacks

    def _on_router_capture(self, message: CaptureFrameMessage) -> None:
        """Arming signal from the router. This must never touch the socket."""

        with self._lock:
            self._router_capture_signals += 1
            self._capture_armed_id = message.req_id

    def _on_narration(self, narration: MissingNarration) -> None:
        """Route the observation runtime's narration into the speech path."""

        say = narration.response.say
        if not say.strip():
            return
        if self._conversation.speaking:
            # Dialogue speech is already using the browser's one audio graph.
            LOGGER.info("narration dropped: dialogue speech is already playing")
            return
        self._narration.speak(say)

    def _emit_wake_action(self, action: Action) -> None:
        """Submit one PRD 10.4 waking beat through the normal plan path.

        The waking sequence uses the same executor and router as a model plan,
        so its ``whirr_short`` reaches the browser through the ordinary cue
        path and its postures obey the same output stage. Nothing about the
        wake sequence bypasses the body.
        """

        self._plans.submit((action,))

    def _baseline_labels(self) -> tuple[str, ...]:
        """Cheap tick-safe baseline: the mirrored memory, never a disk load."""

        return tuple(
            record.canonical for record in self._blackboard.snapshot().scene_memory
        )

    def _resolve_look_target(
        self, name: str, snapshot: BlackboardSnapshot
    ) -> LookAtTarget | None:
        """Turn a semantic look-at name into a local geometry fact.

        The resolver publishes measurement, never intent: ``person`` reuses the
        azimuth and elevation the browser already derived from the face
        centroid, and ``obj:<id>`` applies the identical projection to the
        object's stored bounding box.
        """

        if name == "person":
            gaze = snapshot.gaze
            if not gaze.present:
                return None
            return LookAtTarget("person", float(gaze.az), float(gaze.el))
        if name == "scene":
            return LookAtTarget("scene", 0.0, SCENE_LOOK_ELEVATION_RAD)
        if not name.startswith("obj:"):
            return None
        object_id = name.removeprefix("obj:")
        for record in snapshot.scene_memory:
            if record.id != object_id:
                continue
            x, y, width, height = record.bbox_norm
            with self._lock:
                camera = self._camera
            azimuth, elevation = target_angles(x + width / 2.0, y + height / 2.0, camera)
            return LookAtTarget(name, azimuth, elevation)
        return None

    # ---------------------------------------------------------- 10 Hz domain

    def tick_behavior(self, now: float) -> None:
        """Run one 10 Hz beat in the order both merged modules document.

        ``BehaviorFSM.tick`` -> ``ConversationCoordinator.tick`` ->
        ``ActionRouter.tick`` -> ``ObservationRuntime.tick``. The router ticks
        the plan executor internally; this method never does.
        """

        snapshot = self._blackboard.snapshot()

        transition = self._fsm.tick(snapshot, now)
        if transition is not None:
            self._apply_transition(transition)

        self._conversation.tick()

        # The coordinator may have submitted a plan and re-mirrored its depth,
        # so the router must route against the fresh view, not the stale one.
        snapshot = self._blackboard.snapshot()
        if self._fsm.state not in PLAN_HELD_STATES:
            with self._director_lock:
                try:
                    self._router.tick(snapshot, now)
                except Exception:
                    # The director rejected the action. Dropping the plan is
                    # the only way to avoid a blocker nobody will release.
                    LOGGER.exception("action routing failed; cancelling the plan")
                    self._router.cancel("routing_failed")

        self._tick_observations(now)
        self._post_plan_drained()
        self._publish_telemetry(now)

        with self._lock:
            self._behavior_ticks += 1
            self._state_name = self._fsm.state
            self._arousal = AROUSAL_BY_STATE.get(self._state_name, 0.15)

    def _apply_transition(self, transition: Transition) -> None:
        """Apply one state beat to the body, cancelling everything it must.

        On a dormant or disengaging beat the plan, the observation blocker, and
        the speech staging must all go together. Each step is guarded
        independently so that no single failure can strand one of the others.
        """

        cancelling = transition.current in CANCELLING_STATES

        def apply_to_body() -> None:
            with self._director_lock:
                self._router.apply_transition(transition)

        def cancel_plan() -> None:
            with self._director_lock:
                self._router.cancel(transition.reason)

        steps: list[tuple[str, Callable[[], object]]] = [("director", apply_to_body)]
        if cancelling:
            steps += [
                ("conversation", self._conversation.disengage),
                ("narration", self._narration.disengage),
                ("observation", self._observations.disengage),
                ("plan", cancel_plan),
            ]
        for name, step in steps:
            try:
                step()
            except Exception:
                LOGGER.exception("transition step %s failed for %s", name, transition.reason)
        if cancelling:
            with self._lock:
                self._capture_armed_id = None
                self._pending_blocker_id = None
                self._pending_blocker_since = None

    def _tick_observations(self, now: float) -> None:
        """Tick the observation runtime, deferring only an un-armed blocker."""

        if self._should_tick_observations(now):
            self._observations.tick()

    def _should_tick_observations(self, now: float) -> bool:
        if self._observations.stage is not ObservationStage.IDLE:
            # Work is in flight: the runtime must drain its own completions.
            return True
        pending = self._plans.pending_observation_id
        with self._lock:
            if pending is None:
                self._pending_blocker_id = None
                self._pending_blocker_since = None
                return True
            if self._pending_blocker_id != pending:
                self._pending_blocker_id = pending
                self._pending_blocker_since = now
            if self._capture_armed_id == pending:
                return True
            since = self._pending_blocker_since
            if since is not None and now - since >= OBSERVATION_ARM_TIMEOUT_S:
                self._observation_arm_timeouts += 1
                LOGGER.warning(
                    "observation %s armed by watchdog after %.2f s without a "
                    "director capture beat",
                    pending,
                    now - since,
                )
                self._capture_armed_id = pending
                return True
        return False

    def _post_plan_drained(self) -> None:
        """Announce a drained plan; the FSM alone decides what that means."""

        if self._fsm.state is not BehaviorState.ACTING:
            return
        if self._plans.depth:
            return
        self._fsm.post_event(BehaviorEvent.PLAN_DRAINED)

    def _publish_telemetry(self, now: float) -> None:
        del now
        snapshot = self._blackboard.snapshot()
        sample = self._last_sample
        clamps = sample.clamps if sample is not None else snapshot.telemetry.clamps
        telemetry = Telemetry(
            state=self._fsm.state.value,
            plan_depth=len(snapshot.plan_queue),
            memory_count=len(snapshot.scene_memory),
            last_latency_ms=0.0 if self._latency is None else self._latency.last_latency_ms,
            clamps=clamps,
        )
        if telemetry != snapshot.telemetry:
            self._blackboard.set_telemetry(telemetry)

    def _behavior_loop(self) -> None:
        interval = 1.0 / BEHAVIOR_HZ
        next_tick = self._monotonic()
        while not self._stopping.is_set():
            try:
                self.tick_behavior(self._behavior_time())
            except Exception:
                LOGGER.exception("behaviour tick failed")
            next_tick += interval
            delay = next_tick - self._monotonic()
            if delay < -interval:
                next_tick = self._monotonic()
                delay = 0.0
            if delay > 0.0:
                self._sleep(delay)

    def _behavior_time(self) -> float:
        """Epoch time that never moves backwards for the FSM or the executor.

        ``BehaviorFSM.tick`` and ``PlanExecutor.tick`` both raise on a backward
        timestamp, and ``time.time()`` can step back under NTP. A small step
        back is clamped so the character keeps ticking; a large one is treated
        as a genuine clock correction and adopted.
        """

        now = self._clock()
        with self._lock:
            previous = self._last_behavior_time
            if previous is not None and now < previous:
                if previous - now <= MAX_TICK_BACKSTEP_S:
                    now = previous
                else:
                    LOGGER.warning("wall clock stepped back %.3f s", previous - now)
            self._last_behavior_time = now
        return now

    # --------------------------------------------------------- 120 Hz domain

    def tick_animation(self) -> AnimationSample | None:
        """Advance one 120 Hz step and publish ``body_state`` on every second.

        This is the tick that must never block on I/O. It reads one blackboard
        snapshot, looks up two pure speech envelopes, runs the director's fixed
        step, and swaps an immutable body-state snapshot into the transport.
        Nothing here serializes, opens a file, touches the event loop, or takes
        the router, FSM, or conversation staging lock.
        """

        speaking = self._conversation.speaking or self._narration.speaking
        snapshot = self._blackboard.snapshot()

        with self._director_lock:
            with self._lock:
                if self._animation_anchor is None:
                    self._animation_anchor = self._clock()
                    self._animation_step = 0
                now = self._animation_anchor + self._animation_step * FIXED_DT

            if speaking:
                amplitude = max(
                    self._conversation.speech_amplitude(now),
                    self._narration.speech_amplitude(now),
                )
                self._animation.set_speech_amplitude(amplitude)
            else:
                self._animation.clear_speech_amplitude()

            try:
                sample = self._director.tick(snapshot, now)
            except AnimationDiscontinuityError:
                LOGGER.warning(
                    "animation discontinuity; re-anchoring the fixed timeline"
                )
                self._director.reset()
                with self._lock:
                    self._animation_anchor = self._clock()
                    self._animation_step = 0
                return None

            with self._lock:
                self._animation_step += 1
                self._animation_ticks += 1
                self._last_sample = sample
                publish = self._animation_step % BODY_STATE_EVERY_N_TICKS == 0

        if publish:
            self._publish_body_state(sample, snapshot, now, speaking=speaking)
        return sample

    def _publish_body_state(
        self,
        sample: AnimationSample,
        snapshot: BlackboardSnapshot,
        now: float,
        *,
        speaking: bool,
    ) -> None:
        """Swap one immutable 60 Hz snapshot into the transport, nothing more.

        ``speaking`` is the value the caller already read for the speech-bob
        amplitude. Reusing it keeps the published flag and the head bob
        describing the same instant, and halves the lock traffic.
        """

        gaze = snapshot.gaze
        with self._lock:
            sequence = self._body_states
            self._body_states += 1
            state_name = self._state_name
            arousal = self._arousal
        state = BodyStateMessage(
            t=now,
            seq=sequence,
            joints=JointsState(
                base_yaw=sample.joints.base_yaw,
                shoulder_pitch=sample.joints.shoulder_pitch,
                elbow_pitch=sample.joints.elbow_pitch,
                neck_yaw=sample.joints.neck_yaw,
                head_pitch=sample.joints.head_pitch,
            ),
            light=LightState(
                intensity=sample.light.intensity,
                color_k=int(sample.light.color_k),
                pattern=sample.light.pattern,  # type: ignore[arg-type]
                bloom=sample.light.bloom,
            ),
            audio=AudioState(speaking=speaking, arousal=arousal),
            telemetry=TelemetryState(
                state=state_name.value,  # type: ignore[arg-type]
                plan_depth=len(snapshot.plan_queue),
                memory_count=len(snapshot.scene_memory),
                last_latency_ms=(
                    0.0 if self._latency is None else self._latency.last_latency_ms
                ),
                clamps=WireClampCounts(
                    vel=sample.clamps.velocity, limit=sample.clamps.limit
                ),
                gaze=TelemetryGaze(
                    present=bool(gaze.present),
                    yaw_deg=float(gaze.yaw_deg),
                    pitch_deg=float(gaze.pitch_deg),
                ),
            ),
        )
        self._protocol.publish_body_state(state)

    def _animation_loop(self) -> None:
        """Pace the fixed timeline with a monotonic clock, never an epoch one."""

        interval = FIXED_DT
        next_tick = self._monotonic()
        while not self._stopping.is_set():
            try:
                self.tick_animation()
            except Exception:
                LOGGER.exception("animation tick failed")
            next_tick += interval
            delay = next_tick - self._monotonic()
            if delay < -interval * 4.0:
                # Far behind: adopt the present rather than burning catch-up
                # ticks that would only make the backlog worse.
                next_tick = self._monotonic()
                delay = 0.0
            if delay > 0.0:
                self._sleep(delay)


def _check_rates(config: FrozenConfig) -> None:
    """Fail loudly if config and the compiled animation rate ever disagree."""

    expected = (
        ("fsm_hz", BEHAVIOR_HZ),
        ("animation_hz", ANIMATION_HZ),
        ("body_state_hz", BODY_STATE_HZ),
    )
    for key, value in expected:
        configured = float(config.runtime[key])
        if configured != value:
            raise ValueError(
                f"runtime.{key} is {configured} but the core runs at {value}"
            )


def build_app(**kwargs: object) -> LuxoApp:
    """Thin factory kept so ``core.main`` never reaches for private names."""

    return LuxoApp(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ANIMATION_HZ",
    "AROUSAL_BY_STATE",
    "AppStatus",
    "BEHAVIOR_HZ",
    "BODY_STATE_EVERY_N_TICKS",
    "BODY_STATE_HZ",
    "CameraGeometry",
    "LatencyRecorder",
    "LuxoApp",
    "NarrationSpeaker",
    "OBSERVATION_ARM_TIMEOUT_S",
    "PLAN_HELD_STATES",
    "ProtocolPort",
    "build_app",
    "target_angles",
]
