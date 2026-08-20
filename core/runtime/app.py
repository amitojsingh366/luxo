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

The demo reset (PRD 13.1)
-------------------------

``reset.py`` owns the between-takes reset and documents its own guarantees.
Two things about it belong here, because they are wiring decisions:

* :meth:`LuxoApp.request_reset` only records the request. The whole reset is
  applied by :meth:`LuxoApp.tick_behavior`, at the very top of the beat and
  before the blackboard snapshot is taken, so the rest of that tick runs on the
  new baseline instead of on a mixture of both. This is the same discipline the
  conversation coordinator uses for worker completions, and it is what makes
  the request path safe to call from a POSIX signal handler.
* Nothing about the reset reaches the transport. It publishes no message, adds
  no message type, and never touches ``_hellos``, ``_camera``, or
  ``_camera_ready``, so the connected browser keeps its page, its socket, and
  its camera and microphone permission across as many resets as a shoot needs.

Speech ownership — one speaker, one ``tts_done``
------------------------------------------------

Two things in this process produce a line to say: a brain reply, and the
observation runtime's narration of what went missing from the desk (PRD 8.3).
They are **not** two speakers. ``ConversationCoordinator.speak`` stages an
unprompted line into the very slot a reply occupies, so both reach the browser
through one synthesis, one wire validation, one raw prefix-free PCM delivery,
one ``speak_begin``/``speak_end`` pair, and one bounded retry.

That matters here rather than there because this module is where a second owner
would have to be built. It briefly had one. The cost of two owners is not
duplicated code, it is a **split fact**: with two ``speaking`` flags,
``body_state`` has to union them and an inbound ``tts_done`` has to be *routed*
— and a router is a thing that can route wrong. A ``tts_done`` that lands on the
component that was not speaking clears nothing and the FSM waits in ``SPEAKING``
for a completion that has already been consumed. So:

* ``body_state.audio.speaking`` is ``ConversationCoordinator.speaking``. Not a
  union, not a maximum, not a fallback — the one field.
* :meth:`LuxoApp._on_tts_done` calls ``ConversationCoordinator.on_tts_done`` and
  nothing else. There is no branch to get wrong.
* :meth:`LuxoApp._on_vad` no longer polices barge-in itself. ``on_vad_start``
  already refuses whenever a line is staged or sounding, whoever authored it, so
  the PRD 16 rule is enforced by the owner instead of alongside it.
* :class:`UnpromptedSpeech` is a **read-only projection** of that one owner, for
  diagnostics. It has no state and no mutator; that is what makes it a view
  rather than a second owner.

A narration is refused, never queued: see :meth:`LuxoApp._on_narration`.

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

:meth:`tick_animation` calls only: a blackboard snapshot, one speech-envelope
lookup, the director's fixed step, and ``publish_body_state``. It never
serializes a message, never touches the event loop, never opens a file, never
submits a worker job, and never calls a model.

Being precise about the locks it does take, because "no I/O" is not the same as
"no waiting":

* the **director lock**, held for the fixed step, contended only by the 10 Hz
  behaviour tick doing pure in-memory routing;
* the **blackboard lock**, for one snapshot;
* the **app state lock**, for counters and the cached state name;
* the **conversation lock**, for ``speaking`` and ``speech_amplitude``. Each
  reads two fields and returns; the speech worker holds that lock only across a
  chunk enqueue, never across Piper synthesis, which happens outside it.

Every one of those is a bounded in-memory critical section. It never takes the
router, FSM, observation, or plan-executor lock at all. Telemetry it needs from
the behaviour domain (state name, arousal, last latency) is cached as a plain
attribute by the domain that owns it, so the 120 Hz tick never waits on the
10 Hz one to read it.

One lock edge elsewhere is worth naming, because it is not visible from the tick
order. ``ObservationRuntime`` holds its own lock while it invokes
``narration_callback``, so :meth:`LuxoApp._on_narration` takes the conversation
lock *under* the observation lock. That is safe in exactly one direction, and it
is the direction that holds: the coordinator, under its own lock, reaches only
the FSM, the blackboard, the plan executor, and the outbound callbacks. It never
reaches the observation runtime, so the two can never close a cycle.
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
from ..speech.tts import TextToSpeech
from ..wake_sequence import WakeSequenceCoordinator
from .actions import CANCELLING_STATES, ActionRouter
from .interactions import ConversationCoordinator, Stage
from .observations import ObservationRuntime, ObservationStage
from .reset import DemoReset, ResetReport

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
    """Immutable diagnostic view; carries no transcript, image, or plan text.

    ``narrations_spoken`` counts lines this module handed to
    ``ConversationCoordinator.speak`` and that it accepted — staged for speech,
    not confirmed played. Completion is the browser's to declare and lands on
    the coordinator, which is where to read it from; counting it here would
    mean tracking a line the coordinator already owns.
    ``narrations_dropped`` counts the rest: blank lines, and lines refused
    because the one speech slot was in use. Together they are every narration
    that reached :meth:`LuxoApp._on_narration`, so a line can never vanish
    unaccounted for.
    """

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
    narrations_dropped: int
    resets_applied: int
    last_latency_ms: float
    running: bool


class UnpromptedSpeech:
    """Read-only view of the one unprompted line the coordinator is holding.

    Narration used to be spoken by a second component living here, which meant
    two owners of ``speaking`` and an inbound ``tts_done`` that had to be routed
    between them. ``ConversationCoordinator.speak`` removed the need for that:
    an unprompted line now occupies the same staging slot as a brain reply.

    What is left is this — a projection, not a participant. It reads the
    coordinator's own status and reports the part of it that is about an
    unprompted line, for diagnostics and for the demo-reset checks that assert a
    narration is not left sounding. It holds no state, synthesizes nothing,
    owns no worker, and deliberately exposes **no mutator at all**: there is no
    ``speak``, no ``on_tts_done``, no ``reset``. That absence is the proof it
    cannot become a second owner by accident.
    """

    __slots__ = ("_conversation",)

    _AWAITING: Final = frozenset({Stage.AWAITING_DONE, Stage.STALLED})

    def __init__(self, conversation: ConversationCoordinator) -> None:
        self._conversation = conversation

    @property
    def speaking(self) -> bool:
        """True while the line the browser is playing is an unprompted one.

        Narrower than ``ConversationCoordinator.speaking``, and deliberately so:
        that property is the ``body_state`` fact and covers every line, whoever
        authored it. This one answers the different question "is a *narration*
        sounding", which is only ever asked by diagnostics.
        """

        status = self._conversation.status
        return status.speaking and status.unprompted

    @property
    def awaiting_done(self) -> bool:
        """True from a delivered unprompted line until the browser reports it.

        ``STALLED`` counts: a partial delivery already put audio in front of the
        listener and is waiting on the same browser ``tts_done``.
        """

        status = self._conversation.status
        return status.unprompted and status.stage in self._AWAITING


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
        warm_executor: Executor | None = None,
        latency_executor: Executor | None = None,
        reset_executor: Executor | None = None,
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

        # A projection of the coordinator, not a second speaker. Narration and
        # dialogue share one staging slot, one worker pool, and one ``speaking``.
        self._narration = UnpromptedSpeech(self._conversation)

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
        self._narrations_spoken = 0
        self._narrations_dropped = 0
        self._cue_failures = 0

        self._camera = CameraGeometry()
        self._hellos = 0
        self._camera_ready = False

        # Read by the animation thread; written by the behaviour thread. Plain
        # attribute assignment keeps the 120 Hz tick off every other lock.
        self._state_name = BehaviorState.BOOT
        self._arousal = AROUSAL_BY_STATE[BehaviorState.BOOT]

        # The between-takes reset. It is handed the same director lock every
        # other entry into the animation body takes, so its body step is
        # serialized against the 120 Hz tick like any other.
        #
        # ``conversation`` and ``narration`` are deliberately the same object.
        # ``reset.py`` names them as two steps because there were once two
        # components; there is now one, and the collapse is stated here rather
        # than by quietly dropping a step the module still declares in
        # ``STEP_NAMES``. ``ConversationCoordinator.reset`` is idempotent —
        # it invalidates the generation, clears the staged line, the envelope
        # and the ``speaking`` flag, and forgets recent turns — so running it
        # twice in the sequence clears the same baseline twice and cannot
        # regress the coverage either step used to provide.
        self._reset = DemoReset(
            fsm=self._fsm,
            blackboard=self._blackboard,
            conversation=self._conversation,
            narration=self._conversation,
            observations=self._observations,
            router=self._router,
            director_lock=self._director_lock,
            memory_store=self._memory,
            on_cleared=self._on_reset_cleared,
            executor=reset_executor,
        )

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
    def narration(self) -> UnpromptedSpeech:
        """Diagnostics only. A view of :attr:`conversation`, not a component."""

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

    @property
    def demo_reset(self) -> DemoReset:
        return self._reset

    def status(self) -> AppStatus:
        snapshot = self._blackboard.snapshot()
        state = self._fsm.state
        resets = self._reset.status.applied
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
                narrations_spoken=self._narrations_spoken,
                narrations_dropped=self._narrations_dropped,
                resets_applied=resets,
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
            ("reset", self._reset.close),
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

    def request_reset(self, reason: str = "reset") -> None:
        """Ask for a between-takes reset (PRD 13.1). Records only; does nothing.

        Safe to call from a POSIX signal handler running between bytecodes on
        the main thread, from the transport thread, or from a check. The reset
        itself is applied by the next :meth:`tick_behavior`, which is the only
        thread allowed to advance any stage.
        """

        self._reset.request(reason)

    def _drain_reset(self, now: float) -> ResetReport | None:
        """Apply a requested reset before anything else in the beat."""

        try:
            return self._reset.drain(now)
        except Exception:
            # A reset that raises must never take the behaviour tick with it;
            # the character keeps running on whatever state survived.
            LOGGER.exception("the demo reset failed and was abandoned")
            return None

    def _on_reset_cleared(self) -> None:
        """Clear the runtime's own arming state as the last reset step.

        These four are the only pieces of the observation handshake this module
        owns rather than delegates. ``_capture_armed_id`` in particular must go
        with the plan it belonged to: a stale arming signal that outlived its
        blocker would let the next blocker be ticked before the director's
        capture beat, which is exactly the deferral the watchdog exists to
        bound. Every counter above them is cumulative telemetry and is kept.
        """

        with self._lock:
            self._capture_armed_id = None
            self._pending_blocker_id = None
            self._pending_blocker_since = None
            self._state_name = BehaviorState.DORMANT
            self._arousal = AROUSAL_BY_STATE[BehaviorState.DORMANT]

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
            # asked for. Clear both so a reconnect cannot strand either. One
            # conversation reset covers a narration too: it is the same slot.
            self._conversation.reset()
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
        """Hand a VAD start straight to the one owner of the staging slot.

        PRD 16 forbids barge-in, and this method used to enforce that for
        narration on its own because narration audio was not coordinator
        speech. It is now, so ``on_vad_start`` refuses on its own whenever a
        line is staged or sounding, whoever authored it. Keeping a second check
        here would be a copy of a rule the owner already applies, and copies of
        rules drift.
        """

        self._conversation.on_vad_start(message.t)

    def _on_tts_done(self, message: TtsDoneMessage) -> None:
        """The browser's completion has exactly one destination and no branch.

        This is the whole point of collapsing the two speakers. A routed
        ``tts_done`` is a ``tts_done`` that can be routed to the component that
        was not speaking, which consumes the browser's only completion signal
        and leaves the FSM waiting in ``SPEAKING`` for one that never comes.
        """

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
        """Stage the observation runtime's narration as an unprompted line.

        ``ConversationCoordinator.speak`` is the whole hop: the line is put in
        the same slot a brain reply occupies and the next behaviour tick
        dispatches it through the same worker, synthesis, wire validation, raw
        prefix-free PCM delivery and ``speak_begin``/``speak_end`` pair. Nothing
        about narration reaches the transport from here.

        **A blank line is dropped before it reaches the coordinator.** That is
        not a redundant guard: ``PlanResponse`` caps ``say`` at a length but
        does not require it to be non-empty, and ``speak`` substitutes the
        in-character *dialogue* fallback for a blank line, which is right for a
        reply that failed and wrong for a narration — the lamp would answer an
        observation nobody prompted with "my thoughts got tangled".

        **A refused narration is dropped with a log, never retried or queued.**
        Three reasons, in order of weight:

        * Nothing is stranded by dropping it. ``ObservationRuntime`` releases
          the plan blocker before it narrates, so the line is the entire cost;
          the rest of the plan runs either way.
        * Retrying needs somewhere to hold the line until the slot frees, and
          that place is a second staging slot — which is a second owner of the
          spoken line under another name, and is exactly what this wiring
          exists to remove.
        * A narration is a remark about a moment. Deferring it behind a reply
          means saying "your keys wandered off" after the conversation has
          moved on, which is worse than not saying it.

        The refusal is counted rather than swallowed, so ``status()`` can still
        show that a line was lost.
        """

        say = narration.response.say
        if not say.strip():
            LOGGER.debug("narration dropped: the model returned a blank line")
            with self._lock:
                self._narrations_dropped += 1
            return
        if self._conversation.speak(say):
            with self._lock:
                self._narrations_spoken += 1
            return
        with self._lock:
            self._narrations_dropped += 1
        LOGGER.info("narration dropped: the speech slot is already in use")

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

        A requested demo reset is drained before any of that, and before the
        snapshot is read, so the whole beat runs on exactly one baseline.
        """

        self._drain_reset(now)
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
                # One disengage covers dialogue and narration alike: they are
                # the same staged line in the same component.
                ("conversation", self._conversation.disengage),
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
        snapshot, looks up one pure speech envelope, runs the director's fixed
        step, and swaps an immutable body-state snapshot into the transport.
        Nothing here serializes, opens a file, touches the event loop, or takes
        the router, FSM, or conversation staging lock.

        ``speaking`` is read from the coordinator and from nowhere else. It is
        one field, not a union of two, because there is one component that can
        be speaking. The head bob and the published flag therefore cannot
        disagree, and neither can outlive the other.
        """

        speaking = self._conversation.speaking
        snapshot = self._blackboard.snapshot()

        with self._director_lock:
            with self._lock:
                if self._animation_anchor is None:
                    self._animation_anchor = self._clock()
                    self._animation_step = 0
                now = self._animation_anchor + self._animation_step * FIXED_DT

            if speaking:
                self._animation.set_speech_amplitude(
                    self._conversation.speech_amplitude(now)
                )
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
    "OBSERVATION_ARM_TIMEOUT_S",
    "PLAN_HELD_STATES",
    "ProtocolPort",
    "UnpromptedSpeech",
    "build_app",
    "target_angles",
]
