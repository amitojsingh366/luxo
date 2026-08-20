"""Nonblocking conversation staging: VAD/PCM to STT to brain to speech.

The owning runtime must tick exactly this order at 10 Hz::

    BehaviorFSM.tick() -> ConversationCoordinator.tick() -> PlanExecutor.tick()

Inference, network, and synthesis work runs only on worker threads. A worker
completion callback may run inline on the thread that registered it, so it is
allowed to do exactly one thing: enqueue a generation-tagged completion.
Transcript publication, plan submission, blackboard mirroring, and FSM result
events all happen on the serialized tick, which is why the FSM observes each
stage exactly one tick after the worker finished it.

Speech is different: complete synthesis, validation, raw PCM delivery, and the
``speak_begin``/``speak_end`` callbacks stay on a worker. The delivery callback
receives raw, even-sized PCM chunks of at most 8192 bytes with no frame prefix,
because ``ProtocolServer.publish_tts_pcm`` is the sole owner of the ``0x03``
binary prefix (PRD 8.4, 10.2).

``SPEECH_DONE`` is never fabricated. Only a browser ``tts_done`` posts it. A
delivery that never reached ``speak_begin`` retries a bounded number of times
and then waits for explicit recovery; a partial delivery sends one best-effort
``speak_end`` and then waits for the browser, a reset, or a disengage.

A line usually arrives as a brain reply, but :meth:`ConversationCoordinator.speak`
stages an *unprompted* one — an observation narration, say — into that same
machinery instead of standing up a parallel speaker. The coordinator therefore
stays the single owner of the ``speaking`` fact and the single destination for
``tts_done``. An unprompted line differs from a reply in exactly two ways, both
because it is not a dialogue turn: it drives no FSM transition, because the
character never entered ``SPEAKING`` for it and leaving a state it never entered
would be an invented transition, and it records no latency milestone, because
PRD 11.1 times the VAD-to-audio dialogue path. Synthesis, wire validation, raw
PCM delivery, the ``speaking`` fact, the envelope, the bounded retry, and the
browser's sole authority over completion are all shared, not duplicated.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final

from ..blackboard import Blackboard, UtteranceFact
from ..brain.client import (
    BrainClient,
    ConversationUnavailableError,
    ObservationOrigin,
    RecentExchange,
)
from ..brain.memory import CloudSceneObject
from ..brain.schema import Action, ActionOp, PlanResponse
from ..fsm import BehaviorEvent, BehaviorFSM, BehaviorState
from ..instrumentation import Milestone
from ..plan_executor import PlanExecutor
from ..protocol.messages import SpeakBeginMessage, SpeakEndMessage
from ..speech.stt import SAMPLE_HZ as STT_SAMPLE_HZ
from ..speech.stt import SpeechToText, Transcript
from ..speech.tts import DEFAULT_CHUNK_BYTES, ENVELOPE_HZ
from ..speech.tts import SAMPLE_HZ as TTS_SAMPLE_HZ
from ..speech.tts import SpeechAudio, TextToSpeech

LOGGER = logging.getLogger(__name__)

# Deterministic local wording for conversation and speech delivery failures.
FALLBACK_SAY: Final = "Oops—my thoughts got tangled! Can we try that again?"
MAX_RECENT: Final = 3
MAX_SPEECH_ATTEMPTS: Final = 3
SPEECH_RETRY_BACKOFF_S: Final = 0.25
DEFAULT_WORKERS: Final = 2

SpeakCallback = Callable[[SpeakBeginMessage | SpeakEndMessage], None]
PcmCallback = Callable[[bytes], None]
MilestoneCallback = Callable[[Milestone, float], None]
ObservationOriginCallback = Callable[
    [ObservationOrigin, tuple[RecentExchange, ...]], bool
]


def _log_text(value: str) -> str:
    """Keep terminal logs readable without allowing control characters."""

    return "".join(character if character.isprintable() else "�" for character in value)


def _plan_log(plan: tuple[Action, ...]) -> str:
    actions: list[dict[str, object]] = []
    for action in plan:
        item: dict[str, object] = {"op": action.op.value}
        for field in ("name", "target", "preset", "pattern", "arc", "speed", "ms"):
            value = getattr(action, field, None)
            if value is None:
                continue
            item[field] = value.value if isinstance(value, Enum) else value
        actions.append(item)
    return json.dumps(actions, ensure_ascii=False, separators=(",", ":"))


class Stage(str, Enum):
    """Position of the one in-flight interaction, owned by the coordinator."""

    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    REQUESTING = "requesting"
    REPLIED = "replied"
    DELIVERING = "delivering"
    AWAITING_DONE = "awaiting_done"
    STALLED = "stalled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CoordinatorStatus:
    """Immutable staging view carrying no audio, transcript, or plan payload."""

    generation: int
    stage: Stage
    speaking: bool
    unprompted: bool
    speech_attempts: int
    recent: tuple[RecentExchange, ...]
    last_error: str | None
    closed: bool


@dataclass(frozen=True, slots=True)
class _Completion:
    """One finished worker result, tagged with the generation that issued it."""

    generation: int
    kind: str
    future: Future[object]


@dataclass(frozen=True, slots=True)
class _Speech:
    envelope: tuple[float, ...]
    chunks: tuple[bytes, ...]


class _Stale(Exception):
    """Reset, disengage, or close won the race against a worker send."""


# A completion is only valid in the stage and behaviour state that issued it.
_GUARDS: Final = {
    "stt": (Stage.TRANSCRIBING, BehaviorState.LISTENING),
    "brain": (Stage.REQUESTING, BehaviorState.THINKING),
}
_INACTIVE: Final = frozenset(
    {BehaviorState.BOOT, BehaviorState.DORMANT, BehaviorState.DISENGAGING}
)


class ConversationCoordinator:
    """Stage one interaction across STT, brain, and speech workers.

    Every public method is safe to call from socket, worker, and tick threads.
    The outbound callbacks must be quick, non-blocking enqueues and must never
    call back into the coordinator: they are invoked while the coordinator lock
    is held, so that no reset or close can interleave between the generation
    check and the enqueue. Lock order is coordinator lock, then blackboard lock.
    """

    def __init__(
        self,
        *,
        blackboard: Blackboard,
        fsm: BehaviorFSM,
        plan_executor: PlanExecutor,
        stt: SpeechToText,
        brain: BrainClient,
        tts: TextToSpeech,
        compact_memory: Callable[[], str],
        currently_visible: Callable[[], tuple[CloudSceneObject, ...]],
        speak_callback: SpeakCallback,
        pcm_callback: PcmCallback,
        milestone_callback: MilestoneCallback | None = None,
        observation_origin_callback: ObservationOriginCallback | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        executor: Executor | None = None,
    ) -> None:
        callbacks = (
            compact_memory,
            currently_visible,
            speak_callback,
            pcm_callback,
            clock,
            sleep,
        )
        if not all(callable(item) for item in callbacks):
            raise TypeError("coordinator callbacks must be callable")
        if milestone_callback is not None and not callable(milestone_callback):
            raise TypeError("milestone_callback must be callable")
        if observation_origin_callback is not None and not callable(
            observation_origin_callback
        ):
            raise TypeError("observation_origin_callback must be callable")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must provide submit()")
        self._blackboard = blackboard
        self._fsm = fsm
        self._plans = plan_executor
        self._stt = stt
        self._brain = brain
        self._tts = tts
        self._compact_memory = compact_memory
        self._currently_visible = currently_visible
        self._speak_callback = speak_callback
        self._pcm_callback = pcm_callback
        self._milestone_callback = milestone_callback
        self._observation_origin_callback = observation_origin_callback
        self._clock = clock
        self._sleep = sleep
        self._executor = executor or ThreadPoolExecutor(
            max_workers=DEFAULT_WORKERS, thread_name_prefix="luxo-conversation"
        )
        self._owns_executor = executor is None
        self._lock = RLock()
        self._generation = 0
        self._closed = False
        self._stage = Stage.IDLE
        self._completions: list[_Completion] = []
        self._future: Future[object] | None = None
        self._vad_start: float | None = None
        self._transcript: str | None = None
        self._reply: str | None = None
        self._recent: deque[RecentExchange] = deque(maxlen=MAX_RECENT)
        self._envelope: tuple[float, ...] = ()
        self._speech_started_at: float | None = None
        self._speech_attempts = 0
        self._speaking = False
        self._unprompted = False
        self._last_error: str | None = None

    @property
    def status(self) -> CoordinatorStatus:
        with self._lock:
            return CoordinatorStatus(
                generation=self._generation,
                stage=self._stage,
                speaking=self._speaking,
                unprompted=self._unprompted,
                speech_attempts=self._speech_attempts,
                recent=tuple(self._recent),
                last_error=self._last_error,
                closed=self._closed,
            )

    @property
    def speaking(self) -> bool:
        """Report the ``body_state.audio.speaking`` fact for the 60 Hz writer."""

        with self._lock:
            return self._speaking

    @property
    def current_envelope(self) -> tuple[float, ...]:
        """Return the 50 Hz amplitude envelope of the utterance being spoken."""

        with self._lock:
            return self._envelope

    def speech_amplitude(self, now: float) -> float:
        """Look up the 50 Hz envelope; pure and safe on the animation tick."""

        instant = _time("now", now)
        with self._lock:
            started, envelope = self._speech_started_at, self._envelope
        if started is None or instant < started:
            return 0.0
        index = int((instant - started) * ENVELOPE_HZ)
        return envelope[index] if index < len(envelope) else 0.0

    def on_vad_start(self, t: float) -> bool:
        """Open a new interaction generation from the browser VAD start event."""

        instant = _time("t", t)
        with self._lock:
            if (
                self._closed
                or self._stage is not Stage.IDLE
                or self._speaking
                or self._vad_start is not None
                or self._fsm.state is not BehaviorState.ENGAGED
            ):
                return False
            self._generation += 1
            self._completions.clear()
            self._vad_start = instant
            self._last_error = None
            self._fsm.post_event(BehaviorEvent.VAD_START)
            return True

    def accept_utterance(self, pcm_int16: bytes, vad_end: float | None = None) -> bool:
        """Take one complete 16 kHz int16 utterance and submit it to STT.

        Both timing milestones are recorded before submission, so an instant
        STT completion can never publish ahead of its own VAD milestones.
        """

        pcm = _pcm(pcm_int16)
        with self._lock:
            if (
                self._closed
                or self._stage is not Stage.IDLE
                or self._vad_start is None
                or self._fsm.state is not BehaviorState.LISTENING
            ):
                return False
            end = self._now() if vad_end is None else _time("vad_end", vad_end)
            if end < self._vad_start:
                raise ValueError("vad_end must not precede the VAD start")
            self._emit(Milestone.VAD_END, end)
            self._emit(Milestone.PCM_RECEIVED, self._now())
            return self._start_locked("stt", self._stt.transcribe, pcm, STT_SAMPLE_HZ)

    def speak(self, text: str, *, dispatch: bool = False) -> bool:
        """Stage one line the brain never produced; returns whether it staged.

        This is the public entry point for narration and any other unprompted
        speech. It stages ``text`` into the very slot a brain reply occupies,
        so it can dispatch through the same worker, the same
        synthesis and wire validation, the same raw prefix-free PCM delivery,
        the same ``speak_begin``/``speak_end`` pair, and the same bounded
        retry. There is no second speaker and no second owner of ``speaking``,
        so a browser ``tts_done`` has exactly one place to land.

        A blank or whitespace-only line takes the in-character fallback rather
        than reaching Piper, which rejects empty text.

        ``dispatch=True`` submits the existing delivery worker before returning.
        Observation callbacks use that form because they run after this
        coordinator's once-per-beat tick; without it, a fresh visual line can
        sit staged until the next behaviour beat and be cleared by an intervening
        disengagement. Submission is still nonblocking and synthesis remains on
        the worker.

        **A line already in flight is refused, not queued and not superseded.**
        The browser has one audio graph, so speech is a serial resource: the
        only way to supersede a line the browser has already begun playing is
        to invent an end for audio that is still sounding, and this module's
        governing rule is that completion is the browser's to declare. Queueing
        would need a second staging slot, which is a second owner of the line
        by another name. Refusing keeps one line, one owner, one ``tts_done``,
        and hands the caller a truthful ``False`` to log or drop on.

        Unlike ``on_vad_start`` this does not require ``ENGAGED``; it only
        requires a state in which the tick would not invalidate the
        interaction, because narration is dispatched from ``ACTING`` and
        ``INSPECTING``. It leaves the plan queue alone — an observation has
        already submitted its plan by the time it narrates — and it does not
        append to ``recent``, because an unprompted line is not a dialogue
        turn and must not enter the brain's context as one.
        """

        say = _line(text)
        blank = not say
        with self._lock:
            if (
                self._closed
                or self._stage is not Stage.IDLE
                or self._speaking
                or self._fsm.state in _INACTIVE
            ):
                return False
            self._generation += 1
            self._completions.clear()
            self._reply = FALLBACK_SAY if blank else say
            self._unprompted = True
            self._stage = Stage.REPLIED
            self._last_error = "blank_say" if blank else None
            LOGGER.info("CHAT Luxo (scene): %s", _log_text(self._reply))
            if dispatch:
                self._dispatch_speech_locked()
            return True

    def stage_observation_response(
        self,
        origin: ObservationOrigin,
        response: PlanResponse,
    ) -> bool:
        """Reserve the single speech slot for one resolved observation turn.

        The observation runtime keeps its blocker until this succeeds. The FSM
        then moves INSPECTING -> SPEAKING and the ordinary coordinator tick
        dispatches the line, so it cannot be superseded by a new dialogue turn.
        """

        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        if not isinstance(response, PlanResponse):
            raise TypeError("response must be a PlanResponse")
        say = _line(response.say) or FALLBACK_SAY
        with self._lock:
            if (
                self._closed
                or self._stage is not Stage.IDLE
                or self._speaking
                or self._fsm.state is not BehaviorState.INSPECTING
            ):
                return False
            self._generation += 1
            self._completions.clear()
            self._reply = say
            self._unprompted = False
            self._stage = Stage.REPLIED
            self._last_error = "blank_say" if not response.say.strip() else None
            if origin.kind == "dialogue":
                for index in range(len(self._recent) - 1, -1, -1):
                    exchange = self._recent[index]
                    if exchange.human_say == origin.text:
                        self._recent[index] = RecentExchange(origin.text, say)
                        break
            LOGGER.info("CHAT Luxo (scene): %s", _log_text(say))
            return True

    def tick(self) -> None:
        """Drain completions and issue eligible work; never blocks on I/O."""

        with self._lock:
            if self._closed:
                return
            state = self._fsm.state
            completions, self._completions = self._completions, []
            if state in _INACTIVE:
                if self._stage is not Stage.IDLE or self._speaking:
                    self._invalidate_locked()
                return
            for completion in completions:
                self._apply_locked(completion, state)
            if self._stage is Stage.TRANSCRIBED and state is BehaviorState.THINKING:
                self._emit(Milestone.REQUEST_SENT, self._now())
                self._start_locked(
                    "brain", self._converse, self._transcript, tuple(self._recent)
                )
            elif self._stage is Stage.REPLIED and (
                self._unprompted or state is BehaviorState.SPEAKING
            ):
                # An unprompted line dispatches in whatever state the character
                # is in, because it never asked the FSM to enter SPEAKING. The
                # inactive states already returned above.
                self._dispatch_speech_locked()
            self._mirror_plan_locked()

    def _dispatch_speech_locked(self) -> None:
        """Submit the one staged line without running synthesis on the caller."""

        self._stage = Stage.DELIVERING
        try:
            self._future = self._executor.submit(
                self._deliver, self._generation, self._reply, self._unprompted
            )
        except Exception as error:
            # A refused worker is an unstarted delivery, so it lands in the
            # same explicit recovery path as an exhausted retry.
            self._stage = Stage.FAILED
            self._speech_attempts = MAX_SPEECH_ATTEMPTS
            self._last_error = type(error).__name__

    def on_tts_done(self, t: float) -> bool:
        """Accept the browser as the only authority for speech completion.

        An unprompted line releases the same staging slot but posts no event:
        the FSM was never moved into ``SPEAKING`` for it, so announcing that
        speech finished would push a transition the character never made.
        """

        instant = _time("t", t)
        with self._lock:
            started = self._speech_started_at
            unprompted = self._unprompted
            if (
                self._closed
                or self._stage not in (Stage.AWAITING_DONE, Stage.STALLED)
                or (not unprompted and self._fsm.state is not BehaviorState.SPEAKING)
                or (started is not None and instant < started)
            ):
                return False
            self._stage = Stage.IDLE
            self._reply = None
            self._vad_start = None
            self._speech_attempts = 0
            self._unprompted = False
            self._clear_speech_locked()
            if not unprompted:
                self._fsm.post_event(BehaviorEvent.SPEECH_DONE)
            return True

    def retry_speech(self) -> bool:
        """Re-arm a delivery that never reached ``speak_begin``; returns armed."""

        with self._lock:
            if self._closed or self._stage is not Stage.FAILED or self._reply is None:
                return False
            self._stage = Stage.REPLIED
            self._speech_attempts = 0
            self._last_error = None
            return True

    def disengage(self) -> None:
        """Abandon the interaction on gaze loss or an explicit runtime stop."""

        with self._lock:
            if not self._closed:
                self._invalidate_locked()

    def reset(self) -> None:
        """Abandon the interaction and forget recent turns; use on reconnect."""

        with self._lock:
            if self._closed:
                return
            self._invalidate_locked()
            self._recent.clear()
            self._last_error = None

    def close(self) -> None:
        """Make the coordinator permanently inert and release an owned pool."""

        with self._lock:
            if self._closed:
                return
            self._invalidate_locked()
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_locked(
        self, kind: str, work: Callable[..., object], *args: object
    ) -> bool:
        """Submit one worker job; a refused submission never reaches the tick."""

        generation = self._generation
        self._stage = Stage.TRANSCRIBING if kind == "stt" else Stage.REQUESTING
        try:
            future = self._executor.submit(work, *args)
        except Exception as error:
            self._stage = Stage.IDLE
            self._vad_start = None
            self._last_error = type(error).__name__
            return False
        self._future = future
        future.add_done_callback(
            lambda done, token=generation, label=kind: self._queue(token, label, done)
        )
        return True

    def _queue(self, generation: int, kind: str, future: Future[object]) -> None:
        """Enqueue only: this may run inline on the registering thread."""

        with self._lock:
            if not self._closed and generation == self._generation:
                self._completions.append(_Completion(generation, kind, future))

    def _apply_locked(self, completion: _Completion, state: BehaviorState) -> None:
        if completion.generation != self._generation:
            return
        if (self._stage, state) != _GUARDS[completion.kind]:
            self._stage = Stage.IDLE
            self._vad_start = None
            self._last_error = "out_of_order_completion"
            return
        self._future = None
        if completion.kind == "stt":
            self._apply_stt_locked(completion.future)
        else:
            self._apply_brain_locked(completion.future)

    def _apply_stt_locked(self, future: Future[object]) -> None:
        try:
            result = future.result()
            if not isinstance(result, Transcript) or not result.text.strip():
                raise TypeError("STT must return a non-empty Transcript")
        except Exception as error:
            # The FSM stays LISTENING: only gaze loss or a later transcript
            # may move it, because a transcript event must never be invented.
            self._stage = Stage.IDLE
            self._vad_start = None
            self._last_error = type(error).__name__
            self._blackboard.publish_utterance(None)
            return
        now = self._now()
        self._transcript = _line(result.text)
        LOGGER.info("CHAT human: %s", _log_text(self._transcript))
        self._stage = Stage.TRANSCRIBED
        self._blackboard.publish_utterance(UtteranceFact(now, self._transcript))
        self._fsm.post_event(BehaviorEvent.TRANSCRIPT_READY)
        self._emit(Milestone.TRANSCRIPT, now)

    def _apply_brain_locked(self, future: Future[object]) -> None:
        fallback = False
        try:
            reply = future.result()
            if not isinstance(reply, PlanResponse):
                raise TypeError("brain must return a PlanResponse")
        except ConversationUnavailableError as error:
            reply = PlanResponse(_conversation_fallback(error), ())
            fallback = True
            self._last_error = error.kind
        except Exception as error:
            reply = PlanResponse(FALLBACK_SAY, ())
            fallback = True
            self._last_error = type(error).__name__
        say = _line(reply.say)
        if not say:
            # Blank say is schema-valid but Piper-invalid, so it takes the
            # existing in-character fallback line and an empty plan.
            reply = PlanResponse(FALLBACK_SAY, ())
            say = FALLBACK_SAY
            fallback = True
            self._last_error = "blank_say"
        LOGGER.info("CHAT Luxo: %s", _log_text(say))
        LOGGER.info("CHAT plan: %s", _plan_log(reply.plan))
        observes = bool(reply.plan and reply.plan[-1].op is ActionOp.OBSERVE)
        if observes and self._observation_origin_callback is not None:
            origin = ObservationOrigin("dialogue", self._transcript or "")
            try:
                bound = self._observation_origin_callback(origin, tuple(self._recent))
            except Exception:
                LOGGER.exception("observation origin callback failed")
                bound = False
            if not bound:
                # An unbound capture could resolve against a stale turn. Drop
                # the terminal observe while preserving the cloud-authored say
                # and all non-capture actions.
                reply = PlanResponse(reply.say, reply.plan[:-1])
        now = self._now()
        with self._blackboard.lock:
            self._plans.submit(reply.plan)
            self._mirror_plan_locked()
            self._blackboard.publish_utterance(None)
        self._fsm.post_event(
            BehaviorEvent.MODEL_FALLBACK if fallback else BehaviorEvent.MODEL_RESPONSE
        )
        self._recent.append(RecentExchange(self._transcript or "", say))
        self._transcript = None
        self._reply = say
        self._unprompted = False
        self._stage = Stage.REPLIED
        self._emit(Milestone.RESPONSE, now)

    def _mirror_plan_locked(self) -> None:
        """Mirror executor depth onto the blackboard inside the same tick.

        This runs with a plan submission and again before every
        ``PlanExecutor.tick()``, holding the blackboard lock across the read
        and the write, so a reader that takes only the blackboard lock can
        never see the two disagree. A blocked action stays counted, so an
        empty mirror always means a genuinely drained plan.
        """

        with self._blackboard.lock:
            executor_state = self._plans.state
            queued = executor_state.queued_actions
            if executor_state.active_blocker is not None:
                queued = (executor_state.active_blocker,) + queued
            if queued != self._blackboard.snapshot().plan_queue:
                self._blackboard.set_plan(queued)

    def _converse(
        self, transcript: str, recent: tuple[RecentExchange, ...]
    ) -> PlanResponse:
        """Worker-side brain call carrying only the minimal PRD 8.1.1 payload."""

        memory = self._compact_memory()
        if not isinstance(memory, str) or "\n" in memory or "\r" in memory:
            raise ValueError("compact memory must be a single line")
        context = [
            {
                "human": _log_text(exchange.human_say),
                "luxo": _log_text(exchange.lamp_say),
            }
            for exchange in recent[-MAX_RECENT:]
        ]
        LOGGER.info(
            "BRAIN request current=%s memory=%s recent=%s",
            json.dumps(_log_text(transcript), ensure_ascii=False),
            json.dumps(_log_text(memory), ensure_ascii=False),
            json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        )
        visible = self._currently_visible()
        if not isinstance(visible, tuple) or not all(
            isinstance(item, CloudSceneObject) for item in visible
        ):
            raise TypeError("currently_visible must return CloudSceneObject values")
        return self._brain.converse(
            transcript,
            memory,
            visible,
            recent[-MAX_RECENT:],
        )

    def _deliver(self, generation: int, text: str, unprompted: bool = False) -> None:
        """Synthesize and stream one utterance; this never runs on the tick."""

        for attempt in range(1, MAX_SPEECH_ATTEMPTS + 1):
            try:
                self._attempt(generation, text, unprompted)
            except _Stale:
                return
            except Exception as error:
                reason = type(error).__name__
                with self._lock:
                    began = self._speaking
                if began:
                    self._end_partial(generation, reason, attempt)
                    return
                if attempt == MAX_SPEECH_ATTEMPTS:
                    self._settle(generation, Stage.FAILED, reason, attempt)
                    return
                if not self._settle(generation, Stage.DELIVERING, reason, attempt):
                    return
                self._sleep(SPEECH_RETRY_BACKOFF_S)
            else:
                self._settle(generation, Stage.AWAITING_DONE, None, attempt)
                return

    def _attempt(self, generation: int, text: str, unprompted: bool = False) -> None:
        """Run one complete delivery attempt, from synthesis to ``speak_end``."""

        speech = self._synthesize(text)
        self._send(
            generation, self._speak_callback, SpeakBeginMessage(ENVELOPE_HZ), speech
        )
        for index, chunk in enumerate(speech.chunks):
            self._send(generation, self._pcm_callback, chunk)
            if index == 0 and not unprompted:
                # An unprompted line has no VAD start, so it belongs to no
                # PRD 11.1 interaction row and must not mark one.
                self._emit(Milestone.FIRST_AUDIO_CHUNK, self._now())
        self._send(generation, self._speak_callback, SpeakEndMessage())

    def _send(
        self,
        generation: int,
        callback: Callable[..., None],
        payload: object,
        speech: _Speech | None = None,
    ) -> None:
        """Validate the generation and enqueue one payload without a gap.

        Passing ``speech`` marks this send as ``speak_begin``: speech state is
        armed before the callback runs and rolled back if it raises, so a
        failed begin still counts as speech that never started.
        """

        with self._lock:
            if self._closed or generation != self._generation:
                raise _Stale()
            if speech is not None:
                self._envelope = speech.envelope
                self._speech_started_at = self._now()
                self._speaking = True
            try:
                callback(payload)
            except Exception:
                if speech is not None:
                    self._clear_speech_locked()
                raise

    def _synthesize(self, text: str) -> _Speech:
        """Synthesize and validate audio against the PRD 8.4 wire contract."""

        audio = self._tts.synthesize(text)
        if not isinstance(audio, SpeechAudio):
            raise TypeError("TTS must return SpeechAudio")
        if audio.sample_hz != TTS_SAMPLE_HZ or audio.envelope_hz != ENVELOPE_HZ:
            raise ValueError("TTS returned an incompatible audio contract")
        chunks = tuple(self._tts.chunks(audio, max_bytes=DEFAULT_CHUNK_BYTES))
        if not chunks or b"".join(chunks) != audio.pcm_int16:
            raise ValueError("TTS chunks do not reconstruct the utterance")
        for chunk in chunks:
            if not chunk or len(chunk) % 2 or len(chunk) > DEFAULT_CHUNK_BYTES:
                raise ValueError("TTS chunk is not even-sized PCM of at most 8 KiB")
        return _Speech(tuple(audio.envelope), chunks)

    def _settle(
        self, generation: int, stage: Stage, reason: str | None, attempts: int
    ) -> bool:
        """Record one delivery outcome unless the generation already moved on."""

        with self._lock:
            if self._closed or generation != self._generation:
                return False
            self._stage = stage
            self._speech_attempts = attempts
            if reason is not None:
                self._last_error = reason
            return True

    def _end_partial(self, generation: int, reason: str, attempts: int) -> None:
        """Send one best-effort ``speak_end`` and then wait for the browser.

        Audio already reached the browser, so this must not retry and must not
        fabricate completion: only ``tts_done``, reset, or disengage clears it.
        """

        with self._lock:
            if not self._settle(generation, Stage.STALLED, reason, attempts):
                return
            try:
                self._speak_callback(SpeakEndMessage())
            except Exception:
                LOGGER.exception("speak_end after partial delivery failed")

    def _emit(self, milestone: Milestone, t: float) -> None:
        """Report one latency milestone; instrumentation never breaks staging."""

        if self._milestone_callback is None:
            return
        try:
            self._milestone_callback(milestone, t)
        except Exception:
            LOGGER.exception("milestone callback failed")

    def _invalidate_locked(self) -> None:
        self._generation += 1
        self._completions.clear()
        future, self._future = self._future, None
        if future is not None:
            future.cancel()
        self._stage = Stage.IDLE
        self._transcript = None
        self._reply = None
        self._unprompted = False
        self._vad_start = None
        self._speech_attempts = 0
        self._clear_speech_locked()
        with self._blackboard.lock:
            self._plans.clear()
            self._blackboard.set_plan(())
            self._blackboard.publish_utterance(None)

    def _clear_speech_locked(self) -> None:
        self._envelope = ()
        self._speech_started_at = None
        self._speaking = False

    def _now(self) -> float:
        return _time("clock", self._clock())


def _line(value: object) -> str:
    """Collapse dialogue to the single line the payload boundaries require."""

    if not isinstance(value, str):
        raise TypeError("dialogue text must be a string")
    return " ".join(value.split())


def _conversation_fallback(error: ConversationUnavailableError) -> str:
    """Render one deterministic line from a typed operational outcome."""

    if error.kind != "rate_limited":
        return FALLBACK_SAY
    if error.retry_after_s is None:
        return "OpenRouter is rate-limiting me—please try again later."
    seconds = max(1, int(math.ceil(error.retry_after_s)))
    return f"OpenRouter is rate-limiting me—please try again in {seconds} seconds."


def _time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    instant = float(value)
    if not math.isfinite(instant) or instant < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return instant


def _pcm(value: bytes) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) % 2:
        raise ValueError("utterance must be complete signed-int16 PCM bytes")
    return value


__all__ = [
    "ConversationCoordinator",
    "CoordinatorStatus",
    "FALLBACK_SAY",
    "MAX_RECENT",
    "MAX_SPEECH_ATTEMPTS",
    "ObservationOriginCallback",
    "SPEECH_RETRY_BACKOFF_S",
    "Stage",
]
