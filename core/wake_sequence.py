"""Non-blocking model warm-up and the locked Luxo waking intent sequence.

The coordinator owns readiness facts, not behavior timing. Blocking model work
runs through an executor; callbacks only enqueue semantic actions and closed FSM
events. No joint value, animation duration, easing, or light number exists here.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

from .brain.client import BrainClient
from .brain.schema import (
    Action,
    ActionOp,
    LightPattern,
    LightPreset,
    PostureName,
    SfxName,
)
from .fsm import BehaviorEvent
from .speech.stt import SpeechToText
from .speech.tts import TextToSpeech


WAKE_ACTIONS = (
    Action(op=ActionOp.POSTURE, name=PostureName.SLUMP),
    Action(
        op=ActionOp.LIGHT,
        preset=LightPreset.WARM_IDLE,
        pattern=LightPattern.FLICKER,
    ),
    Action(op=ActionOp.SFX, name=SfxName.WHIRR_SHORT),
    Action(op=ActionOp.POSTURE, name=PostureName.REST),
)
"""Locked beat: slump, flicker on, whirr, then rise to rest."""


class WakeSequenceError(RuntimeError):
    """One or more startup boundaries failed without becoming ready."""

    def __init__(self, failures: Mapping[str, BaseException]) -> None:
        self.failures = MappingProxyType(dict(sorted(failures.items())))
        summary = ", ".join(
            f"{name} ({type(error).__name__})" for name, error in self.failures.items()
        )
        super().__init__(f"startup warm-up failed: {summary}")


@dataclass(frozen=True, slots=True)
class WakeSequenceStatus:
    attempt: int
    warming: bool
    models_warm: bool
    browser_hello: bool
    camera_ready: bool
    ready_for_dormant: bool
    last_failed_components: tuple[str, ...]


class WakeSequenceCoordinator:
    """Coordinate model workers, browser facts, and once-only startup cues."""

    _COMPONENTS = ("stt", "tts", "brain")

    def __init__(
        self,
        *,
        stt: SpeechToText,
        tts: TextToSpeech,
        brain: BrainClient,
        emit_action: Callable[[Action], object],
        post_event: Callable[[BehaviorEvent], object],
        executor: Executor | None = None,
    ) -> None:
        warmers = {"stt": stt, "tts": tts, "brain": brain}
        for name, component in warmers.items():
            if not callable(getattr(component, "warm", None)):
                raise TypeError(f"{name} must provide warm()")
        if not callable(emit_action) or not callable(post_event):
            raise TypeError("startup callbacks must be callable")
        self._warmers = warmers
        self._emit_action = emit_action
        self._post_event = post_event
        self._executor = executor
        self._owns_executor = executor is None
        self._lock = threading.RLock()
        self._attempt = 0
        self._attempt_future: Future[None] | None = None
        self._pending: set[str] = set()
        self._failures: dict[str, BaseException] = {}
        self._models_warm = False
        self._browser_hello = False
        self._camera_ready = False
        self._models_event_sent = False
        self._browser_event_sent = False
        self._opening_actions_sent = False
        self._rest_action_sent = False
        self._closed = False

    @property
    def ready_for_dormant(self) -> bool:
        with self._lock:
            return self._ready_locked()

    def status(self) -> WakeSequenceStatus:
        with self._lock:
            future = self._attempt_future
            return WakeSequenceStatus(
                attempt=self._attempt,
                warming=future is not None and not future.done(),
                models_warm=self._models_warm,
                browser_hello=self._browser_hello,
                camera_ready=self._camera_ready,
                ready_for_dormant=self._ready_locked(),
                last_failed_components=tuple(sorted(self._failures)),
            )

    def mark_browser_hello(self) -> None:
        with self._lock:
            self._ensure_open()
            self._browser_hello = True
            self._publish_browser_ready_locked()

    def mark_camera_ready(self) -> None:
        with self._lock:
            self._ensure_open()
            self._camera_ready = True
            self._publish_browser_ready_locked()

    def warm(self) -> Future[None]:
        """Submit all blocking warm calls and return immediately.

        Concurrent calls share one future. After failure the next call begins a
        clean attempt; already-warmed component boundaries remain idempotent.
        """

        with self._lock:
            self._ensure_open()
            if self._attempt_future is not None and (
                self._models_warm or not self._attempt_future.done()
            ):
                return self._attempt_future
            self._emit_opening_actions_locked()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=len(self._COMPONENTS),
                    thread_name_prefix="luxo-warm",
                )
            self._attempt += 1
            attempt = self._attempt
            aggregate: Future[None] = Future()
            self._attempt_future = aggregate
            self._pending = set(self._COMPONENTS)
            self._failures = {}
            for name in self._COMPONENTS:
                try:
                    worker = self._executor.submit(self._warmers[name].warm)
                except BaseException as error:
                    worker = Future()
                    worker.set_exception(error)
                worker.add_done_callback(
                    lambda completed, component=name: self._worker_done(
                        attempt, component, completed
                    )
                )
            return aggregate

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor if self._owns_executor else None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _worker_done(
        self,
        attempt: int,
        component: str,
        worker: Future[object],
    ) -> None:
        with self._lock:
            if attempt != self._attempt or component not in self._pending:
                return
            try:
                error = worker.exception()
            except BaseException as cancelled:
                error = cancelled
            if error is not None:
                self._failures[component] = error
            self._pending.remove(component)
            if self._pending:
                return
            aggregate = self._attempt_future
            if aggregate is None or aggregate.done():
                return
            if self._failures:
                aggregate.set_exception(WakeSequenceError(self._failures))
                return
            try:
                if not self._rest_action_sent:
                    self._emit_action(WAKE_ACTIONS[-1])
                    self._rest_action_sent = True
                if not self._models_event_sent:
                    self._post_event(BehaviorEvent.MODELS_WARM)
                    self._models_event_sent = True
            except BaseException as error:
                self._failures = {"callback": error}
                aggregate.set_exception(WakeSequenceError(self._failures))
                return
            self._models_warm = True
            aggregate.set_result(None)

    def _emit_opening_actions_locked(self) -> None:
        if self._opening_actions_sent:
            return
        for action in WAKE_ACTIONS[:-1]:
            self._emit_action(action)
        self._opening_actions_sent = True

    def _publish_browser_ready_locked(self) -> None:
        if self._browser_hello and self._camera_ready and not self._browser_event_sent:
            self._post_event(BehaviorEvent.BROWSER_READY)
            self._browser_event_sent = True

    def _ready_locked(self) -> bool:
        return self._models_warm and self._browser_hello and self._camera_ready

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("wake sequence is closed")


__all__ = [
    "WAKE_ACTIONS",
    "WakeSequenceCoordinator",
    "WakeSequenceError",
    "WakeSequenceStatus",
]
