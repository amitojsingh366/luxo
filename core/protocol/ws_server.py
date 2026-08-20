"""Reconnect-safe WebSocket transport for the Python mind."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from core.animation.poses import load_pose_library

from .messages import (
    AudioState,
    BinaryFrame,
    BinaryFrameType,
    BodyStateMessage,
    BrowserToCoreMessage,
    ClampCounts,
    CoreToBrowserMessage,
    Direction,
    HelloMessage,
    JointsState,
    LightState,
    ProtocolError,
    TelemetryGaze,
    TelemetryState,
    parse_binary_frame,
    parse_text_message,
    serialize_binary_frame,
    serialize_text_message,
)


LOGGER = logging.getLogger(__name__)
MessageCallback = Callable[[BrowserToCoreMessage], object | Awaitable[object]]
BinaryCallback = Callable[[BinaryFrame], object | Awaitable[object]]


def canonical_rest_body_state(*, seq: int = 0, timestamp: float = 0.0) -> BodyStateMessage:
    """Return the complete initial body state at the configured home pose."""

    home = load_pose_library().home

    return BodyStateMessage(
        t=timestamp,
        seq=seq,
        joints=JointsState(
            base_yaw=home.base_yaw,
            shoulder_pitch=home.shoulder_pitch,
            elbow_pitch=home.elbow_pitch,
            neck_yaw=home.neck_yaw,
            head_pitch=home.head_pitch,
        ),
        light=LightState(
            intensity=0.55,
            color_k=2700,
            pattern="steady",
            bloom=0.6,
        ),
        audio=AudioState(speaking=False, arousal=0.15),
        telemetry=TelemetryState(
            state="DORMANT",
            plan_depth=0,
            memory_count=0,
            last_latency_ms=0.0,
            clamps=ClampCounts(vel=0, limit=0),
            gaze=TelemetryGaze(present=False, yaw_deg=0.0, pitch_deg=0.0),
        ),
    )


@dataclass(slots=True)
class _ClientSession:
    websocket: Any
    wake: asyncio.Event
    events: deque[str | bytes]
    hello: HelloMessage | None = None
    pending_body: str | None = None
    writer: asyncio.Task[None] | None = None


class ProtocolServer:
    """Own the WebSocket I/O boundary without blocking animation producers.

    ``publish_body_state`` only replaces an immutable snapshot under a short
    lock.  A 60 Hz broadcaster coalesces snapshots per client, and each client
    has an independent writer task.  Slow or disconnected browsers therefore
    cannot block the animation thread or another browser.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        fps: float = 60.0,
        *,
        on_message: MessageCallback | None = None,
        on_binary: BinaryCallback | None = None,
        ping_interval: float = 10.0,
        ping_timeout: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.host = host
        self.port = port
        self.fps = fps
        self.on_message = on_message
        self.on_binary = on_binary
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._clock = clock
        self._clients: dict[Any, _ClientSession] = {}
        self._body_lock = threading.Lock()
        self._body_template = canonical_rest_body_state()
        self._last_state: BodyStateMessage | None = None
        self._sequence = 0
        self._server: Any = None
        self._broadcast_task: asyncio.Task[None] | None = None
        self._callback_tasks: set[asyncio.Task[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.last_hello: HelloMessage | None = None
        self.hello_count = 0

    @property
    def connected_count(self) -> int:
        return len(self._clients)

    @property
    def ready_count(self) -> int:
        return sum(session.hello is not None for session in self._clients.values())

    def publish_body_state(self, state: BodyStateMessage) -> None:
        """Publish a state snapshot immediately, without awaiting socket I/O."""

        if not isinstance(state, BodyStateMessage):
            raise TypeError("state must be BodyStateMessage")
        with self._body_lock:
            self._body_template = state

    def publish_event(self, message: CoreToBrowserMessage) -> None:
        """Queue a discrete JSON event for every browser that sent ``hello``."""

        payload = serialize_text_message(message, Direction.CORE_TO_BROWSER)
        self._call_in_server_loop(self._enqueue_event_for_ready_clients, payload)

    def publish_tts_pcm(self, payload: bytes | bytearray | memoryview) -> None:
        """Queue one prefixed TTS PCM chunk for connected, ready browsers."""

        frame = serialize_binary_frame(
            BinaryFrameType.TTS_PCM, payload, Direction.CORE_TO_BROWSER
        )
        self._call_in_server_loop(self._enqueue_event_for_ready_clients, frame)

    def _call_in_server_loop(self, callback: Callable[..., None], *args: object) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("protocol server is not running")
        loop.call_soon_threadsafe(callback, *args)

    def _enqueue_event_for_ready_clients(self, payload: str | bytes) -> None:
        for session in self._clients.values():
            if session.hello is not None:
                session.events.append(payload)
                session.wake.set()

    def next_body_state(self, timestamp: float | None = None) -> BodyStateMessage:
        """Stamp the newest producer snapshot with transport sequence and time."""

        with self._body_lock:
            template = self._body_template
        state = replace(
            template,
            t=self._clock() if timestamp is None else timestamp,
            seq=self._sequence,
        )
        self._sequence += 1
        self._last_state = state
        return state

    def broadcast_once(self, timestamp: float | None = None) -> BodyStateMessage:
        """Build and coalesce one body-state tick; exposed for offline checks."""

        state = self.next_body_state(timestamp)
        payload = serialize_text_message(state, Direction.CORE_TO_BROWSER)
        for session in self._clients.values():
            if session.hello is not None:
                session.pending_body = payload
                session.wake.set()
        return state

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / self.fps
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        while True:
            self.broadcast_once()
            next_tick += interval
            delay = next_tick - loop.time()
            if delay < -interval:
                next_tick = loop.time()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))

    async def _writer(self, session: _ClientSession) -> None:
        while True:
            await session.wake.wait()
            session.wake.clear()
            while session.events:
                await session.websocket.send(session.events.popleft())
            if session.pending_body is not None:
                payload, session.pending_body = session.pending_body, None
                await session.websocket.send(payload)

    def _dispatch(self, callback: MessageCallback | BinaryCallback | None, value: object) -> None:
        if callback is None:
            return
        try:
            result = callback(value)  # type: ignore[arg-type]
        except Exception:
            LOGGER.exception("protocol callback failed")
            return
        if inspect.isawaitable(result):
            task = asyncio.create_task(result)
            self._callback_tasks.add(task)
            task.add_done_callback(self._finish_callback)

    def _finish_callback(self, task: asyncio.Task[Any]) -> None:
        self._callback_tasks.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            LOGGER.error(
                "asynchronous protocol callback failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _handle_frame(self, session: _ClientSession, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            frame = parse_binary_frame(raw, Direction.BROWSER_TO_CORE)
            self._dispatch(self.on_binary, frame)
            return

        message = parse_text_message(raw, Direction.BROWSER_TO_CORE)
        if isinstance(message, HelloMessage):
            session.hello = message
            self.last_hello = message
            self.hello_count += 1
            state = self._last_state or self.next_body_state()
            session.pending_body = serialize_text_message(
                state, Direction.CORE_TO_BROWSER
            )
            session.wake.set()
        self._dispatch(self.on_message, message)

    async def handle_connection(self, websocket: Any) -> None:
        """Handle one connection; a later connection starts with clean state."""

        session = _ClientSession(websocket, asyncio.Event(), deque())
        self._clients[websocket] = session
        session.writer = asyncio.create_task(self._writer(session))
        try:
            async for raw in websocket:
                try:
                    await self._handle_frame(session, raw)
                except ProtocolError as error:
                    LOGGER.warning("closing invalid protocol connection: %s", error)
                    await websocket.close(code=1003, reason="invalid protocol frame")
                    break
        finally:
            self._clients.pop(websocket, None)
            if session.writer is not None:
                session.writer.cancel()
                with suppress(asyncio.CancelledError):
                    await session.writer

    async def start(self) -> None:
        """Start accepting clients and producing body states at 60 Hz."""

        if self._server is not None:
            return
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._server = await serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=2 * 1024 * 1024,
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        LOGGER.info("protocol server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop accepting clients while leaving the object restartable."""

        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._broadcast_task
            self._broadcast_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in tuple(self._callback_tasks):
            task.cancel()
        if self._callback_tasks:
            await asyncio.gather(*self._callback_tasks, return_exceptions=True)
        self._callback_tasks.clear()
        self._loop = None

    async def serve_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Future()
        finally:
            await self.stop()
