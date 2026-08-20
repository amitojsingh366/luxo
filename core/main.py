"""Command-line entry point: build the character and own the process.

``core.main`` is deliberately thin. It resolves configuration and asset paths,
constructs the real STT, TTS, and brain boundaries, hands them to
:class:`core.runtime.app.LuxoApp`, and then owns exactly two things the app
does not: the asyncio loop the WebSocket server runs on, and process exit.

Secrets. ``OPENROUTER_API_KEY`` is read from the environment and nowhere else.
This module never opens the repository ``.env``; loading it is ``run.sh``'s job.

Asset paths. ``setup.sh`` and ``config/models.yaml`` own where weights land.
Every path below is overridable by environment variable so the two can be
reconciled without touching Python, and every default matches the manifest
destination that exists today.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .config import DEFAULT_CONFIG_PATH, FrozenConfig, load_config
from .logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "luxo"
# setup.sh copies the binary out of the build tree into <cache>/bin.
DEFAULT_WHISPER_BINARY = DEFAULT_CACHE_DIR / "bin" / "whisper-cli"
DEFAULT_WHISPER_MODEL = DEFAULT_CACHE_DIR / "ggml-base.en-q5_1.bin"
DEFAULT_PIPER_MODEL = DEFAULT_CACHE_DIR / "en_US-lessac-medium.onnx"
DEFAULT_PIPER_CONFIG = DEFAULT_CACHE_DIR / "en_US-lessac-medium.onnx.json"
DEFAULT_MEMORY_PATH = DEFAULT_CACHE_DIR / "scene_memory.json"
DEFAULT_LATENCY_CSV = Path(__file__).resolve().parents[1] / "measurements" / "latency.csv"

FREE_PROFILE_MODEL = "openrouter/free"
"""Placeholder model id for profile ``free``; ``config`` leaves the id unset."""


class ServingProtocol(Protocol):
    async def serve_forever(self) -> None: ...


class ServerFactory(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        fps: float,
    ) -> ServingProtocol: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luxo-core")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and exit without opening a socket",
    )
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help=(
            "serve the body-state stream without assembling the character; "
            "used to exercise the transport before models are installed"
        ),
    )
    return parser


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _default_server_factory(
    *,
    host: str,
    port: int,
    fps: float,
) -> ServingProtocol:
    from .protocol import ProtocolServer

    return ProtocolServer(host=host, port=port, fps=fps)


def _endpoint(config: FrozenConfig) -> tuple[str, int]:
    """Resolve the loopback endpoint; the PRD forbids any non-local bind."""

    parsed = urlsplit(str(config.network.websocket_url))
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        raise ValueError("network.websocket_url must include a host and port")
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the core binds loopback only; 0.0.0.0 is never valid")
    return host, port


def build_models(config: FrozenConfig, metrics_callback: object = None) -> dict[str, object]:
    """Construct the three real model boundaries from the environment.

    Every construction here can fail on a machine without assets, and that is
    intentional: a missing model must be reported by name at startup rather
    than discovered as a dead character during the recording.

    ``metrics_callback`` is passed straight to the brain. It is the only way
    OpenRouter's reported model id and token counts reach the PRD 11.1 CSV,
    because ``BrainClient`` takes that callback at construction time.
    """

    from .brain.client import OpenRouterBrainClient
    from .speech.phonemes import create_phoneme_encoder
    from .speech.stt import WhisperCppSpeechToText
    from .speech.tts import PiperTextToSpeech

    piper_config = _env_path("LUXO_PIPER_CONFIG", DEFAULT_PIPER_CONFIG)
    profile = str(config.brain.profile)
    model = os.environ.get("OPENROUTER_MODEL") or (
        FREE_PROFILE_MODEL if profile == "free" else ""
    )

    stt = WhisperCppSpeechToText(
        binary_path=_env_path("LUXO_WHISPER_BIN", DEFAULT_WHISPER_BINARY),
        model_path=_env_path("LUXO_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
    )
    tts = PiperTextToSpeech(
        _env_path("LUXO_PIPER_MODEL", DEFAULT_PIPER_MODEL),
        piper_config,
        create_phoneme_encoder(piper_config),
        length_scale=float(config.speech.piper_length_scale),
    )
    brain = OpenRouterBrainClient(
        profile=profile,
        model=model or None,
        metrics_callback=metrics_callback,  # type: ignore[arg-type]
    )
    return {"stt": stt, "tts": tts, "brain": brain, "model": brain.model, "profile": profile}


def build_app(config: FrozenConfig, server: object):
    """Assemble one :class:`LuxoApp` around an already-built server.

    The latency recorder is built before the models so the brain can report
    its token counts into the same recorder the conversation milestones land
    in; otherwise every CSV row would carry zero tokens.
    """

    from .brain.memory import SceneMemoryStore
    from .instrumentation import InteractionCSVLogger
    from .runtime.app import LatencyRecorder, LuxoApp

    profile = str(config.brain.profile)
    recorder = LatencyRecorder(
        InteractionCSVLogger(_env_path("LUXO_LATENCY_CSV", DEFAULT_LATENCY_CSV)),
        model=os.environ.get("OPENROUTER_MODEL") or FREE_PROFILE_MODEL,
        profile=profile,
    )
    models = build_models(config, recorder.on_call_metrics)
    return LuxoApp(
        protocol=server,  # type: ignore[arg-type]
        stt=models["stt"],  # type: ignore[arg-type]
        tts=models["tts"],  # type: ignore[arg-type]
        brain=models["brain"],  # type: ignore[arg-type]
        memory_store=SceneMemoryStore(_env_path("LUXO_MEMORY_PATH", DEFAULT_MEMORY_PATH)),
        config=config,
        latency_recorder=recorder,
        brain_model=str(models["model"]),
        brain_profile=str(models["profile"]),
    )


async def serve_core(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Serve the body-state stream until shutdown without blocking animation."""

    host, port = _endpoint(config)
    server = server_factory(
        host=host,
        port=port,
        fps=float(config.runtime.body_state_hz),
    )
    await server.serve_forever()


async def run_character(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Run the whole character: transport on this loop, ticks on two threads.

    The protocol server owns this thread's event loop. The behaviour and
    animation ticks run on their own threads and reach the socket only by
    swapping a body-state snapshot or by queueing a discrete event, so a slow
    or absent browser can never stall either tick.
    """

    host, port = _endpoint(config)
    server = server_factory(
        host=host,
        port=port,
        fps=float(config.runtime.body_state_hz),
    )
    app = build_app(config, server)
    _attach_callbacks(server, app)

    stopped = asyncio.Event()
    _install_signal_handlers(stopped)
    app.start()
    serving = asyncio.ensure_future(server.serve_forever())
    shutdown = asyncio.ensure_future(stopped.wait())
    try:
        done, _ = await asyncio.wait(
            {serving, shutdown}, return_when=asyncio.FIRST_COMPLETED
        )
        if serving in done:
            # The transport ended by itself. Surface why, rather than leaving
            # two tick threads running behind a socket that is already gone.
            serving.result()
    finally:
        for task in (serving, shutdown):
            task.cancel()
        # Retrieving both outcomes here is what turns a transport failure into
        # a shutdown instead of an unretrieved-exception warning.
        await asyncio.gather(serving, shutdown, return_exceptions=True)
        app.stop()


def _attach_callbacks(server: object, app: object) -> None:
    """Point the transport at the app without either importing the other.

    Assignment is unconditional on purpose. A server that will not accept the
    two callbacks is a server whose inbound frames would be silently dropped,
    which is worse than failing here at startup.
    """

    server.on_message = app.on_message  # type: ignore[attr-defined]
    server.on_binary = app.on_binary  # type: ignore[attr-defined]


def _install_signal_handlers(stopped: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        handled = getattr(signal, name, None)
        if handled is None:  # pragma: no cover - every supported platform has both
            continue
        try:
            loop.add_signal_handler(handled, stopped.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-main thread
            LOGGER.debug("no loop signal handler for %s", name)


def run_core(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Own the asyncio loop for the default launch: the whole character."""

    asyncio.run(run_character(config, server_factory))


def run_protocol_only(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Own the asyncio loop for the transport alone, with no character."""

    asyncio.run(serve_core(config, server_factory))


def main(argv: Sequence[str] | None = None) -> int:
    """Validate config or run the assembled character until interrupted."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    LOGGER.info(
        "loaded %s core configuration for %s",
        config.runtime.python,
        config.name,
    )
    if args.check:
        return 0

    try:
        if args.protocol_only:
            run_protocol_only(config)
        else:
            run_core(config)
    except KeyboardInterrupt:
        LOGGER.info("core shutdown requested")
    except Exception as error:
        # A missing model or an unreadable asset must name itself and exit
        # non-zero rather than leaving a half-built character running.
        LOGGER.error("the character could not start: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
