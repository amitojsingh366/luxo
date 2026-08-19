"""Command-line entry point for the character core scaffold."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .config import DEFAULT_CONFIG_PATH, FrozenConfig, load_config
from .logging_setup import configure_logging


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
    parser = argparse.ArgumentParser(prog="lumen-core")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and exit without opening a socket",
    )
    return parser


def _default_server_factory(
    *,
    host: str,
    port: int,
    fps: float,
) -> ServingProtocol:
    # Protocol is owned by a parallel Phase 0 packet, so keep the import lazy.
    from .protocol import ProtocolServer

    return ProtocolServer(host=host, port=port, fps=fps)


async def serve_core(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Serve the body-state stream until shutdown without blocking animation."""

    endpoint = urlsplit(str(config.network.websocket_url))
    host = endpoint.hostname
    port = endpoint.port
    if host is None or port is None:
        raise ValueError("network.websocket_url must include a host and port")

    server = server_factory(
        host=host,
        port=port,
        fps=float(config.runtime.body_state_hz),
    )
    await server.serve_forever()


def run_server(
    config: FrozenConfig,
    server_factory: ServerFactory = _default_server_factory,
) -> None:
    """Own the asyncio loop for the protocol server."""

    asyncio.run(serve_core(config, server_factory))


def main(argv: Sequence[str] | None = None) -> int:
    """Validate config or run the protocol stream until interrupted."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    logging.getLogger(__name__).info(
        "loaded %s core configuration for %s",
        config.runtime.python,
        config.name,
    )
    if args.check:
        return 0

    try:
        run_server(config)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("core shutdown requested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
