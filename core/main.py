"""Command-line entry point for the character core scaffold."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config
from .logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen-core")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate bootstrap inputs; runtime wiring is an integration packet."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    logging.getLogger(__name__).info(
        "loaded %s core configuration for %s",
        config.runtime.python,
        config.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
