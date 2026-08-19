"""Process-wide logging setup."""

from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure a single stdout handler without duplicating handlers."""

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)


__all__ = ["LOG_FORMAT", "configure_logging"]
