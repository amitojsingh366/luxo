"""Serialized runtime staging for the Luxo character core."""

from .interactions import (
    FALLBACK_SAY,
    MAX_SPEECH_ATTEMPTS,
    SPEECH_RETRY_BACKOFF_S,
    ConversationCoordinator,
    CoordinatorStatus,
    Stage,
)

__all__ = [
    "ConversationCoordinator",
    "CoordinatorStatus",
    "FALLBACK_SAY",
    "MAX_SPEECH_ATTEMPTS",
    "SPEECH_RETRY_BACKOFF_S",
    "Stage",
]
