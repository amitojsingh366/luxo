"""Serialized runtime staging for the Luxo character core."""

# ``CaptureCallback`` is declared identically in both .actions and
# .observations; it is re-exported once from .actions.
from .actions import (
    CANCELLING_STATES,
    CUED_SFX,
    ROUTED_OPS,
    ActionRouter,
    CaptureCallback,
    CueCallback,
    RouterStatus,
)
from .interactions import (
    FALLBACK_SAY,
    MAX_SPEECH_ATTEMPTS,
    SPEECH_RETRY_BACKOFF_S,
    ConversationCoordinator,
    CoordinatorStatus,
    Stage,
)
from .observations import (
    DEFAULT_WORKERS,
    MAX_CAPTURE_ATTEMPTS,
    BaselineLabels,
    FrameRejection,
    NarratePolicy,
    NarrationCallback,
    ObservationRuntime,
    ObservationStage,
    ObservationStatus,
)

__all__ = [
    "ActionRouter",
    "BaselineLabels",
    "CANCELLING_STATES",
    "CUED_SFX",
    "CaptureCallback",
    "ConversationCoordinator",
    "CoordinatorStatus",
    "CueCallback",
    "DEFAULT_WORKERS",
    "FALLBACK_SAY",
    "FrameRejection",
    "MAX_CAPTURE_ATTEMPTS",
    "MAX_SPEECH_ATTEMPTS",
    "NarratePolicy",
    "NarrationCallback",
    "ObservationRuntime",
    "ObservationStage",
    "ObservationStatus",
    "ROUTED_OPS",
    "RouterStatus",
    "SPEECH_RETRY_BACKOFF_S",
    "Stage",
]
