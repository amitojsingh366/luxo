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
    ObservationOriginCallback,
    Stage,
)
from .observations import (
    DEFAULT_WORKERS,
    MAX_CAPTURE_ATTEMPTS,
    BaselineLabels,
    FrameRejection,
    ObservationResolver,
    ObservationRuntime,
    ObservationStage,
    ObservationStatus,
    ResolutionCallback,
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
    "ObservationOriginCallback",
    "ObservationResolver",
    "ObservationRuntime",
    "ObservationStage",
    "ObservationStatus",
    "ResolutionCallback",
    "ROUTED_OPS",
    "RouterStatus",
    "SPEECH_RETRY_BACKOFF_S",
    "Stage",
]
