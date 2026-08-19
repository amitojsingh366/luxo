"""Public plan types at the model/body boundary.

The model can select only these verbs and their semantic arguments. Joint
angles, animation timing, easing, and physical limits are intentionally absent
from this representation, so a model response has no structural path to them.
Validation and JSON repair are assigned to the later ``plan-schema`` packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionOp(str, Enum):
    GESTURE = "gesture"
    LOOK_AT = "look_at"
    LIGHT = "light"
    SFX = "sfx"
    SCAN = "scan"
    OBSERVE = "observe"
    POSTURE = "posture"
    WAIT = "wait"


class GestureName(str, Enum):
    PERK_UP = "perk_up"
    NOD = "nod"
    DOUBLE_TAKE = "double_take"
    RECOIL = "recoil"
    LEAN_IN = "lean_in"
    BOUNCE = "bounce"
    SHAKE_NO = "shake_no"
    SETTLE = "settle"
    DROOP = "droop"
    REGARD = "regard"


class LightPreset(str, Enum):
    WARM_IDLE = "warm_idle"
    WARM_BRIGHT = "warm_bright"
    COOL_DIM = "cool_dim"
    CURIOUS_FOCUS = "curious_focus"
    THINKING_PULSE = "thinking_pulse"
    EXCITED_FLASH = "excited_flash"
    SAD_FADE = "sad_fade"


class LightPattern(str, Enum):
    STEADY = "steady"
    PULSE = "pulse"
    FLICKER = "flicker"
    BLINK = "blink"


class SfxName(str, Enum):
    CHIRP_UP = "chirp_up"
    CHIRP_FOUND = "chirp_found"
    BOING = "boing"
    WHIRR_SHORT = "whirr_short"
    HMM = "hmm"
    BLIP_SAD = "blip_sad"
    FANFARE_SMALL = "fanfare_small"
    CLICK = "click"


class PostureName(str, Enum):
    REST = "rest"
    ALERT = "alert"
    SLUMP = "slump"
    STOOP = "stoop"
    CRANE = "crane"


@dataclass(frozen=True, slots=True)
class Action:
    """One semantic action; op-specific field validation is implemented later."""

    op: ActionOp
    name: GestureName | SfxName | PostureName | None = None
    target: str | None = None
    preset: LightPreset | None = None
    pattern: LightPattern | None = None
    arc: float | None = None
    speed: float | None = None
    ms: int | None = None


@dataclass(frozen=True, slots=True)
class PlanResponse:
    say: str
    plan: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class ObservationResponse:
    present: tuple[str, ...]
    new: tuple["SceneObject", ...]


from .memory import SceneObject  # noqa: E402  (type is shared by this schema)


__all__ = [
    "Action",
    "ActionOp",
    "GestureName",
    "LightPattern",
    "LightPreset",
    "ObservationResponse",
    "PlanResponse",
    "PostureName",
    "SfxName",
]
