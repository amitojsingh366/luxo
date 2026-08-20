"""Validated model responses at the semantic model/body boundary.

The model can select only the verbs and semantic arguments in this module.
Joint angles, animation duration, easing, overshoot, and physical limits have
no representation here; those remain body-owned.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, TypeAlias

LOGGER = logging.getLogger(__name__)

SAY_MAX_CHARACTERS = 120
"""Maximum validated dialogue length required by the prompt contract."""

WAIT_MAX_MS = 60_000
"""Longest model-selected hold: one minute prevents accidental stalls."""

BBOX_EDGE_TOLERANCE = 0.025
"""Maximum normalized edge overflow clipped as vision-model rounding noise."""

OBSERVATION_MAX_VISIBLE = 10
"""Maximum nearest-first object facts accepted from one captured frame."""

_SAFE_OBJECT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_CANONICAL_SEPARATOR = re.compile(r"[\s_-]+")
_CANONICAL_LABEL = re.compile(r"[a-z0-9]+(?: [a-z0-9]+)*\Z")
_NON_DURABLE_SCENE_CANONICALS: Final = frozenset(
    {
        # People and body facts belong to the local gaze pipeline, not durable
        # object memory. Accessories remain eligible in their own right.
        "adult",
        "arm",
        "arms",
        "body",
        "boy",
        "child",
        "eye",
        "eyes",
        "ear",
        "ears",
        "face",
        "feet",
        "finger",
        "fingers",
        "foot",
        "girl",
        "hand",
        "hands",
        "hair",
        "head",
        "human",
        "individual",
        "leg",
        "legs",
        "man",
        "mouth",
        "neck",
        "nose",
        "people",
        "person",
        "subject",
        "shoulder",
        "shoulders",
        "thumb",
        "thumbs",
        "torso",
        "user",
        "woman",
        # These names contain no recallable identity or attributes and otherwise
        # consume the nearest-first memory budget on weak vision responses.
        "generic object",
        "item",
        "miscellaneous object",
        "object",
        "something",
        "thing",
        "unidentified object",
        "unknown",
        "unknown object",
    }
)


class ResponseSchemaError(ValueError):
    """A model payload cannot satisfy its required top-level contract."""


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
    """One validated semantic action with no physical-control fields."""

    op: ActionOp
    name: GestureName | SfxName | PostureName | None = None
    target: str | None = None
    preset: LightPreset | None = None
    pattern: LightPattern | None = None
    arc: float | None = None
    speed: float | None = None
    ms: int | None = None

    def __post_init__(self) -> None:
        _validate_action(self)


@dataclass(frozen=True, slots=True)
class PlanResponse:
    say: str
    plan: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.say, str) or len(self.say) > SAY_MAX_CHARACTERS:
            raise ValueError(f"say must be a string of at most {SAY_MAX_CHARACTERS} characters")
        if not isinstance(self.plan, tuple) or not all(
            isinstance(action, Action) for action in self.plan
        ):
            raise TypeError("plan must be a tuple of Action values")


@dataclass(frozen=True, slots=True)
class ObservedObject:
    """Model-authored object facts, excluding all local memory metadata."""

    label: str
    canonical: str
    attributes: tuple[str, ...]
    bbox_norm: tuple[float, float, float, float] | None = None
    match: str | None = None
    match_provided: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ObservationPrior:
    """Stable local identity exposed to one observation call."""

    id: str
    canonical: str
    attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _SAFE_OBJECT_ID.fullmatch(self.id) is None:
            raise ValueError("observation prior id must be a safe object id")
        if isinstance(self.attributes, (str, bytes)):
            raise TypeError("observation prior attributes must be a tuple of strings")
        try:
            attributes = tuple(
                dict.fromkeys(_plain_text(value, "attribute") for value in self.attributes)
            )
        except TypeError as error:
            raise TypeError(
                "observation prior attributes must be a tuple of strings"
            ) from error
        object.__setattr__(self, "canonical", normalize_canonical_label(self.canonical))
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class ObservationResponse:
    """Nearest-first facts and the model-selected subject from one frame."""

    visible: tuple[ObservedObject, ...]
    focus: int | None = None
    present_prior_ids: tuple[str, ...] | None = None
    raw_saturated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.visible, tuple) or not all(
            isinstance(item, ObservedObject) for item in self.visible
        ):
            raise TypeError("visible must be a tuple of ObservedObject values")
        if len(self.visible) > OBSERVATION_MAX_VISIBLE:
            raise ValueError(f"visible supports at most {OBSERVATION_MAX_VISIBLE} objects")
        if self.focus is not None and (
            isinstance(self.focus, bool)
            or not isinstance(self.focus, int)
            or not 0 <= self.focus < len(self.visible)
        ):
            raise ValueError("focus must be null or an index into visible")
        if self.present_prior_ids is not None:
            if not isinstance(self.present_prior_ids, tuple) or any(
                not isinstance(value, str) or _SAFE_OBJECT_ID.fullmatch(value) is None
                for value in self.present_prior_ids
            ):
                raise TypeError("present_prior_ids must be null or a tuple of safe ids")
            if len(self.present_prior_ids) != len(set(self.present_prior_ids)):
                raise ValueError("present_prior_ids must not contain duplicates")
        if type(self.raw_saturated) is not bool:
            raise TypeError("raw_saturated must be a boolean")


JsonObject: TypeAlias = Mapping[str, object]
RawPayload: TypeAlias = JsonObject | str


def normalize_canonical_label(value: object) -> str:
    """Return a stable lower-case, space-separated object label."""

    if not isinstance(value, str):
        raise ValueError("canonical label must be a string")
    normalized = _CANONICAL_SEPARATOR.sub(" ", value.strip().casefold())
    if not normalized or not _CANONICAL_LABEL.fullmatch(normalized):
        raise ValueError("canonical label must contain only letters, digits, and separators")
    return normalized


def is_durable_scene_canonical(value: object) -> bool:
    """Return whether a cloud-authored fact is worth durable object memory.

    The policy is deliberately limited to canonical object categories. It does
    not affect browser-local face, gaze, or hand geometry used for attention.
    """

    try:
        canonical = normalize_canonical_label(value)
    except ValueError:
        return False
    return canonical not in _NON_DURABLE_SCENE_CANONICALS


def parse_plan_response(payload: RawPayload) -> PlanResponse:
    """Validate a ``converse`` or observation-resolution response.

    Unknown keys are ignored. Invalid actions are logged and dropped without
    invalidating other actions in the plan.
    """

    root = _payload_object(payload, "plan")
    if "say" not in root or "plan" not in root:
        raise ResponseSchemaError("plan response requires 'say' and 'plan'")
    say = _say(root["say"])
    raw_plan = root["plan"]
    if not isinstance(raw_plan, list):
        raise ResponseSchemaError("plan response 'plan' must be an array")

    actions: list[Action] = []
    for index, raw_action in enumerate(raw_plan):
        try:
            actions.append(_parse_action(raw_action))
        except (TypeError, ValueError) as error:
            LOGGER.warning("dropping invalid action at index %d: %s", index, error)
    return PlanResponse(say=say, plan=tuple(actions))


def parse_observation_response(payload: RawPayload) -> ObservationResponse:
    """Validate the single nearest-first object list returned by ``observe``."""

    root = _payload_object(payload, "observation")
    forbidden = {"say", "plan", "dialogue"}.intersection(root)
    if forbidden:
        raise ResponseSchemaError(
            f"observation response must not contain dialogue keys {sorted(forbidden)!r}"
        )
    if "visible" not in root:
        raise ResponseSchemaError("observation response requires 'visible'")
    raw_visible = root["visible"]
    if not isinstance(raw_visible, list):
        raise ResponseSchemaError("observation 'visible' must be an array")
    parsed, retained_indexes = _observed_objects(
        raw_visible, "visible", OBSERVATION_MAX_VISIBLE
    )
    visible = tuple(parsed)
    raw_focus = root.get("focus")
    if raw_focus is None:
        focus = None
    elif (
        isinstance(raw_focus, bool)
        or not isinstance(raw_focus, int)
        or not 0 <= raw_focus < len(raw_visible)
    ):
        LOGGER.warning("dropping invalid observation focus %r", raw_focus)
        focus = None
    else:
        focus = retained_indexes.get(raw_focus)
        if focus is None:
            LOGGER.warning(
                "dropping observation focus %d because its object was not retained",
                raw_focus,
            )
    raw_presence = root.get("present_prior_ids")
    if raw_presence is None:
        presence = None
    elif not isinstance(raw_presence, list) or any(
        not isinstance(value, str) or _SAFE_OBJECT_ID.fullmatch(value) is None
        for value in raw_presence
    ):
        LOGGER.warning("dropping invalid present_prior_ids evidence")
        presence = None
    else:
        presence = tuple(dict.fromkeys(raw_presence))
    return ObservationResponse(
        visible=visible,
        focus=focus,
        present_prior_ids=presence,
        raw_saturated=len(raw_visible) >= OBSERVATION_MAX_VISIBLE,
    )


def _observed_objects(
    values: list[object], field: str, limit: int | None = None
) -> tuple[list[ObservedObject], dict[int, int]]:
    objects: list[ObservedObject] = []
    retained_indexes: dict[int, int] = {}
    for index, raw_object in enumerate(values):
        try:
            item = _parse_observed_object(raw_object)
        except (TypeError, ValueError) as error:
            LOGGER.warning("dropping invalid %s object at index %d: %s", field, index, error)
            continue
        if not is_durable_scene_canonical(item.canonical):
            LOGGER.warning(
                "dropping non-durable %s object at index %d: %s",
                field,
                index,
                item.canonical,
            )
            continue
        retained_indexes[index] = len(objects)
        objects.append(item)
        if limit is not None and len(objects) == limit:
            break
    return objects, retained_indexes


def _payload_object(payload: RawPayload, kind: str) -> JsonObject:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ResponseSchemaError(
                f"{kind} response must be one complete JSON object"
            ) from error
        if not isinstance(decoded, dict):
            raise ResponseSchemaError(f"{kind} response must be a JSON object")
        return decoded
    if not isinstance(payload, Mapping):
        raise ResponseSchemaError(f"{kind} response must be a JSON object")
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _say(value: object) -> str:
    if not isinstance(value, str):
        raise ResponseSchemaError("plan response 'say' must be a string")
    if len(value) > SAY_MAX_CHARACTERS:
        raise ResponseSchemaError(
            f"plan response 'say' must be at most {SAY_MAX_CHARACTERS} characters"
        )
    return value


def _parse_action(value: object) -> Action:
    if not isinstance(value, Mapping):
        raise TypeError("action must be an object")
    raw_op = value.get("op")
    try:
        op = ActionOp(raw_op)
    except (TypeError, ValueError):
        raise ValueError(f"unknown op {raw_op!r}") from None

    if op is ActionOp.GESTURE:
        return Action(op=op, name=_enum_field(value, "name", GestureName))
    if op is ActionOp.LOOK_AT:
        return Action(op=op, target=_look_target(value.get("target")))
    if op is ActionOp.LIGHT:
        pattern = value.get("pattern")
        return Action(
            op=op,
            preset=_enum_field(value, "preset", LightPreset),
            pattern=None if pattern is None else _enum_value(pattern, LightPattern, "pattern"),
        )
    if op is ActionOp.SFX:
        return Action(op=op, name=_enum_field(value, "name", SfxName))
    if op is ActionOp.SCAN:
        return Action(
            op=op,
            arc=_optional_finite_number(value.get("arc"), "arc"),
            speed=_optional_finite_number(value.get("speed"), "speed", positive=True),
        )
    if op is ActionOp.OBSERVE:
        return Action(op=op)
    if op is ActionOp.POSTURE:
        return Action(op=op, name=_enum_field(value, "name", PostureName))
    return Action(op=op, ms=_wait_ms(value.get("ms")))


def _validate_action(action: Action) -> None:
    if not isinstance(action.op, ActionOp):
        raise TypeError("op must be an ActionOp")
    populated = {
        field
        for field in ("name", "target", "preset", "pattern", "arc", "speed", "ms")
        if getattr(action, field) is not None
    }
    allowed = {
        ActionOp.GESTURE: {"name"},
        ActionOp.LOOK_AT: {"target"},
        ActionOp.LIGHT: {"preset", "pattern"},
        ActionOp.SFX: {"name"},
        ActionOp.SCAN: {"arc", "speed"},
        ActionOp.OBSERVE: set(),
        ActionOp.POSTURE: {"name"},
        ActionOp.WAIT: {"ms"},
    }[action.op]
    if not populated <= allowed:
        raise ValueError(f"{action.op.value} has fields for another op")

    if action.op is ActionOp.GESTURE and not isinstance(action.name, GestureName):
        raise ValueError("gesture requires a valid name")
    if action.op is ActionOp.LOOK_AT:
        _look_target(action.target)
    if action.op is ActionOp.LIGHT:
        if not isinstance(action.preset, LightPreset):
            raise ValueError("light requires a valid preset")
        if action.pattern is not None and not isinstance(action.pattern, LightPattern):
            raise ValueError("light pattern is invalid")
    if action.op is ActionOp.SFX and not isinstance(action.name, SfxName):
        raise ValueError("sfx requires a valid name")
    if action.op is ActionOp.SCAN:
        _optional_finite_number(action.arc, "arc")
        _optional_finite_number(action.speed, "speed", positive=True)
    if action.op is ActionOp.POSTURE and not isinstance(action.name, PostureName):
        raise ValueError("posture requires a valid name")
    if action.op is ActionOp.WAIT:
        _wait_ms(action.ms)


def _enum_field(value: Mapping[str, object], field: str, enum: type[Enum]) -> Enum:
    if field not in value:
        raise ValueError(f"{field} is required")
    return _enum_value(value[field], enum, field)


def _enum_value(value: object, enum: type[Enum], field: str) -> Enum:
    try:
        return enum(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} is not in the closed vocabulary") from None


def _look_target(value: object) -> str:
    if value in ("person", "scene"):
        return str(value)
    if isinstance(value, str) and value.startswith("obj:"):
        object_id = value.removeprefix("obj:")
        if _SAFE_OBJECT_ID.fullmatch(object_id):
            return value
    raise ValueError("target must be person, scene, or obj:<safe-id>")


def _optional_finite_number(
    value: object, field: str, *, positive: bool = False
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _wait_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("wait ms must be an integer")
    if not 0 <= value <= WAIT_MAX_MS:
        raise ValueError(f"wait ms must be between 0 and {WAIT_MAX_MS}")
    return value


def _parse_observed_object(value: object) -> ObservedObject:
    if not isinstance(value, Mapping):
        raise TypeError("visible object must be an object")
    required = {"label", "canonical"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"visible object is missing {sorted(missing)!r}")

    match_provided = "match" in value
    match = value.get("match")
    if match is not None and (
        not isinstance(match, str) or _SAFE_OBJECT_ID.fullmatch(match) is None
    ):
        LOGGER.warning("dropping invalid observation match %r", match)
        match = None
    label = _plain_text(value["label"], "label")
    canonical = normalize_canonical_label(value["canonical"])
    attributes = value.get("attributes", [])
    if not isinstance(attributes, list):
        raise ValueError("attributes must be an array")
    normalized_attributes = tuple(
        dict.fromkeys(_plain_text(attribute, "attribute") for attribute in attributes)
    )
    raw_bbox = value.get("bbox_norm")
    if raw_bbox is None:
        bbox = None
    else:
        try:
            bbox = _bbox(raw_bbox)
        except ValueError as error:
            LOGGER.warning("dropping invalid observation bbox: %s", error)
            bbox = None
    return ObservedObject(
        match=match,
        label=label,
        canonical=canonical,
        attributes=normalized_attributes,
        bbox_norm=bbox,
        match_provided=match_provided,
    )


def _plain_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_norm must contain x, y, width, and height")
    numbers = tuple(_required_finite_number(part, "bbox_norm") for part in value)
    x, y, width, height = numbers
    if not all(0.0 <= part <= 1.0 for part in numbers):
        raise ValueError("bbox_norm values must be between 0 and 1")
    right = x + width
    bottom = y + height
    if width > 0 and height > 0 and x >= 0 and y >= 0 and right <= 1.0 and bottom <= 1.0:
        return x, y, width, height
    if (
        width > 0
        and height > 0
        and right <= 1.0 + BBOX_EDGE_TOLERANCE
        and bottom <= 1.0 + BBOX_EDGE_TOLERANCE
    ):
        clipped_width = min(1.0, right) - x
        clipped_height = min(1.0, bottom) - y
        if clipped_width > 0 and clipped_height > 0:
            return x, y, clipped_width, clipped_height
    raise ValueError("bbox_norm must describe a positive box inside the frame")


def _required_finite_number(value: object, field: str) -> float:
    parsed = _optional_finite_number(value, field)
    if parsed is None:
        raise ValueError(f"{field} must be a finite number")
    return parsed


__all__ = [
    "Action",
    "ActionOp",
    "BBOX_EDGE_TOLERANCE",
    "GestureName",
    "LightPattern",
    "LightPreset",
    "ObservedObject",
    "OBSERVATION_MAX_VISIBLE",
    "ObservationPrior",
    "ObservationResponse",
    "PlanResponse",
    "PostureName",
    "ResponseSchemaError",
    "SAY_MAX_CHARACTERS",
    "SfxName",
    "WAIT_MAX_MS",
    "is_durable_scene_canonical",
    "normalize_canonical_label",
    "parse_observation_response",
    "parse_plan_response",
]
