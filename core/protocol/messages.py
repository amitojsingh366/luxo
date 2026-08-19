"""Typed parsing and serialization for Lumen WebSocket frames.

JSON Schema is the validation source of truth.  This module implements the
small JSON Schema subset used by ``schema/messages.schema.json`` with only the
Python standard library, then converts valid objects into immutable dataclasses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "messages.schema.json"


class ProtocolError(ValueError):
    """Raised when a text or binary frame violates the protocol."""


class Direction(str, Enum):
    BROWSER_TO_CORE = "browser-to-core"
    CORE_TO_BROWSER = "core-to-browser"


class BinaryFrameType(IntEnum):
    UTTERANCE_PCM = 0x01
    CAPTURE_JPEG = 0x02
    TTS_PCM = 0x03


BINARY_DIRECTIONS: dict[BinaryFrameType, Direction] = {
    BinaryFrameType.UTTERANCE_PCM: Direction.BROWSER_TO_CORE,
    BinaryFrameType.CAPTURE_JPEG: Direction.BROWSER_TO_CORE,
    BinaryFrameType.TTS_PCM: Direction.CORE_TO_BROWSER,
}


@dataclass(frozen=True, slots=True)
class BinaryFrame:
    kind: BinaryFrameType
    payload: bytes


@dataclass(frozen=True, slots=True)
class CameraSpec:
    w: int
    h: int
    hfov_deg: float


@dataclass(frozen=True, slots=True)
class HelloMessage:
    type: Literal["hello"] = field(default="hello", init=False)
    fps: int = 60
    camera: CameraSpec = field(default_factory=lambda: CameraSpec(640, 480, 60.0))


@dataclass(frozen=True, slots=True)
class GazeMessage:
    t: float
    present: bool
    yaw_deg: float
    pitch_deg: float
    az: float
    el: float
    conf: float
    type: Literal["gaze"] = field(default="gaze", init=False)


@dataclass(frozen=True, slots=True)
class VadMessage:
    t: float
    event: Literal["start"] = "start"
    type: Literal["vad"] = field(default="vad", init=False)


@dataclass(frozen=True, slots=True)
class TtsDoneMessage:
    t: float
    type: Literal["tts_done"] = field(default="tts_done", init=False)


@dataclass(frozen=True, slots=True)
class ErrorMessage:
    where: str
    detail: str
    type: Literal["error"] = field(default="error", init=False)


@dataclass(frozen=True, slots=True)
class JointsState:
    base_yaw: float
    shoulder_pitch: float
    elbow_pitch: float
    neck_yaw: float
    head_pitch: float


@dataclass(frozen=True, slots=True)
class LightState:
    intensity: float
    color_k: int
    pattern: Literal["steady", "pulse", "flicker", "blink"]
    bloom: float


@dataclass(frozen=True, slots=True)
class AudioState:
    speaking: bool
    arousal: float


@dataclass(frozen=True, slots=True)
class ClampCounts:
    vel: int
    limit: int


@dataclass(frozen=True, slots=True)
class TelemetryGaze:
    present: bool
    yaw_deg: float
    pitch_deg: float


FsmState: TypeAlias = Literal[
    "BOOT",
    "DORMANT",
    "NOTICING",
    "ENGAGED",
    "LISTENING",
    "THINKING",
    "SPEAKING",
    "INSPECTING",
    "ACTING",
    "DISENGAGING",
]


@dataclass(frozen=True, slots=True)
class TelemetryState:
    state: FsmState
    plan_depth: int
    memory_count: int
    last_latency_ms: float
    clamps: ClampCounts
    gaze: TelemetryGaze


@dataclass(frozen=True, slots=True)
class BodyStateMessage:
    t: float
    seq: int
    joints: JointsState
    light: LightState
    audio: AudioState
    telemetry: TelemetryState
    type: Literal["body_state"] = field(default="body_state", init=False)


SfxName: TypeAlias = Literal[
    "chirp_up",
    "chirp_found",
    "boing",
    "whirr_short",
    "hmm",
    "blip_sad",
    "fanfare_small",
    "click",
]


@dataclass(frozen=True, slots=True)
class CueMessage:
    sfx: SfxName
    type: Literal["cue"] = field(default="cue", init=False)


@dataclass(frozen=True, slots=True)
class CaptureFrameMessage:
    req_id: str
    type: Literal["capture_frame"] = field(default="capture_frame", init=False)


@dataclass(frozen=True, slots=True)
class SpeakBeginMessage:
    envelope_hz: float
    type: Literal["speak_begin"] = field(default="speak_begin", init=False)


@dataclass(frozen=True, slots=True)
class SpeakEndMessage:
    type: Literal["speak_end"] = field(default="speak_end", init=False)


BrowserToCoreMessage: TypeAlias = (
    HelloMessage | GazeMessage | VadMessage | TtsDoneMessage | ErrorMessage
)
CoreToBrowserMessage: TypeAlias = (
    BodyStateMessage
    | CueMessage
    | CaptureFrameMessage
    | SpeakBeginMessage
    | SpeakEndMessage
)
TextMessage: TypeAlias = BrowserToCoreMessage | CoreToBrowserMessage


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _resolve(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = node.get("$ref")
    if ref is None:
        return node
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise ProtocolError(f"unsupported schema reference {ref!r}")
    try:
        return root["$defs"][ref.removeprefix(prefix)]
    except KeyError as error:
        raise ProtocolError(f"unresolved schema reference {ref!r}") from error


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate(value: Any, node: Mapping[str, Any], root: Mapping[str, Any], path: str) -> None:
    node = _resolve(node, root)
    if "oneOf" in node:
        failures: list[str] = []
        matches = 0
        for candidate in node["oneOf"]:
            try:
                _validate(value, candidate, root, path)
                matches += 1
            except ProtocolError as error:
                failures.append(str(error))
        if matches != 1:
            detail = failures[0] if failures else "matched multiple message schemas"
            raise ProtocolError(f"{path}: expected exactly one message schema ({detail})")
        return

    if "const" in node and value != node["const"]:
        raise ProtocolError(f"{path}: expected {node['const']!r}")
    if "enum" in node and value not in node["enum"]:
        raise ProtocolError(f"{path}: value {value!r} is outside the closed enum")

    expected = node.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ProtocolError(f"{path}: expected object")
        properties = node.get("properties", {})
        required = node.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ProtocolError(f"{path}: missing required fields {missing}")
        if node.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ProtocolError(f"{path}: unknown fields {unknown}")
        for name, child in value.items():
            if name in properties:
                _validate(child, properties[name], root, f"{path}.{name}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise ProtocolError(f"{path}: expected array")
        for index, child in enumerate(value):
            _validate(child, node["items"], root, f"{path}[{index}]")
        if len(value) < node.get("minItems", 0):
            raise ProtocolError(f"{path}: array is too short")
        if "maxItems" in node and len(value) > node["maxItems"]:
            raise ProtocolError(f"{path}: array is too long")
        return
    if expected == "string":
        if not isinstance(value, str):
            raise ProtocolError(f"{path}: expected string")
        if len(value) < node.get("minLength", 0):
            raise ProtocolError(f"{path}: string is too short")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ProtocolError(f"{path}: expected boolean")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(f"{path}: expected integer")
    elif expected == "number":
        if not _number(value):
            raise ProtocolError(f"{path}: expected finite number")
    elif expected == "null" and value is not None:
        raise ProtocolError(f"{path}: expected null")

    if _number(value):
        if "minimum" in node and value < node["minimum"]:
            raise ProtocolError(f"{path}: value is below minimum {node['minimum']}")
        if "maximum" in node and value > node["maximum"]:
            raise ProtocolError(f"{path}: value is above maximum {node['maximum']}")
        if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
            raise ProtocolError(
                f"{path}: value must exceed {node['exclusiveMinimum']}"
            )
        if "exclusiveMaximum" in node and value >= node["exclusiveMaximum"]:
            raise ProtocolError(
                f"{path}: value must be below {node['exclusiveMaximum']}"
            )


def validate_text_object(value: Mapping[str, Any]) -> None:
    """Validate a decoded text message against the canonical schema."""

    _validate(value, load_schema(), load_schema(), "message")


def _message_from_dict(data: dict[str, Any]) -> TextMessage:
    kind = data["type"]
    values = {name: value for name, value in data.items() if name != "type"}
    if kind == "hello":
        values["camera"] = CameraSpec(**values["camera"])
        return HelloMessage(**values)
    if kind == "gaze":
        return GazeMessage(**values)
    if kind == "vad":
        return VadMessage(**values)
    if kind == "tts_done":
        return TtsDoneMessage(**values)
    if kind == "error":
        return ErrorMessage(**values)
    if kind == "body_state":
        telemetry = values["telemetry"]
        values["joints"] = JointsState(**values["joints"])
        values["light"] = LightState(**values["light"])
        values["audio"] = AudioState(**values["audio"])
        values["telemetry"] = TelemetryState(
            **{
                **telemetry,
                "clamps": ClampCounts(**telemetry["clamps"]),
                "gaze": TelemetryGaze(**telemetry["gaze"]),
            }
        )
        return BodyStateMessage(**values)
    if kind == "cue":
        return CueMessage(**values)
    if kind == "capture_frame":
        return CaptureFrameMessage(**values)
    if kind == "speak_begin":
        return SpeakBeginMessage(**values)
    if kind == "speak_end":
        return SpeakEndMessage()
    raise ProtocolError(f"unknown message type {kind!r}")


def parse_text_message(frame: str, direction: Direction | None = None) -> TextMessage:
    """Parse, schema-validate, and type a WebSocket text frame."""

    if not isinstance(frame, str):
        raise ProtocolError("text frame must be str")
    try:
        data = json.loads(frame)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ProtocolError(f"invalid JSON text frame: {error}") from error
    if not isinstance(data, dict):
        raise ProtocolError("text frame must decode to an object")
    validate_text_object(data)
    message = _message_from_dict(data)
    _validate_text_direction(message, direction)
    return message


def _validate_text_direction(message: TextMessage, direction: Direction | None) -> None:
    if direction is None:
        return
    browser_message = isinstance(
        message, (HelloMessage, GazeMessage, VadMessage, TtsDoneMessage, ErrorMessage)
    )
    actual = Direction.BROWSER_TO_CORE if browser_message else Direction.CORE_TO_BROWSER
    if actual is not direction:
        raise ProtocolError(
            f"{message.type!r} travels {actual.value}, not {direction.value}"
        )


def serialize_text_message(
    message: TextMessage | Mapping[str, Any], direction: Direction | None = None
) -> str:
    """Validate and serialize a typed message or mapping as compact JSON."""

    if is_dataclass(message) and not isinstance(message, type):
        data = asdict(message)
    elif isinstance(message, Mapping):
        data = dict(message)
    else:
        raise ProtocolError("message must be a protocol dataclass or mapping")
    validate_text_object(data)
    typed = _message_from_dict(data)
    _validate_text_direction(typed, direction)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def parse_binary_frame(
    frame: bytes | bytearray | memoryview, direction: Direction | None = None
) -> BinaryFrame:
    """Parse a one-byte-prefixed binary WebSocket frame."""

    data = bytes(frame)
    if not data:
        raise ProtocolError("binary frame is missing its one-byte prefix")
    try:
        kind = BinaryFrameType(data[0])
    except ValueError as error:
        raise ProtocolError(f"unknown binary frame prefix 0x{data[0]:02x}") from error
    actual = BINARY_DIRECTIONS[kind]
    if direction is not None and actual is not direction:
        raise ProtocolError(
            f"binary prefix 0x{kind:02x} travels {actual.value}, not {direction.value}"
        )
    return BinaryFrame(kind=kind, payload=data[1:])


def serialize_binary_frame(
    kind: BinaryFrameType, payload: bytes | bytearray | memoryview, direction: Direction | None = None
) -> bytes:
    """Serialize payload with exactly one binary type-prefix byte."""

    try:
        typed_kind = BinaryFrameType(kind)
    except ValueError as error:
        raise ProtocolError(f"unknown binary frame prefix {kind!r}") from error
    actual = BINARY_DIRECTIONS[typed_kind]
    if direction is not None and actual is not direction:
        raise ProtocolError(
            f"binary prefix 0x{typed_kind:02x} travels {actual.value}, not {direction.value}"
        )
    return bytes((typed_kind,)) + bytes(payload)
