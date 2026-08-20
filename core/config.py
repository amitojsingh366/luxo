"""Immutable configuration loading for the character core.

``default.yaml`` is deliberately JSON-compatible YAML. JSON is a strict YAML
subset, so the assembled application can validate configuration without adding
a YAML dependency that the PRD does not require.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, Union


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or unsafe."""


ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = Union[
    ConfigScalar,
    "FrozenConfig",
    tuple["ConfigValue", ...],
]


class FrozenConfig(Mapping[str, ConfigValue]):
    """A recursively immutable config mapping with attribute-style access."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(
            self,
            "_values",
            MappingProxyType({key: _freeze(value) for key, value in values.items()}),
        )

    def __getitem__(self, key: str) -> ConfigValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> ConfigValue:
        try:
            return self._values[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: object) -> None:
        raise TypeError("configuration is immutable")

    def __repr__(self) -> str:
        return f"FrozenConfig({self._values!r})"


def _freeze(value: Any) -> ConfigValue:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ConfigError("configuration keys must be strings")
        return FrozenConfig(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ConfigError(f"unsupported configuration value: {value!r}")


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

_EXPECTED_JOINTS = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "neck_yaw",
    "head_pitch",
)

_EXPECTED_OUTPUT_ORDER = (
    "sum_layers",
    "spring_damper",
    "velocity_clamp",
    "soft_limit_clamp",
)

_EXPECTED_ENUMS: dict[str, tuple[str, ...]] = {
    "states": (
        "BOOT", "DORMANT", "NOTICING", "ENGAGED", "LISTENING", "THINKING",
        "SPEAKING", "INSPECTING", "ACTING", "DISENGAGING",
    ),
    "ops": (
        "gesture", "look_at", "light", "sfx", "scan", "observe", "posture", "wait",
    ),
    "gestures": (
        "perk_up", "nod", "double_take", "recoil", "lean_in", "bounce",
        "shake_no", "settle", "droop", "regard",
    ),
    "look_at_targets": ("person", "obj:<id>", "scene"),
    "light_presets": (
        "warm_idle", "warm_bright", "cool_dim", "curious_focus",
        "thinking_pulse", "excited_flash", "sad_fade",
    ),
    "light_patterns": ("steady", "pulse", "flicker", "blink"),
    "sfx": (
        "chirp_up", "chirp_found", "boing", "whirr_short", "hmm", "blip_sad",
        "fanfare_small", "click",
    ),
    "postures": ("rest", "alert", "slump", "stoop", "crane"),
}


def load_config(path: str | Path | None = None) -> FrozenConfig:
    """Load, validate, and recursively freeze a Luxo configuration file."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"configuration is not JSON-compatible YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    _validate(raw)
    return FrozenConfig(raw)


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ConfigError(f"{path} must be {qualifier}")
    return number


def _require_keys(value: Mapping[str, Any], path: str, keys: set[str]) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ConfigError(f"{path} keys differ; missing={missing}, unknown={unknown}")


def _validate(raw: dict[str, Any]) -> None:
    top_keys = {
        "name", "runtime", "network", "gaze", "speech", "vision", "joints",
        "animation", "brain", "vocabulary",
    }
    _require_keys(raw, "root", top_keys)
    if raw["name"] != "Luxo":
        raise ConfigError("name must be the owner-selected value 'Luxo'")

    runtime = _object(raw["runtime"], "runtime")
    _require_keys(
        runtime, "runtime",
        {"python", "fsm_hz", "animation_hz", "body_state_hz", "gaze_stale_s", "animation_jitter_p99_ms"},
    )
    if runtime["python"] != "3.12":
        raise ConfigError("runtime.python must be the locked value '3.12'")
    for key in runtime.keys() - {"python"}:
        _number(runtime[key], f"runtime.{key}", positive=True)

    network = _object(raw["network"], "network")
    _require_keys(network, "network", {"websocket_url"})
    if network["websocket_url"] != "ws://127.0.0.1:8765":
        raise ConfigError("network.websocket_url must remain loopback port 8765")

    gaze = _object(raw["gaze"], "gaze")
    _require_keys(
        gaze, "gaze",
        {
            "yaw_threshold_deg", "pitch_threshold_deg", "min_detection_confidence",
            "engage_dwell_s", "notice_hold_s", "disengage_dwell_s",
            "disengage_droop_s", "ema_alpha", "publish_hz",
        },
    )
    for key, value in gaze.items():
        _number(value, f"gaze.{key}", positive=True)
    if not 0 < float(gaze["min_detection_confidence"]) <= 1:
        raise ConfigError("gaze.min_detection_confidence must be in (0, 1]")
    if not 0 < float(gaze["ema_alpha"]) <= 1:
        raise ConfigError("gaze.ema_alpha must be in (0, 1]")

    speech = _object(raw["speech"], "speech")
    _require_keys(
        speech, "speech",
        {
            "vad_trailing_silence_ms", "minimum_utterance_ms", "input_sample_hz",
            "output_sample_hz", "tts_chunk_max_bytes", "envelope_hz", "piper_length_scale",
        },
    )
    for key, value in speech.items():
        _number(value, f"speech.{key}", positive=True)

    vision = _object(raw["vision"], "vision")
    _require_keys(vision, "vision", {"capture_longest_edge_px", "jpeg_quality", "capture_max_bytes"})
    for key, value in vision.items():
        _number(value, f"vision.{key}", positive=True)
    if float(vision["jpeg_quality"]) > 100:
        raise ConfigError("vision.jpeg_quality must be at most 100")

    _validate_joints(raw["joints"])
    _validate_animation(raw["animation"])
    _validate_brain(raw["brain"])
    _validate_vocabulary(raw["vocabulary"])


def _validate_joints(value: Any) -> None:
    joints = _object(value, "joints")
    if tuple(joints) != _EXPECTED_JOINTS:
        raise ConfigError(f"joints must be ordered as {_EXPECTED_JOINTS!r}")
    fields = {"urdf_name", "hard_min", "hard_max", "soft_min", "soft_max", "velocity_max"}
    for name, raw_joint in joints.items():
        joint = _object(raw_joint, f"joints.{name}")
        _require_keys(joint, f"joints.{name}", fields)
        if joint["urdf_name"] != f"{name}_joint":
            raise ConfigError(f"joints.{name}.urdf_name does not match the body")
        hard_min = _number(joint["hard_min"], f"joints.{name}.hard_min")
        hard_max = _number(joint["hard_max"], f"joints.{name}.hard_max")
        soft_min = _number(joint["soft_min"], f"joints.{name}.soft_min")
        soft_max = _number(joint["soft_max"], f"joints.{name}.soft_max")
        _number(joint["velocity_max"], f"joints.{name}.velocity_max", positive=True)
        if not hard_min < soft_min < soft_max < hard_max:
            raise ConfigError(f"joints.{name} limits must satisfy hard < soft < soft < hard")


def _validate_animation(value: Any) -> None:
    animation = _object(value, "animation")
    _require_keys(animation, "animation", {"idle", "timing", "look_at", "springs", "output_order"})
    if tuple(animation["output_order"]) != _EXPECTED_OUTPUT_ORDER:
        raise ConfigError("animation.output_order must preserve the safety pipeline")

    idle = _object(animation["idle"], "animation.idle")
    _require_keys(idle, "animation.idle", {"shoulder_amplitude_rad", "breathing_hz", "noise_amplitude_rad"})
    for key, item in idle.items():
        _number(item, f"animation.idle.{key}", positive=True)

    timing = _object(animation["timing"], "animation.timing")
    timing_fields = {
        "anticipation_min_ms", "anticipation_max_ms", "arrival_stagger_min_ms",
        "arrival_stagger_max_ms", "strong_pose_hold_min_ms", "blink_intensity_scale",
        "blink_duration_ms", "blink_head_bob_rad", "arrival_order",
    }
    _require_keys(timing, "animation.timing", timing_fields)
    for key in timing_fields - {"arrival_order", "blink_head_bob_rad"}:
        _number(timing[key], f"animation.timing.{key}", positive=True)
    _number(timing["blink_head_bob_rad"], "animation.timing.blink_head_bob_rad")
    if tuple(timing["arrival_order"]) != _EXPECTED_JOINTS:
        raise ConfigError("animation.timing.arrival_order must follow mass order")

    look_at = _object(animation["look_at"], "animation.look_at")
    _require_keys(
        look_at, "animation.look_at",
        {"neck_error_clamp_rad", "neck_recenter_time_constant_s", "front_world_axis", "positive_head_pitch_direction"},
    )
    _number(look_at["neck_error_clamp_rad"], "animation.look_at.neck_error_clamp_rad", positive=True)
    _number(look_at["neck_recenter_time_constant_s"], "animation.look_at.neck_recenter_time_constant_s", positive=True)
    if look_at["front_world_axis"] != "-x" or look_at["positive_head_pitch_direction"] != "down":
        raise ConfigError("animation.look_at must preserve the head joint pi-flip convention")

    springs = _object(animation["springs"], "animation.springs")
    if tuple(springs) != _EXPECTED_JOINTS:
        raise ConfigError("animation.springs must define all joints in body order")
    for name, raw_spring in springs.items():
        spring = _object(raw_spring, f"animation.springs.{name}")
        _require_keys(spring, f"animation.springs.{name}", {"omega", "zeta"})
        _number(spring["omega"], f"animation.springs.{name}.omega", positive=True)
        _number(spring["zeta"], f"animation.springs.{name}.zeta", positive=True)


def _validate_brain(value: Any) -> None:
    brain = _object(value, "brain")
    _require_keys(
        brain, "brain",
        {
            "profile", "say_max_characters", "recent_exchange_pairs",
            "response_stream_max_bytes", "json_repair_retries", "profiles",
        },
    )
    if brain["profile"] != "free":
        raise ConfigError("brain.profile must be the owner-selected value 'free'")
    for key in ("say_max_characters", "recent_exchange_pairs", "response_stream_max_bytes", "json_repair_retries"):
        _number(brain[key], f"brain.{key}", positive=True)
    profiles = _object(brain["profiles"], "brain.profiles")
    _require_keys(profiles, "brain.profiles", {"private", "free"})
    private = _object(profiles["private"], "brain.profiles.private")
    free = _object(profiles["free"], "brain.profiles.free")
    _require_keys(private, "brain.profiles.private", {"model", "provider"})
    _require_keys(free, "brain.profiles.free", {"model", "provider"})
    if private["model"] is not None or free["model"] is not None:
        raise ConfigError("brain profile models remain unset until the owner decides")
    expected_private = {"zdr": True, "data_collection": "deny", "allow_fallbacks": False}
    if private["provider"] != expected_private or free["provider"] != {}:
        raise ConfigError("brain profile placeholders do not match the locked PRD")


def _validate_vocabulary(value: Any) -> None:
    vocabulary = _object(value, "vocabulary")
    _require_keys(vocabulary, "vocabulary", set(_EXPECTED_ENUMS))
    for name, expected in _EXPECTED_ENUMS.items():
        actual = vocabulary[name]
        if not isinstance(actual, list) or tuple(actual) != expected:
            raise ConfigError(f"vocabulary.{name} must remain the closed PRD enum")
