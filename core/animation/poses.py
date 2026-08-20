"""Immutable loading and lookup for Luxo's closed canonical pose library."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, cast

from . import JOINT_NAMES, JointName, JointVector


PoseName: TypeAlias = Literal["rest", "alert", "slump", "stoop", "crane"]

POSE_NAMES: Final[tuple[PoseName, ...]] = (
    "rest",
    "alert",
    "slump",
    "stoop",
    "crane",
)
HOME_POSE_NAME: Final[PoseName] = "rest"
ENGAGED_BASE_YAW_RAD: Final = 0.44
DEFAULT_POSES_PATH = Path(__file__).resolve().parents[2] / "config" / "poses.yaml"

SOFT_LIMITS: Final[Mapping[JointName, tuple[float, float]]] = MappingProxyType(
    {
        "base_yaw": (-2.450, 2.450),
        "shoulder_pitch": (-0.650, 0.950),
        "elbow_pitch": (-1.700, 0.300),
        "neck_yaw": (-1.250, 1.250),
        "head_pitch": (-0.800, 0.600),
    }
)


class PoseError(ValueError):
    """Raised when a pose library is malformed or unsafe."""


class UnknownPoseError(KeyError):
    """Raised when a caller requests a name outside the closed pose set."""


class PoseLibrary(Mapping[PoseName, JointVector]):
    """An immutable mapping from a closed pose name to a joint vector."""

    __slots__ = ("_poses",)

    def __init__(self, poses: Mapping[str, JointVector]) -> None:
        actual_names = set(poses)
        expected_names = set(POSE_NAMES)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unknown = sorted(actual_names - expected_names)
            raise PoseError(
                f"pose names differ; missing={missing}, unknown={unknown}"
            )

        copied: dict[PoseName, JointVector] = {}
        for raw_name in POSE_NAMES:
            vector = poses[raw_name]
            if not isinstance(vector, JointVector):
                raise PoseError(f"pose {raw_name!r} must be a JointVector")
            _validate_vector(raw_name, vector)
            copied[raw_name] = vector

        if copied[HOME_POSE_NAME] == JointVector():
            raise PoseError("rest cannot be the all-zero joint vector")
        object.__setattr__(self, "_poses", MappingProxyType(copied))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("pose library is immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("pose library is immutable")

    def __getitem__(self, name: PoseName) -> JointVector:
        try:
            return self._poses[name]
        except KeyError as exc:
            expected = ", ".join(POSE_NAMES)
            raise UnknownPoseError(
                f"unknown pose {name!r}; expected one of: {expected}"
            ) from exc

    def __iter__(self) -> Iterator[PoseName]:
        return iter(POSE_NAMES)

    def __len__(self) -> int:
        return len(POSE_NAMES)

    @property
    def home_name(self) -> PoseName:
        """Return the one valid neutral/home pose name."""

        return HOME_POSE_NAME

    @property
    def home(self) -> JointVector:
        """Return the neutral/home joint vector."""

        return self._poses[HOME_POSE_NAME]

    def pose(self, name: str) -> JointVector:
        """Return a typed joint vector or raise a clear unknown-pose error."""

        return self[cast(PoseName, name)]


def load_pose_library(path: str | Path | None = None) -> PoseLibrary:
    """Load and validate JSON-compatible YAML containing Luxo's poses."""

    pose_path = Path(path) if path is not None else DEFAULT_POSES_PATH
    try:
        raw = json.loads(pose_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PoseError(f"cannot read pose library: {pose_path}") from exc
    except json.JSONDecodeError as exc:
        raise PoseError(f"pose library is not JSON-compatible YAML: {exc}") from exc

    root = _object(raw, "root")
    _require_keys(root, "root", {"home", "poses"})
    if root["home"] != HOME_POSE_NAME:
        raise PoseError("home must be 'rest'; no other default pose is valid")

    raw_poses = _object(root["poses"], "poses")
    vectors: dict[str, JointVector] = {}
    for name, raw_vector in raw_poses.items():
        vector = _object(raw_vector, f"poses.{name}")
        _require_keys(vector, f"poses.{name}", set(JOINT_NAMES))
        values = {
            joint: _number(vector[joint], f"poses.{name}.{joint}")
            for joint in JOINT_NAMES
        }
        vectors[name] = JointVector(**values)
    return PoseLibrary(vectors)


def _validate_vector(name: str, vector: JointVector) -> None:
    for joint in JOINT_NAMES:
        value = getattr(vector, joint)
        if not math.isfinite(value):
            raise PoseError(f"poses.{name}.{joint} must be finite")
        lower, upper = SOFT_LIMITS[joint]
        if not lower <= value <= upper:
            raise PoseError(
                f"poses.{name}.{joint}={value} exceeds soft limits "
                f"[{lower}, {upper}]"
            )


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PoseError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise PoseError(f"{path} keys must be strings")
    return value


def _require_keys(value: Mapping[str, Any], path: str, keys: set[str]) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise PoseError(f"{path} keys differ; missing={missing}, unknown={unknown}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PoseError(f"{path} must be finite")
    return number


__all__ = [
    "DEFAULT_POSES_PATH",
    "ENGAGED_BASE_YAW_RAD",
    "HOME_POSE_NAME",
    "POSE_NAMES",
    "PoseError",
    "PoseLibrary",
    "PoseName",
    "SOFT_LIMITS",
    "UnknownPoseError",
    "load_pose_library",
]
