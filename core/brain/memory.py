"""Small, local scene memory with a privacy-limited prompt projection.

The store records perception facts only. It deliberately contains no intent,
comparison policy, embeddings, or model-facing full-record serializer.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Iterable

from .schema import normalize_canonical_label


MAX_SCENE_OBJECTS: Final = 10
_LEGACY_MAX_OBJECTS: Final = 19
_FIELDS: Final = frozenset(
    {
        "id",
        "label",
        "canonical",
        "attributes",
        "bbox_norm",
        "first_seen",
        "last_seen",
        "present",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_OBJECT_ID = re.compile(r"obj_(\d+)\Z")
_COMPACT_DELIMITERS: Final = frozenset(":;(),")
_SENSITIVE_ATTRIBUTE_TERMS: Final = frozenset(
    {
        "address",
        "age",
        "biometric",
        "email",
        "ethnicity",
        "face",
        "facial",
        "gender",
        "health",
        "identity",
        "location",
        "medical",
        "name",
        "phone",
        "race",
        "religion",
    }
)


class SceneMemoryError(ValueError):
    """A scene-memory file or record is malformed."""


@dataclass(frozen=True, slots=True)
class CloudSceneObject:
    """Privacy-limited scene fact safe for OpenRouter text payloads.

    Stable identity lets the model select ``look_at`` targets. Canonical names
    and filtered attributes provide the visible facts it needs to reason. Raw
    labels, bounding boxes, timestamps, and local presence metadata never have
    a representation in this projection.
    """

    id: str
    canonical: str
    attributes: tuple[str, ...]

    def __post_init__(self) -> None:
        object_id = _text(self.id, "id", compact_safe=True)
        if _SAFE_ID.fullmatch(object_id) is None:
            raise SceneMemoryError("id must contain only letters, digits, '.', '_', or '-'")
        try:
            canonical = normalize_canonical_label(self.canonical)
        except ValueError as exc:
            raise SceneMemoryError("canonical must be a safe canonical label") from exc
        if isinstance(self.attributes, (str, bytes)):
            raise SceneMemoryError("attributes must be a sequence of strings")
        try:
            attributes = tuple(
                sorted(
                    {
                        _text(attribute, "attribute", compact_safe=True).casefold()
                        for attribute in self.attributes
                    }
                )
            )
        except TypeError as exc:
            raise SceneMemoryError("attributes must be a sequence of strings") from exc
        object.__setattr__(self, "id", object_id)
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "attributes", cloud_safe_attributes(attributes))


def _text(value: object, field: str, *, compact_safe: bool = False) -> str:
    if not isinstance(value, str):
        raise SceneMemoryError(f"{field} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise SceneMemoryError(f"{field} must not be empty")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise SceneMemoryError(f"{field} contains control characters")
    if compact_safe and any(character in _COMPACT_DELIMITERS for character in normalized):
        raise SceneMemoryError(f"{field} contains a reserved compact-line delimiter")
    return normalized


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneMemoryError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SceneMemoryError(f"{field} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class SceneObject:
    id: str
    label: str
    canonical: str
    attributes: tuple[str, ...]
    bbox_norm: tuple[float, float, float, float] | None
    first_seen: float
    last_seen: float
    present: bool

    def __post_init__(self) -> None:
        object_id = _text(self.id, "id", compact_safe=True)
        if _SAFE_ID.fullmatch(object_id) is None:
            raise SceneMemoryError("id must contain only letters, digits, '.', '_', or '-'")
        label = _text(self.label, "label")
        canonical = _text(self.canonical, "canonical", compact_safe=True).casefold()

        if isinstance(self.attributes, (str, bytes)):
            raise SceneMemoryError("attributes must be a sequence of strings")
        try:
            attributes = tuple(
                sorted(
                    {
                        _text(attribute, "attribute", compact_safe=True).casefold()
                        for attribute in self.attributes
                    }
                )
            )
        except TypeError as exc:
            raise SceneMemoryError("attributes must be a sequence of strings") from exc

        if self.bbox_norm is None:
            bbox = None
        else:
            if isinstance(self.bbox_norm, (str, bytes)):
                raise SceneMemoryError("bbox_norm must contain exactly four numbers")
            try:
                bbox_values = tuple(self.bbox_norm)
            except TypeError as exc:
                raise SceneMemoryError("bbox_norm must contain exactly four numbers") from exc
            if len(bbox_values) != 4:
                raise SceneMemoryError("bbox_norm must contain exactly four numbers")
            bbox = tuple(
                _number(value, f"bbox_norm[{index}]")
                for index, value in enumerate(bbox_values)
            )
            x, y, width, height = bbox
            if not all(0.0 <= value <= 1.0 for value in bbox):
                raise SceneMemoryError("bbox_norm values must be within [0, 1]")
            if width <= 0.0 or height <= 0.0:
                raise SceneMemoryError("bbox_norm width and height must be positive")
            if x + width > 1.0 or y + height > 1.0:
                raise SceneMemoryError("bbox_norm must not extend beyond the frame")

        first_seen = _number(self.first_seen, "first_seen")
        last_seen = _number(self.last_seen, "last_seen")
        if first_seen < 0.0 or last_seen < 0.0:
            raise SceneMemoryError("timestamps must be nonnegative")
        if first_seen > last_seen:
            raise SceneMemoryError("first_seen must not exceed last_seen")
        if type(self.present) is not bool:
            raise SceneMemoryError("present must be a boolean")

        object.__setattr__(self, "id", object_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "bbox_norm", bbox)
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "last_seen", last_seen)


def _id_key(item: SceneObject) -> tuple[int, int | str, str]:
    match = _OBJECT_ID.fullmatch(item.id)
    if match is None:
        return (1, item.id, item.canonical)
    return (0, int(match.group(1)), item.canonical)


def _validate_collection(
    objects: Iterable[SceneObject], *, maximum: int | None = MAX_SCENE_OBJECTS
) -> tuple[SceneObject, ...]:
    records = tuple(objects)
    if maximum is not None and len(records) > maximum:
        raise SceneMemoryError(f"scene memory supports at most {maximum} objects")
    if any(not isinstance(record, SceneObject) for record in records):
        raise SceneMemoryError("scene memory entries must be SceneObject records")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise SceneMemoryError("scene memory contains duplicate ids")
    return records


def _legacy_selection(records: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
    """Select the ten strongest records while migrating an old 19-record file."""

    if len(records) <= MAX_SCENE_OBJECTS:
        return records
    return tuple(
        sorted(
            records,
            key=lambda record: (
                not record.present,
                -record.last_seen,
                _id_key(record),
            ),
        )[:MAX_SCENE_OBJECTS]
    )


def _record_dict(record: SceneObject) -> dict[str, Any]:
    return {
        "id": record.id,
        "label": record.label,
        "canonical": record.canonical,
        "attributes": list(record.attributes),
        "bbox_norm": None if record.bbox_norm is None else list(record.bbox_norm),
        "first_seen": record.first_seen,
        "last_seen": record.last_seen,
        "present": record.present,
    }


class SceneMemoryStore:
    """Persist a bounded, flat list of scene facts as deterministic JSON."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[SceneObject, ...]:
        try:
            with self._path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            return ()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SceneMemoryError(f"cannot load scene memory {self._path}: {exc}") from exc

        if not isinstance(payload, list):
            raise SceneMemoryError(f"invalid scene memory {self._path}: root must be a JSON list")
        records: list[SceneObject] = []
        for index, value in enumerate(payload):
            if not isinstance(value, dict):
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: entry {index} must be an object"
                )
            keys = frozenset(value)
            if keys != _FIELDS:
                missing = sorted(_FIELDS - keys)
                unknown = sorted(keys - _FIELDS)
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: entry {index} has "
                    f"missing fields {missing} and unknown fields {unknown}"
                )
            try:
                records.append(SceneObject(**value))
            except (SceneMemoryError, TypeError) as exc:
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: entry {index}: {exc}"
                ) from exc
        try:
            validated = _validate_collection(records, maximum=_LEGACY_MAX_OBJECTS)
            selected = _legacy_selection(validated)
        except SceneMemoryError as exc:
            raise SceneMemoryError(f"invalid scene memory {self._path}: {exc}") from exc
        if len(selected) != len(validated):
            try:
                self.save(selected)
            except OSError as exc:
                raise SceneMemoryError(
                    f"cannot migrate scene memory {self._path}: {exc}"
                ) from exc
        return selected

    def save(self, objects: tuple[SceneObject, ...]) -> None:
        records = _validate_collection(objects)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    [_record_dict(record) for record in records],
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self._path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def update(self, objects: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
        observed = _validate_collection(objects)
        existing = self.load()
        by_id = {record.id: record for record in existing}
        next_id = max(
            (
                int(match.group(1))
                for record in existing
                if (match := _OBJECT_ID.fullmatch(record.id))
            ),
            default=0,
        ) + 1

        visible: list[SceneObject] = []
        for record in observed:
            prior = by_id.get(record.id)
            if prior is None:
                new_id = f"obj_{next_id:03d}"
                next_id += 1
                visible.append(replace(record, id=new_id))
                continue
            visible.append(
                replace(
                    record,
                    id=prior.id,
                    first_seen=min(prior.first_seen, record.first_seen),
                )
            )

        visible_ids = {record.id for record in visible}
        historical = sorted(
            (
                replace(record, present=False)
                for record in existing
                if record.id not in visible_ids
            ),
            key=lambda record: (-record.last_seen, _id_key(record)),
        )
        remaining = MAX_SCENE_OBJECTS - len(visible)
        result = _validate_collection((*visible, *historical[:remaining]))
        self.save(result)
        return result

    def compact_line(self) -> str:
        parts: list[str] = []
        for record in self.load():
            attributes = cloud_safe_attributes(record.attributes)
            suffix = f"({','.join(attributes)})" if attributes else ""
            parts.append(f"{record.id}:{record.canonical}{suffix}")
        return "; ".join(parts)

    def currently_visible(self) -> tuple[CloudSceneObject, ...]:
        """Return the bounded present-only scene projection for text calls."""

        return tuple(
            CloudSceneObject(
                id=record.id,
                canonical=record.canonical,
                attributes=cloud_safe_attributes(record.attributes),
            )
            for record in self.load()
            if record.present
        )


def _attribute_terms(attribute: str) -> frozenset[str]:
    return frozenset(re.findall(r"[\w]+", attribute.casefold()))


def cloud_safe_attributes(attributes: Iterable[str]) -> tuple[str, ...]:
    """Remove sensitive extracted attributes from every text-cloud payload."""

    return tuple(
        attribute
        for attribute in attributes
        if not (_attribute_terms(attribute) & _SENSITIVE_ATTRIBUTE_TERMS)
    )


__all__ = [
    "CloudSceneObject",
    "MAX_SCENE_OBJECTS",
    "SceneMemoryError",
    "SceneMemoryStore",
    "SceneObject",
    "cloud_safe_attributes",
]
