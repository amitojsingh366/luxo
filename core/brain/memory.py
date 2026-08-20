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
from typing import Any, Final, Iterable, Sequence

from .schema import is_durable_scene_canonical, normalize_canonical_label


MAX_SCENE_OBJECTS: Final = 10
_LEGACY_MAX_OBJECTS: Final = 19
_FORMAT_VERSION: Final = 2
_ENVELOPE_FIELDS: Final = frozenset({"version", "next_id", "objects"})
_LEGACY_FIELDS: Final = frozenset(
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
_FIELDS: Final = _LEGACY_FIELDS | {"priority", "requested_at"}
_MEMORY_PRIORITIES: Final = frozenset({"incidental", "requested"})
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
    priority: str = "incidental"
    requested_at: float | None = None

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
        if not isinstance(self.priority, str) or self.priority not in _MEMORY_PRIORITIES:
            raise SceneMemoryError("priority must be incidental or requested")
        if self.requested_at is None:
            requested_at = None
        else:
            requested_at = _number(self.requested_at, "requested_at")
            if requested_at < 0.0:
                raise SceneMemoryError("requested_at must be nonnegative")
        if (self.priority == "requested") != (requested_at is not None):
            raise SceneMemoryError("requested priority requires requested_at")

        object.__setattr__(self, "id", object_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "bbox_norm", bbox)
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "last_seen", last_seen)
        object.__setattr__(self, "requested_at", requested_at)


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


def _eligible_records(records: Iterable[SceneObject]) -> tuple[SceneObject, ...]:
    return tuple(
        record for record in records if is_durable_scene_canonical(record.canonical)
    )


def _durable_selection(records: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
    """Remove non-object noise, then select the ten strongest durable facts."""

    eligible = _eligible_records(records)
    if len(eligible) <= MAX_SCENE_OBJECTS:
        return eligible
    return _ranked_selection(eligible)


def _ranked_selection(records: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
    return tuple(
        sorted(
            records,
            key=_retention_key,
        )[:MAX_SCENE_OBJECTS]
    )


def _retention_key(record: SceneObject) -> tuple[bool, float, bool, float, object]:
    """Rank requested facts ahead of incidental visual background."""

    return (
        record.priority != "requested",
        -(record.requested_at or 0.0),
        not record.present,
        -record.last_seen,
        _id_key(record),
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
        "priority": record.priority,
        "requested_at": record.requested_at,
    }


def _next_id_from_strings(values: Iterable[str]) -> int:
    return max(
        (
            int(match.group(1))
            for value in values
            if (match := _OBJECT_ID.fullmatch(value))
        ),
        default=0,
    ) + 1


def _next_numeric_id(records: Iterable[SceneObject]) -> int:
    return _next_id_from_strings(record.id for record in records)


class SceneMemoryStore:
    """Persist bounded facts and allocator state in a versioned JSON envelope."""

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

        legacy = isinstance(payload, list)
        if legacy:
            raw_records = payload
            next_id = None
        elif isinstance(payload, dict) and frozenset(payload) == _ENVELOPE_FIELDS:
            if payload["version"] != _FORMAT_VERSION:
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: unsupported version"
                )
            next_id = payload["next_id"]
            raw_records = payload["objects"]
            if (
                isinstance(next_id, bool)
                or not isinstance(next_id, int)
                or next_id < 1
                or not isinstance(raw_records, list)
            ):
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: invalid allocator envelope"
                )
        else:
            raise SceneMemoryError(
                f"invalid scene memory {self._path}: root must be a legacy list "
                "or versioned envelope"
            )
        records: list[SceneObject] = []
        migrated_record = False
        for index, value in enumerate(raw_records):
            if not isinstance(value, dict):
                raise SceneMemoryError(
                    f"invalid scene memory {self._path}: entry {index} must be an object"
                )
            keys = frozenset(value)
            if keys == _LEGACY_FIELDS:
                value = {**value, "priority": "incidental", "requested_at": None}
                keys = _FIELDS
                migrated_record = True
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
            minimum_next_id = _next_numeric_id(validated)
            if next_id is None:
                next_id = minimum_next_id
            elif next_id < minimum_next_id:
                raise SceneMemoryError("next_id is below an existing object id")
            selected = _durable_selection(validated)
        except SceneMemoryError as exc:
            raise SceneMemoryError(f"invalid scene memory {self._path}: {exc}") from exc
        if legacy or migrated_record or len(selected) != len(validated):
            try:
                self._save(selected, next_id)
            except OSError as exc:
                raise SceneMemoryError(
                    f"cannot migrate scene memory {self._path}: {exc}"
                ) from exc
        return selected

    def save(self, objects: tuple[SceneObject, ...]) -> None:
        supplied = _validate_collection(objects, maximum=None)
        records = _validate_collection(_eligible_records(supplied))
        next_id = max(_next_numeric_id(supplied), self._stored_next_id())
        self._save(records, next_id)

    def _save(self, records: tuple[SceneObject, ...], next_id: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "version": _FORMAT_VERSION,
                        "next_id": next_id,
                        "objects": [_record_dict(record) for record in records],
                    },
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

    def _stored_next_id(self) -> int:
        """Best-effort allocator floor used when explicitly replacing facts."""

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return 1
        if isinstance(payload, dict):
            value = payload.get("next_id")
            return value if isinstance(value, int) and not isinstance(value, bool) else 1
        if isinstance(payload, list):
            identifiers = tuple(
                value.get("id")
                for value in payload
                if isinstance(value, dict) and isinstance(value.get("id"), str)
            )
            return _next_id_from_strings(identifiers)
        return 1

    def update(self, objects: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
        supplied = _validate_collection(objects, maximum=None)
        observed = _validate_collection(_eligible_records(supplied))
        existing = self.load()
        by_id = {record.id: record for record in existing}
        next_id = max(_next_numeric_id(existing), self._stored_next_id())

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
        historical = [
            replace(record, present=False)
            for record in existing
            if record.id not in visible_ids
        ]
        pool = (*visible, *historical)
        order = {record.id: index for index, record in enumerate(pool)}
        result = _validate_collection(
            tuple(
                sorted(
                    pool,
                    key=lambda record: (
                        record.priority != "requested",
                        -(record.requested_at or 0.0),
                        not record.present,
                        -record.last_seen,
                        order[record.id],
                        _id_key(record),
                    ),
                )[:MAX_SCENE_OBJECTS]
            )
        )
        self._save(result, next_id)
        return result

    def touch_requested(
        self,
        object_ids: Sequence[str],
        requested_at: float,
    ) -> tuple[SceneObject, ...]:
        """Promote direct historical references without changing scene facts.

        The cloud may refer only to ids already supplied in its memory context.
        Touching those ids records user interest for retention, but deliberately
        does not claim that an absent object became visible or alter its saved
        visual attributes.
        """

        if isinstance(object_ids, (str, bytes)):
            raise SceneMemoryError("object_ids must be a sequence of strings")
        timestamp = _number(requested_at, "requested_at")
        if timestamp < 0.0:
            raise SceneMemoryError("requested_at must be nonnegative")
        identifiers: list[str] = []
        seen: set[str] = set()
        try:
            values = tuple(object_ids)
        except TypeError as exc:
            raise SceneMemoryError("object_ids must be a sequence of strings") from exc
        for value in values:
            identifier = _text(value, "object id", compact_safe=True)
            if _SAFE_ID.fullmatch(identifier) is None:
                raise SceneMemoryError(
                    "object id must contain only letters, digits, '.', '_', or '-'"
                )
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

        existing = self.load()
        if not identifiers:
            return existing
        selected = frozenset(identifiers)
        touched = tuple(
            replace(
                record,
                priority="requested",
                requested_at=max(record.requested_at or 0.0, timestamp),
            )
            if record.id in selected
            else record
            for record in existing
        )
        if touched == existing:
            return existing
        result = _ranked_selection(touched)
        self._save(result, self._stored_next_id())
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
