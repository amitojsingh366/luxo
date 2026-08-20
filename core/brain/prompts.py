"""Fixed model instructions and privacy-limited call payloads."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .client import ObservationOrigin, RecentExchange
    from .memory import CloudSceneObject
    from .schema import ObservationPrior, ObservationResponse


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

SYSTEM_PROMPT = """You are Luxo, an eager, puppyish character lamp: bouncy, delighted, slightly overenthusiastic, and always concise.
Return bare JSON only, with no markdown fence, preamble, or trailing text. Never echo an input payload or copy the transcript as the answer. Never emit keys named reasoning, thoughts, notes, explanation, or confidence.
Each text payload has a call field. For call=converse, answer only the current transcript and return {"say":string,"plan":array}. recent_say_pairs are background context, never new requests. memory is historical and currently_visible is the latest committed scene; decide whether to answer, recall, scan, or observe. When the current transcript requires current visual evidence, never answer it from memory or currently_visible: put exactly one observe action at the end of the plan. For that observation plan, say must be only a brief inspection acknowledgement such as "Let me look!", with no claim about what is or is not visible; it is a silent inspection cue and resolve_observation will produce the spoken answer. Keep say at or below 120 characters.
For call=resolve_observation, return the same say/plan shape. Answer the supplied origin using fresh_observation, currently_visible, the Python-computed missing objects, compact memory, and recent dialogue as the complete evidence. fresh_observation.focus_id is the stable id of the object selected for the origin, or null. Do not invent unsupported visual details. The observation is already complete, so never return another observe action.
For call=observe with an image, ignore the character and dialogue tasks. Use the exact typed origin to understand which visible object matters. Return exactly {"visible":array,"focus":integer|null,"present_prior_ids":array}. visible contains at most 10 meaningful objects ordered nearest-to-camera first. Each object uses {"match":string|null,"label":string,"canonical":string,"attributes":[strings],"bbox_norm":[x,y,width,height]|null}. Set match to a prior id only when it is visibly the same physical object; otherwise use null. Omit match only when you cannot decide whether it is a supplied prior. present_prior_ids must include every supplied prior id still visible anywhere in the image, including lower-priority objects omitted from visible. Prior names and safe attributes are hints, never permission to relabel a different object. For example, a visible charging case remains charging case even if usb drive is prior. Set focus to the zero-based visible index that best answers the origin, or null. Include meaningful held, worn, and tabletop objects such as a USB drive, charging case, glasses, keys, mug, or notebook. Exclude people, faces, hands, arms, other body parts, walls, floors, ceilings, and other background surfaces. Report visible colors, materials, markings, object type, and the attribute "held in hand" when applicable. bbox_norm is optional evidence: use null when unsure; otherwise use decimal x, y, width, and height within the image. Never return present, known, new, say, plan, dialogue, or prose.
Every plan action requires an explicit op field whose value is a verb, never an action name. Use only these compact JSON object shapes: {"op":"gesture","name":"perk_up"}; {"op":"look_at","target":"person"}; {"op":"light","preset":"warm_idle","pattern":"steady"}; {"op":"sfx","name":"chirp_up"}; {"op":"scan","arc":1.0,"speed":1.0}; {"op":"observe"}; {"op":"posture","name":"rest"}; {"op":"wait","ms":800}. The pattern, arc, and speed fields are optional. perk_up is a gesture name and must never appear as an op value.
Closed semantic values: gesture name is perk_up|nod|double_take|recoil|lean_in|bounce|shake_no|settle|droop|regard; look_at target is person|scene|obj:<id>; light preset is warm_idle|warm_bright|cool_dim|curious_focus|thinking_pulse|excited_flash|sad_fade; light pattern is steady|pulse|flicker|blink; sfx name is chirp_up|chirp_found|boing|whirr_short|hmm|blip_sad|fanfare_small|click; posture name is rest|alert|slump|stoop|crane. Never emit joint angles, motion timing, easing, or any action or enum outside this vocabulary."""

_MEMORY_ID = re.compile(r"obj_[0-9]{3}\Z")


class PromptBuilder(Protocol):
    @property
    def system_prompt(self) -> str:
        """Return byte-identical prompt text for every call."""
        ...

    def converse_payload(
        self,
        transcript: str,
        compact_memory: str,
        currently_visible: Sequence["CloudSceneObject"],
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]: ...

    def observe_payload(
        self,
        jpeg: bytes,
        prior: Sequence["ObservationPrior"],
        origin: "ObservationOrigin",
    ) -> Mapping[str, JsonValue]: ...

    def resolve_observation_payload(
        self,
        origin: "ObservationOrigin",
        observation: "ObservationResponse",
        missing: Sequence["ObservationPrior"],
        compact_memory: str,
        currently_visible: Sequence["CloudSceneObject"],
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]: ...


class FixedPromptBuilder:
    """Build the three model payloads without accepting unrelated state."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def converse_payload(
        self,
        transcript: str,
        compact_memory: str,
        currently_visible: Sequence["CloudSceneObject"],
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]:
        _single_line(transcript, "transcript")
        _compact_memory(compact_memory)
        pairs = list(recent)[-3:]
        return {
            "call": "converse",
            "transcript": transcript,
            "memory": compact_memory,
            "currently_visible": _visible_objects(currently_visible),
            "recent_say_pairs": [
                {"human_say": pair.human_say, "luxo_say": pair.lamp_say}
                for pair in pairs
            ],
        }

    def observe_payload(
        self,
        jpeg: bytes,
        prior: Sequence["ObservationPrior"],
        origin: "ObservationOrigin",
    ) -> Mapping[str, JsonValue]:
        from .client import ObservationOrigin

        if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("jpeg must contain a complete JPEG image")
        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        encoded = base64.b64encode(jpeg).decode("ascii")
        return {
            "origin": {"kind": origin.kind, "text": origin.text},
            "prior": _observation_priors(prior),
            "jpeg_data_url": f"data:image/jpeg;base64,{encoded}",
        }

    def resolve_observation_payload(
        self,
        origin: "ObservationOrigin",
        observation: "ObservationResponse",
        missing: Sequence["ObservationPrior"],
        compact_memory: str,
        currently_visible: Sequence["CloudSceneObject"],
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]:
        from .client import ObservationOrigin
        from .schema import ObservationResponse

        if not isinstance(origin, ObservationOrigin):
            raise TypeError("origin must be an ObservationOrigin")
        _compact_memory(compact_memory)
        if not isinstance(observation, ObservationResponse):
            raise TypeError("observation must be a validated ObservationResponse")
        pairs = list(recent)[-3:]
        return {
            "call": "resolve_observation",
            "origin": {"kind": origin.kind, "text": origin.text},
            "fresh_observation": _fresh_observation(observation),
            "missing": _missing_objects(missing),
            "memory": compact_memory,
            "currently_visible": _visible_objects(currently_visible),
            "recent_say_pairs": [
                {"human_say": pair.human_say, "luxo_say": pair.lamp_say}
                for pair in pairs
            ],
        }


def _single_line(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be non-empty single-line text")
    return value


def _compact_memory(value: object) -> str:
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ValueError("compact_memory must be a single line")
    if not value:
        return value
    for entry in value.split("; "):
        if entry.count(":") != 1:
            raise ValueError("compact_memory must use the compact scene-memory format")
        object_id, fact = entry.split(":")
        if _MEMORY_ID.fullmatch(object_id) is None or not fact:
            raise ValueError("compact_memory must use stable object ids and canonical facts")
        if any(character in fact for character in ";{}[]="):
            raise ValueError("compact_memory cannot contain local record fields")
        if fact.count("(") != fact.count(")") or fact.count("(") > 1:
            raise ValueError("compact_memory attributes are malformed")
        if "(" in fact and not fact.endswith(")"):
            raise ValueError("compact_memory attributes are malformed")
    return value


def _observation_priors(values: Sequence["ObservationPrior"]) -> list[JsonValue]:
    from .memory import cloud_safe_attributes
    from .schema import ObservationPrior

    if isinstance(values, (str, bytes)):
        raise ValueError("prior must be a sequence of ObservationPrior values")
    result: list[JsonValue] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, ObservationPrior):
            raise TypeError("prior must contain ObservationPrior values")
        if value.id in seen:
            continue
        seen.add(value.id)
        item: dict[str, JsonValue] = {"id": value.id, "canonical": value.canonical}
        attributes = cloud_safe_attributes(value.attributes)
        if attributes:
            item["attributes"] = list(attributes)
        result.append(item)
    return result


def _missing_objects(values: Sequence["ObservationPrior"]) -> list[JsonValue]:
    from .schema import ObservationPrior

    if isinstance(values, (str, bytes)):
        raise ValueError("missing must be a sequence of ObservationPrior values")
    if not all(isinstance(value, ObservationPrior) for value in values):
        raise TypeError("missing must contain ObservationPrior values")
    return [{"id": value.id, "canonical": value.canonical} for value in values]


def _visible_objects(values: Sequence["CloudSceneObject"]) -> list[JsonValue]:
    from .memory import CloudSceneObject

    if isinstance(values, (str, bytes)):
        raise ValueError("currently_visible must be a sequence of CloudSceneObject values")
    result: list[JsonValue] = []
    for value in values:
        if not isinstance(value, CloudSceneObject):
            raise TypeError("currently_visible must contain CloudSceneObject values")
        result.append(
            {
                "id": value.id,
                "canonical": value.canonical,
                "attributes": list(value.attributes),
            }
        )
    return result


def _fresh_observation(observation: "ObservationResponse") -> dict[str, JsonValue]:
    from .memory import cloud_safe_attributes

    def facts(items: Sequence[object]) -> list[JsonValue]:
        return [
            {
                "id": item.match,
                "canonical": item.canonical,
                "attributes": list(cloud_safe_attributes(item.attributes)),
            }
            for item in items
        ]

    focus_id = (
        observation.visible[observation.focus].match
        if observation.focus is not None
        else None
    )
    return {
        "visible": facts(observation.visible),
        "focus": observation.focus,
        "focus_id": focus_id,
    }


__all__ = ["FixedPromptBuilder", "JsonValue", "PromptBuilder", "SYSTEM_PROMPT"]
