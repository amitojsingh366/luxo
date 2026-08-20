"""Fixed model instructions and privacy-limited call payloads."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from .schema import normalize_canonical_label

if TYPE_CHECKING:
    from .client import RecentExchange
    from .schema import ObservationResponse


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

SYSTEM_PROMPT = """You are Luxo, an eager, puppyish character lamp: bouncy, delighted, slightly overenthusiastic, and always concise.
Return bare JSON only, with no markdown fence, preamble, or trailing text. Never echo or restate any input transcript, memory, labels, image, or goal. Never emit keys named reasoning, thoughts, notes, explanation, or confidence.
For a payload containing transcript, answer only that current transcript and return {"say":string,"plan":array}. recent_say_pairs are background context, never new requests: do not repeat, continue, combine, or answer an earlier human_say again. Never volunteer what is visible or missing unless the current transcript asks about the scene. The say value must be an answer, never a copy of transcript. Current visual evidence overrides memory: when the transcript asks a deictic or present visual-detail question such as what color is this, what is in my hand, or what do you see here, never answer from memory even when it has a matching object. Return a brief looking or checking acknowledgement and a plan containing exactly one {"op":"observe"}; do not scan merely to inspect a held or current object. For an explicitly historical or non-current question, answer from memory when it contains the matching object and requested attribute, and do not observe; when that requested fact is absent, say you do not know. If asked what is missing, gone, changed, or still visible, respond with a brief checking-now line and include scan then observe; Python will compute the answer after the image. Keep say at or below 120 characters. If the current transcript otherwise asks you to look at, identify, inspect, learn, or remember a shown object or scene, do not claim you saw or remembered it; respond with a brief looking-now line and include an observe op in the plan, preceded by scan when the scene rather than one held object is the target. For a payload containing missing, use only those Python-computed labels and return the same say/plan shape. If missing is empty, say that nothing seems missing and use an empty plan. Otherwise react with brief concern, name only those missing labels, ask where they went, and include scan then observe to look around once.
For an image request, ignore the character/dialogue task. present may contain only labels from prior_canonical that remain visible. known must contain fresh detailed object records for every label in present. new must contain every salient visible object not in prior_canonical. Each known and new object must use {"label":string,"canonical":string,"attributes":[strings],"bbox_norm":[x,y,width,height]}. Report visible colors, materials, markings, object type, and the attribute "held in hand" when applicable, especially for an object held in or near a hand. bbox_norm must use decimal values from 0 through 1; width and height are sizes, not right and bottom coordinates. If prior_canonical is empty, present and known must be [] and all discovered objects must go in new. Always return exactly present, known, and new as top-level arrays; never return say, plan, dialogue, or prose, even when the image contains a person.
For a payload containing visual_intent and fresh_objects, return {"say":string,"plan":array} using only fresh_objects and memory as evidence. If visual_intent is spontaneous_hand, ask one short, specific question naming the observed hand object and one of its purposes, features, or uses. Otherwise answer the visual question directly and precisely from fresh_objects; admit uncertainty rather than inventing a detail. Never ask generic questions such as what are you working on, what are you doing, what are you holding, or what is in your hand. Keep the plan small and do not observe again.
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
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]: ...

    def observe_payload(
        self,
        jpeg: bytes,
        prior_canonical: Sequence[str],
    ) -> Mapping[str, JsonValue]: ...

    def narrate_payload(self, missing: Sequence[str]) -> Mapping[str, JsonValue]: ...

    def scene_comment_payload(
        self,
        visual_intent: str,
        observation: "ObservationResponse",
        compact_memory: str,
    ) -> Mapping[str, JsonValue]: ...


class FixedPromptBuilder:
    """Build the four model payloads without accepting unrelated state."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def converse_payload(
        self,
        transcript: str,
        compact_memory: str,
        recent: Sequence[RecentExchange],
    ) -> Mapping[str, JsonValue]:
        _single_line(transcript, "transcript")
        _compact_memory(compact_memory)
        pairs = list(recent)[-3:]
        return {
            "transcript": transcript,
            "memory": compact_memory,
            "recent_say_pairs": [
                {"human_say": pair.human_say, "luxo_say": pair.lamp_say}
                for pair in pairs
            ],
        }

    def observe_payload(
        self,
        jpeg: bytes,
        prior_canonical: Sequence[str],
    ) -> Mapping[str, JsonValue]:
        if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("jpeg must contain a complete JPEG image")
        encoded = base64.b64encode(jpeg).decode("ascii")
        return {
            "prior_canonical": _canonical_labels(prior_canonical),
            "jpeg_data_url": f"data:image/jpeg;base64,{encoded}",
        }

    def narrate_payload(self, missing: Sequence[str]) -> Mapping[str, JsonValue]:
        return {"missing": _canonical_labels(missing)}

    def scene_comment_payload(
        self,
        visual_intent: str,
        observation: "ObservationResponse",
        compact_memory: str,
    ) -> Mapping[str, JsonValue]:
        from .schema import ObservationResponse

        _single_line(visual_intent, "visual_intent")
        _compact_memory(compact_memory)
        if not isinstance(observation, ObservationResponse):
            raise TypeError("observation must be a validated ObservationResponse")
        objects = (*observation.known, *observation.new)
        return {
            "visual_intent": visual_intent,
            "fresh_objects": [
                {
                    "label": item.label,
                    "canonical": item.canonical,
                    "attributes": list(item.attributes),
                }
                for item in objects
            ],
            "memory": compact_memory,
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


def _canonical_labels(values: Sequence[str]) -> list[JsonValue]:
    if isinstance(values, (str, bytes)):
        raise ValueError("canonical labels must be a sequence of strings")
    return list(dict.fromkeys(normalize_canonical_label(value) for value in values))


__all__ = ["FixedPromptBuilder", "JsonValue", "PromptBuilder", "SYSTEM_PROMPT"]
