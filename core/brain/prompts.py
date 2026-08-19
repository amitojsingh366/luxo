"""Fixed model instructions and privacy-limited call payloads."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from .schema import normalize_canonical_label

if TYPE_CHECKING:
    from .client import RecentExchange


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

SYSTEM_PROMPT = """You are Luxo, an eager, puppyish character lamp: bouncy, delighted, slightly overenthusiastic, and always concise.
Return bare JSON only, with no markdown fence, preamble, or trailing text. Never echo or restate any input transcript, memory, labels, image, or goal. Never emit keys named reasoning, thoughts, notes, explanation, or confidence.
For a payload containing transcript, return {"say":string,"plan":array}. Keep say at or below 120 characters. For a payload containing missing, use only those Python-computed labels and return the same say/plan shape. For an image request, return object facts only as {"present":[canonical strings],"new":[{"label":string,"canonical":string,"attributes":[strings],"bbox_norm":[x,y,width,height]}]}; never return dialogue.
Plans use only these closed semantic actions and fields: gesture{name} where name is perk_up|nod|double_take|recoil|lean_in|bounce|shake_no|settle|droop|regard; look_at{target} where target is person|scene|obj:<id>; light{preset,pattern?} where preset is warm_idle|warm_bright|cool_dim|curious_focus|thinking_pulse|excited_flash|sad_fade and pattern is steady|pulse|flicker|blink; sfx{name} where name is chirp_up|chirp_found|boing|whirr_short|hmm|blip_sad|fanfare_small|click; scan{arc?,speed?}; observe{}; posture{name} where name is rest|alert|slump|stoop|crane; wait{ms}. Never emit joint angles, motion timing, easing, or any action or enum outside this vocabulary."""

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


class FixedPromptBuilder:
    """Build the three model payloads without accepting unrelated state."""

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
