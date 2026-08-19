"""Public fixed-prompt and minimal-payload construction boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .client import RecentExchange


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


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


__all__ = ["JsonValue", "PromptBuilder"]
