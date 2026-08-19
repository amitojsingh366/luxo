"""Public Piper synthesis boundary.

Piper runs through ONNX Runtime. The core returns PCM plus its precomputed
amplitude envelope; the browser remains the only audio-output owner.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    pcm_int16: bytes
    sample_hz: int
    envelope: tuple[float, ...]
    envelope_hz: int


class TextToSpeech(Protocol):
    @property
    def model_path(self) -> Path: ...

    def warm(self) -> None: ...

    def synthesize(self, text: str) -> SpeechAudio: ...

    def chunks(self, audio: SpeechAudio, max_bytes: int = 8_192) -> Iterator[bytes]: ...


__all__ = ["SpeechAudio", "TextToSpeech"]
