"""Public whisper.cpp transcription boundary.

The browser supplies one complete 16 kHz mono int16 utterance after VAD. This
module never owns a microphone or an audio device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str


class SpeechToText(Protocol):
    @property
    def model_path(self) -> Path: ...

    def warm(self) -> None: ...

    def transcribe(self, pcm_int16: bytes, sample_hz: int = 16_000) -> Transcript: ...


__all__ = ["SpeechToText", "Transcript"]
