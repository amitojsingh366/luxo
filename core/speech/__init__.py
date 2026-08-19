"""Local speech inference interfaces; the browser owns capture and playback."""

from .stt import SpeechToText, Transcript
from .tts import SpeechAudio, TextToSpeech

__all__ = ["SpeechAudio", "SpeechToText", "TextToSpeech", "Transcript"]
