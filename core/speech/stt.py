"""Native whisper.cpp transcription boundary.

The browser owns capture and VAD. This module accepts one complete utterance
as 16 kHz mono signed-int16 PCM and invokes whisper.cpp synchronously. Callers
must run :meth:`transcribe` and :meth:`warm` on an inference worker, never on
the animation tick.
"""

from __future__ import annotations

import math
import os
import re
import stat
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


SAMPLE_HZ = 16_000
MIN_UTTERANCE_MS = 400
MAX_THREADS = 4
DEFAULT_THREADS = 2
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 300.0
_MODEL_NAME = re.compile(r"ggml-base\.en-q5_[01]\.bin", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str


class SpeechToText(Protocol):
    """Synchronous STT boundary intended for an inference worker thread."""

    @property
    def model_path(self) -> Path: ...

    def warm(self) -> None: ...

    def transcribe(self, pcm_int16: bytes, sample_hz: int = SAMPLE_HZ) -> Transcript: ...


class SpeechToTextError(RuntimeError):
    """Base class for failures safe to surface without exposing raw audio."""


class InvalidAudioError(SpeechToTextError, ValueError):
    """The supplied buffer is not a complete supported utterance."""


class UnsafeConfigurationError(SpeechToTextError, ValueError):
    """The backend paths or resource bounds are unsafe or incompatible."""


class BackendExecutionError(SpeechToTextError):
    """whisper.cpp could not be started or returned a failure status."""


class TranscriptionTimeoutError(BackendExecutionError, TimeoutError):
    """whisper.cpp exceeded its configured inference deadline."""


class MissingTranscriptError(BackendExecutionError):
    """whisper.cpp succeeded but did not produce a usable transcript."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int


class CommandRunner(Protocol):
    """Injectable synchronous process boundary for worker-thread use."""

    def run(self, command: Sequence[str], *, timeout_s: float) -> CommandResult: ...


class SubprocessRunner:
    """Run a native executable directly, without a command shell."""

    def run(self, command: Sequence[str], *, timeout_s: float) -> CommandResult:
        completed = subprocess.run(
            tuple(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
        return CommandResult(returncode=completed.returncode)


class WhisperCppSpeechToText:
    """whisper.cpp ``base.en`` Q5 CLI adapter.

    Construction validates the explicit native executable and model paths.
    The model filename is part of the deployment contract; model contents are
    deliberately not read here because ``setup.sh``/``doctor.py`` own hashes.
    """

    def __init__(
        self,
        *,
        binary_path: str | Path,
        model_path: str | Path,
        threads: int = DEFAULT_THREADS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner: CommandRunner | None = None,
    ) -> None:
        self._binary_path = _validate_binary_path(binary_path)
        self._model_path = _validate_model_path(model_path)
        self._threads = _validate_threads(threads)
        self._timeout_s = _validate_timeout(timeout_s)
        self._runner = runner if runner is not None else SubprocessRunner()
        if not callable(getattr(self._runner, "run", None)):
            raise UnsafeConfigurationError("runner must provide a callable run method")
        self._warm_lock = threading.Lock()
        self._warmed = False

    @property
    def binary_path(self) -> Path:
        return self._binary_path

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def warmed(self) -> bool:
        return self._warmed

    def warm(self) -> None:
        """Load/exercise the backend once using 400 ms of local silence."""

        with self._warm_lock:
            if self._warmed:
                return
            silence = bytes(SAMPLE_HZ * MIN_UTTERANCE_MS // 1_000 * 2)
            self._invoke(silence, require_transcript=False)
            self._warmed = True

    def transcribe(self, pcm_int16: bytes, sample_hz: int = SAMPLE_HZ) -> Transcript:
        pcm = _validate_pcm(pcm_int16, sample_hz)
        text = self._invoke(pcm, require_transcript=True)
        return Transcript(text=text)

    def _invoke(self, pcm: bytes, *, require_transcript: bool) -> str:
        with tempfile.TemporaryDirectory(prefix="luxo-stt-") as temp_dir:
            directory = Path(temp_dir)
            wav_path = directory / "utterance.wav"
            output_prefix = directory / "transcript"
            output_path = directory / "transcript.txt"
            _write_wav(wav_path, pcm)

            command = (
                str(self._binary_path),
                "--model",
                str(self._model_path),
                "--file",
                str(wav_path),
                "--language",
                "en",
                "--no-timestamps",
                "--threads",
                str(self._threads),
                "--output-txt",
                "--output-file",
                str(output_prefix),
            )
            try:
                result = self._runner.run(command, timeout_s=self._timeout_s)
            except (subprocess.TimeoutExpired, TimeoutError):
                raise TranscriptionTimeoutError(
                    "whisper.cpp exceeded the configured timeout"
                ) from None
            except OSError:
                raise BackendExecutionError("could not start whisper.cpp backend") from None

            if result.returncode != 0:
                raise BackendExecutionError(
                    f"whisper.cpp exited with status {result.returncode}"
                )
            if not require_transcript:
                return ""
            if not output_path.is_file():
                raise MissingTranscriptError("whisper.cpp produced no transcript file")
            try:
                text = output_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                raise MissingTranscriptError("whisper.cpp transcript is unreadable") from None
            normalized = " ".join(text.split())
            if not normalized:
                raise MissingTranscriptError("whisper.cpp produced an empty transcript")
            return normalized


def _validate_binary_path(value: str | Path) -> Path:
    path = _absolute_path(value, label="binary")
    try:
        mode = path.stat().st_mode
    except OSError:
        raise UnsafeConfigurationError("whisper.cpp binary is not accessible") from None
    if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
        raise UnsafeConfigurationError("whisper.cpp binary must be an executable file")
    return path


def _validate_model_path(value: str | Path) -> Path:
    path = _absolute_path(value, label="model")
    if _MODEL_NAME.fullmatch(path.name) is None:
        raise UnsafeConfigurationError(
            "model filename must identify a ggml base.en Q5 model"
        )
    try:
        metadata = path.stat()
    except OSError:
        raise UnsafeConfigurationError("whisper.cpp model is not accessible") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise UnsafeConfigurationError("whisper.cpp model must be a non-empty regular file")
    return path


def _absolute_path(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise UnsafeConfigurationError(f"{label} path must be explicit")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise UnsafeConfigurationError(f"{label} path must be absolute")
    try:
        return path.resolve(strict=True)
    except OSError:
        raise UnsafeConfigurationError(f"{label} path does not exist") from None


def _validate_threads(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_THREADS:
        raise UnsafeConfigurationError(f"threads must be between 1 and {MAX_THREADS}")
    return value


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsafeConfigurationError("timeout must be a finite number of seconds")
    timeout = float(value)
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT_S:
        raise UnsafeConfigurationError(
            f"timeout must be greater than zero and at most {MAX_TIMEOUT_S:g} seconds"
        )
    return timeout


def _validate_pcm(pcm: bytes, sample_hz: int) -> bytes:
    if sample_hz != SAMPLE_HZ:
        raise InvalidAudioError(f"sample rate must be exactly {SAMPLE_HZ} Hz")
    if not isinstance(pcm, bytes):
        raise InvalidAudioError("PCM utterance must be bytes")
    if not pcm:
        raise InvalidAudioError("PCM utterance must not be empty")
    if len(pcm) % 2:
        raise InvalidAudioError("PCM utterance must contain complete int16 samples")
    minimum_bytes = SAMPLE_HZ * MIN_UTTERANCE_MS // 1_000 * 2
    if len(pcm) < minimum_bytes:
        raise InvalidAudioError(
            f"PCM utterance must be at least {MIN_UTTERANCE_MS} ms"
        )
    return pcm


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_HZ)
        output.writeframes(pcm)


__all__ = [
    "BackendExecutionError",
    "CommandResult",
    "CommandRunner",
    "InvalidAudioError",
    "MissingTranscriptError",
    "SpeechToText",
    "SpeechToTextError",
    "SubprocessRunner",
    "Transcript",
    "TranscriptionTimeoutError",
    "UnsafeConfigurationError",
    "WhisperCppSpeechToText",
]
