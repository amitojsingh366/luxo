"""Piper text-to-speech through ONNX Runtime.

Synthesis is synchronous and must run on a worker thread. This module never
opens an audio device: it returns raw PCM payload bytes for the protocol layer,
which is responsible for adding the ``0x03`` frame prefix.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import struct
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SAMPLE_HZ = 22_050
ENVELOPE_HZ = 50
DEFAULT_LENGTH_SCALE = 0.88
DEFAULT_CHUNK_BYTES = 8_192
_SAMPLES_PER_ENVELOPE_FRAME = SAMPLE_HZ // ENVELOPE_HZ
_SPECIAL_PHONEMES = ("_", "^", "$")


class TtsError(RuntimeError):
    """Base error for the local voice boundary."""


class TtsAssetError(TtsError):
    """A model or metadata asset is missing or unreadable."""


class TtsMetadataError(TtsAssetError):
    """Piper metadata is malformed or incompatible with this audio contract."""


class TtsInputError(TtsError):
    """Text or phoneme IDs cannot be sent to Piper."""


class TtsSessionError(TtsError):
    """ONNX Runtime setup or cleanup failed."""


class TtsInferenceError(TtsError):
    """Piper inference failed or returned malformed audio."""


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


@runtime_checkable
class InferenceSession(Protocol):
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, object],
    ) -> Sequence[object]: ...


PhonemeIdEncoder = Callable[[str], Sequence[int]]
SessionFactory = Callable[[Path], InferenceSession]


@dataclass(frozen=True, slots=True)
class PiperMetadata:
    sample_rate: int
    phoneme_id_map: Mapping[str, tuple[int, ...]]
    num_symbols: int
    noise_scale: float
    noise_w: float


def load_piper_metadata(path: Path) -> PiperMetadata:
    """Load only the Piper JSON fields needed by this synthesis boundary."""

    _require_file(path, "Piper config")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise TtsAssetError(f"cannot read Piper config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TtsMetadataError(f"Piper config is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise TtsMetadataError("Piper config root must be an object")

    audio = _required_object(raw, "audio")
    sample_rate = _required_int(audio, "sample_rate")
    if sample_rate != SAMPLE_HZ:
        raise TtsMetadataError(
            f"Piper audio.sample_rate must be {SAMPLE_HZ}, got {sample_rate}"
        )

    num_symbols = _required_int(raw, "num_symbols")
    if num_symbols <= 0:
        raise TtsMetadataError("Piper num_symbols must be positive")
    raw_map = raw.get("phoneme_id_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise TtsMetadataError("Piper phoneme_id_map must be a non-empty object")
    phoneme_id_map: dict[str, tuple[int, ...]] = {}
    for symbol, raw_ids in raw_map.items():
        if not isinstance(symbol, str) or not symbol:
            raise TtsMetadataError("Piper phoneme_id_map keys must be non-empty strings")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise TtsMetadataError(f"Piper phoneme IDs for {symbol!r} must be a list")
        ids: list[int] = []
        for value in raw_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TtsMetadataError(f"Piper phoneme ID for {symbol!r} must be an integer")
            if value < 0 or value >= num_symbols:
                raise TtsMetadataError(
                    f"Piper phoneme ID {value} for {symbol!r} is outside num_symbols"
                )
            ids.append(value)
        phoneme_id_map[symbol] = tuple(ids)
    missing = [symbol for symbol in _SPECIAL_PHONEMES if symbol not in phoneme_id_map]
    if missing:
        raise TtsMetadataError(f"Piper phoneme_id_map is missing {', '.join(missing)}")

    inference = _required_object(raw, "inference")
    noise_scale = _required_finite(inference, "noise_scale", positive=True)
    noise_w = _required_finite(inference, "noise_w", positive=True)
    return PiperMetadata(
        sample_rate=sample_rate,
        phoneme_id_map=phoneme_id_map,
        num_symbols=num_symbols,
        noise_scale=noise_scale,
        noise_w=noise_w,
    )


def create_onnx_session(model_path: Path) -> InferenceSession:
    """Create a CPU-only session capped for the 4-core Ubuntu target."""

    _require_file(model_path, "Piper model")
    try:
        ort = importlib.import_module("onnxruntime")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        add_entry = getattr(options, "add_session_config_entry", None)
        if callable(add_entry):
            add_entry("session.intra_op.allow_spinning", "0")
            add_entry("session.inter_op.allow_spinning", "0")
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except TtsError:
        raise
    except Exception as exc:
        raise TtsSessionError(f"cannot load Piper model with ONNX Runtime: {model_path}") from exc


class PiperTextToSpeech:
    """Synchronous Piper synthesizer intended only for a worker thread.

    ``phoneme_ids`` is deliberately injected. It owns text normalization and
    phonemization and must return the complete Piper ID sequence, including
    model-appropriate boundary and padding IDs.
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        phoneme_ids: PhonemeIdEncoder,
        *,
        length_scale: float = DEFAULT_LENGTH_SCALE,
        session: InferenceSession | None = None,
        session_factory: SessionFactory = create_onnx_session,
        speaker_id: int | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._config_path = Path(config_path)
        _require_file(self._model_path, "Piper model")
        self.metadata = load_piper_metadata(self._config_path)
        if not callable(phoneme_ids):
            raise TtsInputError("phoneme_ids must be callable")
        self._phoneme_ids = phoneme_ids
        try:
            self.length_scale = _finite_number(length_scale, "length_scale", positive=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TtsInputError("length_scale must be finite and positive") from exc
        if speaker_id is not None and (
            isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0
        ):
            raise TtsInputError("speaker_id must be a non-negative integer")
        self._speaker_id = speaker_id
        try:
            self._session: InferenceSession | None = (
                session if session is not None else session_factory(self._model_path)
            )
        except TtsError:
            raise
        except Exception as exc:
            raise TtsSessionError("Piper session factory failed") from exc
        if self._session is None or not callable(getattr(self._session, "run", None)):
            raise TtsSessionError("Piper session must provide run()")
        self._state_lock = threading.RLock()
        self._warm_lock = threading.Lock()
        self._warmed = False
        self._closed = False

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def warmed(self) -> bool:
        with self._state_lock:
            return self._warmed

    def warm(self) -> None:
        """Exercise Piper once with a tiny utterance; subsequent calls do nothing."""

        with self._warm_lock:
            with self._state_lock:
                if self._closed:
                    raise TtsSessionError("Piper session is closed")
                if self._warmed:
                    return
            ids = self._encode(".")
            self._infer(ids)
            with self._state_lock:
                self._warmed = True

    def synthesize(self, text: str) -> SpeechAudio:
        """Synthesize one utterance; call from a worker, never an animation tick."""

        if not isinstance(text, str) or not text.strip():
            raise TtsInputError("text must be a non-empty string")
        waveform = self._infer(self._encode(text.strip()))
        pcm = float_waveform_to_pcm(waveform)
        return SpeechAudio(
            pcm_int16=pcm,
            sample_hz=SAMPLE_HZ,
            envelope=amplitude_envelope(pcm),
            envelope_hz=ENVELOPE_HZ,
        )

    def chunks(
        self,
        audio: SpeechAudio,
        max_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """Yield prefix-free PCM chunks; empty audio yields no chunks."""

        return chunk_pcm(audio.pcm_int16, max_bytes=max_bytes)

    def close(self) -> None:
        """Release the session reference and close injected sessions that support it."""

        with self._state_lock:
            if self._closed:
                return
            session = self._session
            self._session = None
            self._closed = True
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                raise TtsSessionError("Piper session cleanup failed") from exc

    def __enter__(self) -> PiperTextToSpeech:
        with self._state_lock:
            if self._closed:
                raise TtsSessionError("Piper session is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _encode(self, text: str) -> tuple[int, ...]:
        try:
            raw_ids = self._phoneme_ids(text)
        except Exception as exc:
            raise TtsInputError("phoneme ID generation failed") from exc
        if isinstance(raw_ids, (str, bytes)):
            raise TtsInputError("phoneme ID generator returned an invalid sequence")
        try:
            ids = tuple(raw_ids)
        except TypeError as exc:
            raise TtsInputError("phoneme ID generator must return a sequence") from exc
        if not ids:
            raise TtsInputError("phoneme ID generator returned no IDs")
        valid_ids = {value for values in self.metadata.phoneme_id_map.values() for value in values}
        for value in ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TtsInputError("phoneme IDs must be integers")
            if value not in valid_ids:
                raise TtsInputError(f"phoneme ID {value} is not present in Piper metadata")
        return ids

    def _infer(self, ids: tuple[int, ...]) -> tuple[float, ...]:
        session = self._session_or_raise()
        try:
            feed = _piper_feed(
                ids,
                noise_scale=self.metadata.noise_scale,
                length_scale=self.length_scale,
                noise_w=self.metadata.noise_w,
                speaker_id=self._speaker_id,
            )
            outputs = session.run(None, feed)
        except TtsError:
            raise
        except Exception as exc:
            raise TtsInferenceError("Piper inference failed") from exc
        if not isinstance(outputs, Sequence) or not outputs:
            raise TtsInferenceError("Piper returned no audio output")
        return _mono_waveform(outputs[0])

    def _session_or_raise(self) -> InferenceSession:
        with self._state_lock:
            if self._closed or self._session is None:
                raise TtsSessionError("Piper session is closed")
            return self._session


def float_waveform_to_pcm(waveform: Sequence[float]) -> bytes:
    """Clamp finite float samples and encode signed-int16 little-endian PCM."""

    pcm = bytearray(len(waveform) * 2)
    for index, value in enumerate(waveform):
        if isinstance(value, bool):
            raise TtsInferenceError("Piper waveform contains a non-numeric sample")
        try:
            sample = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TtsInferenceError("Piper waveform contains a non-numeric sample") from exc
        if not math.isfinite(sample):
            raise TtsInferenceError("Piper waveform contains a non-finite sample")
        sample = min(1.0, max(-1.0, sample))
        scaled = sample * (32_768 if sample < 0 else 32_767)
        quantized = min(32_767, max(-32_768, int(round(scaled))))
        struct.pack_into("<h", pcm, index * 2, quantized)
    return bytes(pcm)


def amplitude_envelope(pcm_int16: bytes) -> tuple[float, ...]:
    """Return normalized RMS frames at exactly 50 Hz, including a final partial."""

    if len(pcm_int16) % 2:
        raise TtsInputError("PCM byte length must align to int16 samples")
    if not pcm_int16:
        return ()
    samples = tuple(value[0] for value in struct.iter_unpack("<h", pcm_int16))
    levels: list[float] = []
    for start in range(0, len(samples), _SAMPLES_PER_ENVELOPE_FRAME):
        frame = samples[start : start + _SAMPLES_PER_ENVELOPE_FRAME]
        mean_square = sum((sample / 32_768.0) ** 2 for sample in frame) / len(frame)
        levels.append(math.sqrt(mean_square))
    peak = max(levels, default=0.0)
    if peak <= 0.0:
        return tuple(0.0 for _ in levels)
    return tuple(min(1.0, max(0.0, level / peak)) for level in levels)


def chunk_pcm(pcm_int16: bytes, *, max_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[bytes]:
    """Split prefix-free PCM without ever cutting an int16 sample."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 2:
        raise TtsInputError("max_bytes must be an integer of at least 2")
    if len(pcm_int16) % 2:
        raise TtsInputError("PCM byte length must align to int16 samples")
    chunk_size = max_bytes - (max_bytes % 2)
    for offset in range(0, len(pcm_int16), chunk_size):
        yield bytes(pcm_int16[offset : offset + chunk_size])


def _piper_feed(
    ids: tuple[int, ...],
    *,
    noise_scale: float,
    length_scale: float,
    noise_w: float,
    speaker_id: int | None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "input": [list(ids)],
        "input_lengths": [len(ids)],
        "scales": [noise_scale, length_scale, noise_w],
    }
    if speaker_id is not None:
        values["sid"] = [speaker_id]
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        # Offline injected sessions can consume the plain values above. A real
        # ONNX Runtime installation always brings NumPy as a dependency.
        return values
    return {
        name: numpy.asarray(value, dtype=numpy.float32 if name == "scales" else numpy.int64)
        for name, value in values.items()
    }


def _mono_waveform(value: object) -> tuple[float, ...]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, (str, bytes, bytearray, dict)) or not isinstance(value, Sequence):
        raise TtsInferenceError("Piper audio output must be an array")
    current: Sequence[Any] = value
    while current and any(isinstance(item, Sequence) for item in current):
        if len(current) != 1 or isinstance(current[0], (str, bytes, bytearray, dict)):
            raise TtsInferenceError("Piper audio output must be mono")
        nested = current[0]
        to_list = getattr(nested, "tolist", None)
        if callable(to_list):
            nested = to_list()
        if not isinstance(nested, Sequence):
            raise TtsInferenceError("Piper audio output has a malformed shape")
        current = nested
    if not current:
        raise TtsInferenceError("Piper returned an empty waveform")
    samples: list[float] = []
    for value in current:
        if isinstance(value, bool):
            raise TtsInferenceError("Piper waveform contains a non-numeric sample")
        try:
            sample = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TtsInferenceError("Piper waveform contains a non-numeric sample") from exc
        if not math.isfinite(sample):
            raise TtsInferenceError("Piper waveform contains a non-finite sample")
        samples.append(sample)
    return tuple(samples)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise TtsAssetError(f"{label} file is missing: {path}")


def _required_object(parent: Mapping[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TtsMetadataError(f"Piper {key} must be an object")
    return value


def _required_int(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TtsMetadataError(f"Piper {key} must be an integer")
    return value


def _required_finite(
    parent: Mapping[str, object],
    key: str,
    *,
    positive: bool,
) -> float:
    try:
        return _finite_number(parent[key], key, positive=positive)
    except KeyError as exc:
        raise TtsMetadataError(f"Piper inference.{key} is required") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise TtsMetadataError(f"Piper inference.{key} must be finite and positive") from exc


def _finite_number(value: object, label: str, *, positive: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite and positive")
    return number


__all__ = [
    "DEFAULT_CHUNK_BYTES", "DEFAULT_LENGTH_SCALE", "ENVELOPE_HZ",
    "PiperMetadata", "PiperTextToSpeech", "SAMPLE_HZ", "SpeechAudio", "TextToSpeech",
    "TtsAssetError", "TtsError", "TtsInferenceError", "TtsInputError",
    "TtsMetadataError", "TtsSessionError", "amplitude_envelope", "chunk_pcm",
    "create_onnx_session", "float_waveform_to_pcm", "load_piper_metadata",
]
