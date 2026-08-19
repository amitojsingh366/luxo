"""Text to Piper phoneme IDs via eSpeak NG.

``core/speech/tts.py`` owns phoneme ID to waveform. It deliberately injects
``PhonemeIdEncoder`` so that text normalization and phonemization live here.
This module closes that gap: it normalizes text, phonemizes it to IPA with
eSpeak NG, and maps the result through the voice's ``phoneme_id_map``,
including the boundary and padding IDs Piper's own encoder emits.

eSpeak NG is an owner-approved system dependency (``apt install espeak-ng``
on Ubuntu 24.04, ``brew install espeak-ng`` on macOS). It is *not* a Python
dependency: it is invoked as a native executable behind :class:`EspeakRunner`,
which is injectable so the whole module is testable without it installed.
:func:`espeak_ng_status` reports availability for a preflight check.

Error types come from ``tts.py`` rather than a parallel hierarchy. The single
addition is :class:`PhonemizerError`, rooted at ``TtsError``, for "eSpeak NG
ran but could not be used".
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.speech.tts import (
    TtsAssetError,
    TtsError,
    TtsInputError,
    TtsMetadataError,
    load_piper_metadata,
)

LOGGER = logging.getLogger(__name__)

#: Piper's mandatory special symbols, mirroring ``tts._SPECIAL_PHONEMES``.
PAD_PHONEME = "_"
BOS_PHONEME = "^"
EOS_PHONEME = "$"
SPACE_PHONEME = " "

ESPEAK_BINARY_NAME = "espeak-ng"
ESPEAK_PHONEME_TYPE = "espeak"
DEFAULT_ESPEAK_VOICE = "en-us"
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 60.0

#: Marks that end an eSpeak clause. Piper re-inserts the punctuation phoneme
#: after each clause; the CLI cannot report the terminator, so the split is
#: performed here instead and the same phonemes are appended.
_CLAUSE_MARKS = ".!?,:;"
_CLAUSE_SPLIT = re.compile(f"([{re.escape(_CLAUSE_MARKS)}])")
#: Marks that Piper follows with a word space rather than a sentence break.
_SOFT_CLAUSE_MARKS = frozenset(",:;")
#: eSpeak emits ``(en)``-style markers when it switches language mid-utterance.
#: Every character in those markers is itself a valid Piper symbol, so they are
#: removed before symbol splitting rather than being phonemized as text.
_LANGUAGE_SWITCH = re.compile(r"\([^()]*\)")


class PhonemizerError(TtsError):
    """eSpeak NG could not be executed or returned unusable output."""


@runtime_checkable
class EspeakRunner(Protocol):
    """Injectable synchronous process boundary returning captured stdout."""

    def run(self, command: Sequence[str], *, timeout_s: float) -> str: ...


@runtime_checkable
class Phonemizer(Protocol):
    """Text to IPA. The seam checks replace so eSpeak NG is never invoked."""

    def phonemize(self, text: str, *, voice: str) -> str: ...


class SubprocessEspeakRunner:
    """Run eSpeak NG directly, without a command shell."""

    def run(self, command: Sequence[str], *, timeout_s: float) -> str:
        try:
            completed = subprocess.run(
                tuple(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PhonemizerError("eSpeak NG exceeded its phonemization timeout") from exc
        except OSError as exc:
            raise PhonemizerError("could not start eSpeak NG") from exc
        if completed.returncode != 0:
            raise PhonemizerError(f"eSpeak NG exited with status {completed.returncode}")
        try:
            return completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhonemizerError("eSpeak NG returned undecodable output") from exc


class EspeakNgPhonemizer:
    """Phonemize with the eSpeak NG executable.

    Construction resolves and validates the binary so a missing system
    dependency fails loudly at startup rather than on the first utterance.
    """

    def __init__(
        self,
        *,
        binary_path: str | Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner: EspeakRunner | None = None,
        locate: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._binary_path = _resolve_espeak_binary(binary_path, locate=locate)
        self._timeout_s = _validate_timeout(timeout_s)
        self._runner: EspeakRunner = runner if runner is not None else SubprocessEspeakRunner()
        if not callable(getattr(self._runner, "run", None)):
            raise PhonemizerError("runner must provide a callable run method")

    @property
    def binary_path(self) -> Path:
        return self._binary_path

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def command(self, text: str, *, voice: str) -> tuple[str, ...]:
        """Build the argv used for ``text``; exposed so checks can assert it."""

        return (
            str(self._binary_path),
            "-q",
            "--ipa",
            "-b",
            "1",
            "-v",
            voice,
            "--",
            text,
        )

    def phonemize(self, text: str, *, voice: str) -> str:
        if not isinstance(text, str) or not text:
            raise TtsInputError("phonemizer text must be a non-empty string")
        if not isinstance(voice, str) or not voice.strip():
            raise TtsMetadataError("eSpeak voice must be a non-empty string")
        try:
            output = self._runner.run(self.command(text, voice=voice), timeout_s=self._timeout_s)
        except TtsError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise PhonemizerError("eSpeak NG exceeded its phonemization timeout") from exc
        except Exception as exc:
            raise PhonemizerError("eSpeak NG invocation failed") from exc
        if not isinstance(output, str):
            raise PhonemizerError("eSpeak NG must return text")
        return output


@dataclass(frozen=True, slots=True)
class EspeakNgStatus:
    """Preflight result for eSpeak NG. Never raises; ``doctor.py`` can call it."""

    available: bool
    binary_path: Path | None
    version: str | None
    detail: str


def espeak_ng_status(
    *,
    binary_path: str | Path | None = None,
    locate: Callable[[str], str | None] = shutil.which,
    runner: EspeakRunner | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> EspeakNgStatus:
    """Report whether eSpeak NG is installed and usable, without raising."""

    try:
        resolved = _resolve_espeak_binary(binary_path, locate=locate)
    except TtsError as exc:
        return EspeakNgStatus(
            available=False,
            binary_path=None,
            version=None,
            detail=str(exc),
        )
    probe: EspeakRunner = runner if runner is not None else SubprocessEspeakRunner()
    try:
        raw = probe.run((str(resolved), "--version"), timeout_s=_validate_timeout(timeout_s))
    except Exception as exc:  # a preflight check must not propagate failures
        return EspeakNgStatus(
            available=False,
            binary_path=resolved,
            version=None,
            detail=f"eSpeak NG could not be run: {exc}",
        )
    version = " ".join(str(raw).split()) or None
    return EspeakNgStatus(
        available=True,
        binary_path=resolved,
        version=version,
        detail="eSpeak NG is available",
    )


@dataclass(frozen=True, slots=True)
class VoicePhonemeConfig:
    """The phonemization half of a Piper voice config."""

    voice: str
    phoneme_type: str
    phoneme_id_map: Mapping[str, tuple[int, ...]]
    phoneme_map: Mapping[str, tuple[str, ...]]
    sample_rate: int


def load_voice_phoneme_config(config_path: str | Path) -> VoicePhonemeConfig:
    """Load the eSpeak voice, phoneme map, and ID map from a Piper config.

    ``tts.load_piper_metadata`` performs the ID-map validation, including the
    22.05 kHz audio contract and the mandatory ``_``/``^``/``$`` symbols, so
    only the phonemization fields are re-read here.
    """

    path = Path(config_path).expanduser()
    metadata = load_piper_metadata(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:  # pragma: no cover - re-read of a read file
        raise TtsAssetError(f"cannot read Piper config: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - already parsed once
        raise TtsMetadataError(f"Piper config is not valid JSON: {path}") from exc

    phoneme_type = raw.get("phoneme_type", ESPEAK_PHONEME_TYPE)
    if phoneme_type != ESPEAK_PHONEME_TYPE:
        raise TtsMetadataError(
            f"Piper phoneme_type must be {ESPEAK_PHONEME_TYPE!r}, got {phoneme_type!r}"
        )

    espeak = raw.get("espeak")
    if not isinstance(espeak, dict):
        raise TtsMetadataError("Piper espeak section must be an object")
    voice = espeak.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        raise TtsMetadataError("Piper espeak.voice must be a non-empty string")

    return VoicePhonemeConfig(
        voice=voice.strip(),
        phoneme_type=phoneme_type,
        phoneme_id_map=metadata.phoneme_id_map,
        phoneme_map=_load_phoneme_map(raw.get("phoneme_map")),
        sample_rate=metadata.sample_rate,
    )


class EspeakPhonemeEncoder:
    """Text to Piper phoneme IDs; satisfies ``tts.PhonemeIdEncoder``.

    ID layout matches Piper's own encoder: ``BOS``, one ``PAD``, then every
    phoneme's IDs each followed by a ``PAD``, then ``EOS``.

    Unknown phonemes are **dropped and logged**, never substituted. This
    matches upstream Piper, which collects unmapped symbols as "missing
    phonemes" and omits them, and it is the only deterministic choice that
    cannot inject a sound the speaker did not say. Dropped symbols are counted
    on the instance so a caller can surface them.
    """

    def __init__(
        self,
        voice_config: VoicePhonemeConfig,
        *,
        phonemizer: Phonemizer,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(voice_config, VoicePhonemeConfig):
            raise TtsMetadataError("voice_config must be a VoicePhonemeConfig")
        if not callable(getattr(phonemizer, "phonemize", None)):
            raise PhonemizerError("phonemizer must provide a callable phonemize method")
        self._config = voice_config
        self._phonemizer = phonemizer
        self._logger = logger if logger is not None else LOGGER
        self._ids = dict(voice_config.phoneme_id_map)
        self._remap = dict(voice_config.phoneme_map)
        for symbol in (PAD_PHONEME, BOS_PHONEME, EOS_PHONEME):
            if symbol not in self._ids:
                raise TtsMetadataError(f"Piper phoneme_id_map is missing {symbol!r}")
        # Longest match first, so multi-codepoint symbols (ties, affricates,
        # base plus combining mark) win over their leading codepoint.
        self._symbols = tuple(
            sorted(set(self._ids) | set(self._remap), key=lambda item: (-len(item), item))
        )
        self._max_symbol_length = max(len(symbol) for symbol in self._symbols)
        self._counter_lock = threading.Lock()
        self._dropped_count = 0
        self._dropped_symbols: dict[str, int] = {}

    @property
    def voice(self) -> str:
        return self._config.voice

    @property
    def voice_config(self) -> VoicePhonemeConfig:
        return self._config

    @property
    def dropped_phoneme_count(self) -> int:
        """Total unmapped symbols dropped since construction."""

        with self._counter_lock:
            return self._dropped_count

    @property
    def dropped_phonemes(self) -> Mapping[str, int]:
        """Unmapped symbols dropped so far, with per-symbol counts."""

        with self._counter_lock:
            return dict(self._dropped_symbols)

    def __call__(self, text: str) -> tuple[int, ...]:
        return self.phoneme_ids(self.phonemes(text))

    def phonemes(self, text: str) -> tuple[str, ...]:
        """Normalize, phonemize, and split ``text`` into Piper symbols."""

        normalized = normalize_text(text)
        collected: list[str] = []
        clauses = split_clauses(normalized)
        for index, (body, mark) in enumerate(clauses):
            if body:
                ipa = clean_espeak_output(self._phonemizer.phonemize(body, voice=self._config.voice))
                collected.extend(self._split_symbols(ipa))
            if mark is not None and mark in self._ids:
                collected.append(mark)
                trailing = mark in _SOFT_CLAUSE_MARKS or index + 1 < len(clauses)
                if trailing and SPACE_PHONEME in self._ids:
                    collected.append(SPACE_PHONEME)
        phonemes = self._apply_phoneme_map(collected)
        while phonemes and phonemes[-1] == SPACE_PHONEME:
            phonemes.pop()
        if not phonemes:
            raise TtsInputError("text produced no phonemes eSpeak NG could map")
        return tuple(phonemes)

    def phoneme_ids(self, phonemes: Iterable[str]) -> tuple[int, ...]:
        """Wrap mapped phonemes in Piper's BOS, interspersed PAD, and EOS."""

        ids: list[int] = [*self._ids[BOS_PHONEME], *self._ids[PAD_PHONEME]]
        mapped = 0
        for phoneme in phonemes:
            values = self._ids.get(phoneme)
            if values is None:
                self._record_drop(phoneme)
                continue
            ids.extend(values)
            ids.extend(self._ids[PAD_PHONEME])
            mapped += 1
        ids.extend(self._ids[EOS_PHONEME])
        if mapped == 0:
            raise TtsInputError("no phoneme mapped to a Piper ID")
        return tuple(ids)

    def _split_symbols(self, ipa: str) -> list[str]:
        """Greedy longest-match split, so multi-codepoint symbols survive."""

        symbols: list[str] = []
        index = 0
        length = len(ipa)
        while index < length:
            match = None
            limit = min(self._max_symbol_length, length - index)
            for size in range(limit, 0, -1):
                candidate = ipa[index : index + size]
                if candidate in self._ids or candidate in self._remap:
                    match = candidate
                    break
            if match is None:
                self._record_drop(ipa[index])
                index += 1
                continue
            symbols.append(match)
            index += len(match)
        return symbols

    def _apply_phoneme_map(self, phonemes: Sequence[str]) -> list[str]:
        if not self._remap:
            return list(phonemes)
        expanded: list[str] = []
        for phoneme in phonemes:
            replacement = self._remap.get(phoneme)
            if replacement is None:
                expanded.append(phoneme)
            else:
                expanded.extend(replacement)
        return expanded

    def _record_drop(self, symbol: str) -> None:
        with self._counter_lock:
            self._dropped_count += 1
            self._dropped_symbols[symbol] = self._dropped_symbols.get(symbol, 0) + 1
        self._logger.warning(
            "dropping unmapped phoneme %r (U+%04X) for voice %s",
            symbol,
            ord(symbol[0]) if symbol else 0,
            self._config.voice,
        )


def create_phoneme_encoder(
    config_path: str | Path,
    *,
    phonemizer: Phonemizer | None = None,
    logger: logging.Logger | None = None,
) -> EspeakPhonemeEncoder:
    """Build the encoder ``PiperTextToSpeech`` expects for a voice config.

    ``phonemizer`` defaults to the real eSpeak NG executable; injecting a
    substitute keeps callers, and every check, free of the system dependency.
    """

    voice_config = load_voice_phoneme_config(config_path)
    resolved = phonemizer if phonemizer is not None else EspeakNgPhonemizer()
    return EspeakPhonemeEncoder(voice_config, phonemizer=resolved, logger=logger)


def normalize_text(text: str) -> str:
    """Collapse whitespace and control characters ahead of eSpeak NG.

    NFC is applied to the *input* only. eSpeak NG output is deliberately left
    unnormalized because Piper ID maps mix precomposed symbols (``ç``) with
    base-plus-combining sequences, and either normal form would break one set.
    """

    if not isinstance(text, str):
        raise TtsInputError("text must be a string")
    normalized = unicodedata.normalize("NFC", text)
    cleaned = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    collapsed = " ".join(cleaned.split())
    if not collapsed:
        raise TtsInputError("text must contain at least one printable character")
    return collapsed


def split_clauses(text: str) -> tuple[tuple[str, str | None], ...]:
    """Split into ``(body, terminating mark)`` pairs on eSpeak clause marks."""

    clauses: list[tuple[str, str | None]] = []
    pending = ""
    for piece in _CLAUSE_SPLIT.split(text):
        if piece in _CLAUSE_MARKS and len(piece) == 1:
            clauses.append((pending.strip(), piece))
            pending = ""
        else:
            pending += piece
    remainder = pending.strip()
    if remainder:
        clauses.append((remainder, None))
    return tuple(clauses)


def clean_espeak_output(raw: str) -> str:
    """Strip language-switch markers and collapse eSpeak's line breaks."""

    if not isinstance(raw, str):
        raise PhonemizerError("eSpeak NG must return text")
    return " ".join(_LANGUAGE_SWITCH.sub(" ", raw).split())


def _load_phoneme_map(value: object) -> Mapping[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TtsMetadataError("Piper phoneme_map must be an object")
    mapping: dict[str, tuple[str, ...]] = {}
    for symbol, replacement in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise TtsMetadataError("Piper phoneme_map keys must be non-empty strings")
        if not isinstance(replacement, list) or not replacement:
            raise TtsMetadataError(f"Piper phoneme_map value for {symbol!r} must be a list")
        for item in replacement:
            if not isinstance(item, str) or not item:
                raise TtsMetadataError(
                    f"Piper phoneme_map value for {symbol!r} must contain phoneme strings"
                )
        mapping[symbol] = tuple(replacement)
    return mapping


def _resolve_espeak_binary(
    binary_path: str | Path | None,
    *,
    locate: Callable[[str], str | None],
) -> Path:
    if binary_path is None:
        if not callable(locate):
            raise TtsAssetError("eSpeak NG lookup requires a callable locator")
        found = locate(ESPEAK_BINARY_NAME)
        if not found:
            raise TtsAssetError(
                "eSpeak NG is not installed; install espeak-ng "
                "(apt install espeak-ng on Ubuntu, brew install espeak-ng on macOS)"
            )
        return Path(found)
    if not isinstance(binary_path, (str, Path)) or not str(binary_path).strip():
        raise TtsAssetError("eSpeak NG binary path must be explicit")
    path = Path(binary_path).expanduser()
    if not path.is_absolute():
        raise TtsAssetError("eSpeak NG binary path must be absolute")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise TtsAssetError(f"eSpeak NG binary must be an executable file: {path}")
    return path


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhonemizerError("timeout must be a finite number of seconds")
    timeout = float(value)
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT_S:
        raise PhonemizerError(
            f"timeout must be greater than zero and at most {MAX_TIMEOUT_S:g} seconds"
        )
    return timeout


__all__ = [
    "BOS_PHONEME",
    "DEFAULT_ESPEAK_VOICE",
    "DEFAULT_TIMEOUT_S",
    "EOS_PHONEME",
    "ESPEAK_BINARY_NAME",
    "EspeakNgPhonemizer",
    "EspeakNgStatus",
    "EspeakPhonemeEncoder",
    "EspeakRunner",
    "PAD_PHONEME",
    "Phonemizer",
    "PhonemizerError",
    "SPACE_PHONEME",
    "SubprocessEspeakRunner",
    "VoicePhonemeConfig",
    "clean_espeak_output",
    "create_phoneme_encoder",
    "espeak_ng_status",
    "load_voice_phoneme_config",
    "normalize_text",
    "split_clauses",
]
