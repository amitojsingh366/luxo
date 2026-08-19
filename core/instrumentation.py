"""Worker-side interaction latency timelines and serialized CSV persistence.

This module performs synchronous file I/O and must be called from worker
threads, never the animation tick. Its data model deliberately cannot contain
transcripts, prompts, responses, media, sensor facts, or body state.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


class Milestone(str, Enum):
    VAD_END = "t_vad_end"
    PCM_RECEIVED = "t_pcm_received"
    TRANSCRIPT = "t_transcript"
    REQUEST_SENT = "t_request_sent"
    RESPONSE = "t_response"
    FIRST_AUDIO_CHUNK = "t_first_audio_chunk"


MILESTONE_ORDER = tuple(Milestone)
STAGE_COLUMNS = (
    "stage_ms_vad_to_pcm",
    "stage_ms_transcription",
    "stage_ms_transcript_to_request",
    "stage_ms_model_response",
    "stage_ms_response_to_first_audio",
    "stage_ms_end_to_end",
)
CSV_COLUMNS = tuple(item.value for item in MILESTONE_ORDER) + STAGE_COLUMNS + (
    "model",
    "profile",
    "tokens_in",
    "tokens_out",
)

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,199}\Z")


class TimelineError(ValueError):
    """An interaction timeline is incomplete, out of order, or already final."""


class CSVFileError(RuntimeError):
    """An existing latency file does not have the locked CSV contract."""


@dataclass(frozen=True, slots=True)
class InteractionStatus:
    t_vad_end: float | None
    t_pcm_received: float | None
    t_transcript: float | None
    t_request_sent: float | None
    t_response: float | None
    t_first_audio_chunk: float | None
    model: str | None
    profile: str | None
    tokens_in: int | None
    tokens_out: int | None
    next_milestone: Milestone | None
    complete: bool
    commit_in_progress: bool
    committed: bool


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    t_vad_end: float
    t_pcm_received: float
    t_transcript: float
    t_request_sent: float
    t_response: float
    t_first_audio_chunk: float
    stage_ms_vad_to_pcm: float
    stage_ms_transcription: float
    stage_ms_transcript_to_request: float
    stage_ms_model_response: float
    stage_ms_response_to_first_audio: float
    stage_ms_end_to_end: float
    model: str
    profile: str
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True, slots=True)
class LoggerStatus:
    rows_written: int
    write_failures: int
    last_failure_type: str | None


class FileBoundary(Protocol):
    """Minimal injectable boundary used for recovery and failure isolation."""

    def read(self, path: Path) -> bytes: ...

    def append(self, path: Path, payload: bytes) -> None: ...


class LocalCSVFileBoundary:
    """Append one already-encoded header/row payload and flush it durably."""

    def read(self, path: Path) -> bytes:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return b""

    def append(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"short CSV append: {written} of {len(payload)} bytes")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


FailureCallback = Callable[[Exception], None]


class InteractionTimeline:
    """Thread-safe, ordered milestones for exactly one interaction."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._lock = threading.RLock()
        self._timestamps: list[float] = []
        self._model: str | None = None
        self._profile: str | None = None
        self._tokens_in: int | None = None
        self._tokens_out: int | None = None
        self._commit_in_progress = False
        self._committed = False

    def mark(self, milestone: Milestone, timestamp: float | None = None) -> float:
        """Record the next required milestone, using the injected clock by default."""

        if not isinstance(milestone, Milestone):
            raise TypeError("milestone must be a Milestone")
        instant = _timestamp(self._clock() if timestamp is None else timestamp)
        with self._lock:
            self._ensure_mutable()
            if len(self._timestamps) == len(MILESTONE_ORDER):
                raise TimelineError("all milestones are already recorded")
            expected = MILESTONE_ORDER[len(self._timestamps)]
            if milestone is not expected:
                raise TimelineError(
                    f"expected {expected.value} before {milestone.value}"
                )
            if self._timestamps and instant < self._timestamps[-1]:
                raise TimelineError("milestone timestamps must be monotonic")
            self._timestamps.append(instant)
            return instant

    def set_model_usage(
        self,
        *,
        model: str,
        profile: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Attach only the four model fields permitted by the CSV contract."""

        validated_model = _model_id(model)
        validated_profile = _profile(profile)
        validated_in = _token_count(tokens_in, "tokens_in")
        validated_out = _token_count(tokens_out, "tokens_out")
        with self._lock:
            self._ensure_mutable()
            if self._model is not None:
                raise TimelineError("model usage is already recorded")
            self._model = validated_model
            self._profile = validated_profile
            self._tokens_in = validated_in
            self._tokens_out = validated_out

    def status(self) -> InteractionStatus:
        """Return an immutable copy with no private interaction payloads."""

        with self._lock:
            values: list[float | None] = self._timestamps + [None] * (
                len(MILESTONE_ORDER) - len(self._timestamps)
            )
            next_milestone = (
                MILESTONE_ORDER[len(self._timestamps)]
                if len(self._timestamps) < len(MILESTONE_ORDER)
                else None
            )
            complete = next_milestone is None and self._model is not None
            return InteractionStatus(
                *values,
                self._model,
                self._profile,
                self._tokens_in,
                self._tokens_out,
                next_milestone,
                complete,
                self._commit_in_progress,
                self._committed,
            )

    def _claim_record(self) -> InteractionRecord:
        with self._lock:
            self._ensure_mutable()
            if len(self._timestamps) != len(MILESTONE_ORDER):
                raise TimelineError("all six milestones are required before commit")
            if self._model is None or self._profile is None:
                raise TimelineError("model usage is required before commit")
            if self._tokens_in is None or self._tokens_out is None:
                raise TimelineError("token counts are required before commit")
            self._commit_in_progress = True
            times = tuple(self._timestamps)
            return _record(
                times,
                self._model,
                self._profile,
                self._tokens_in,
                self._tokens_out,
            )

    def _finish_commit(self, succeeded: bool) -> None:
        with self._lock:
            if not self._commit_in_progress:
                raise AssertionError("timeline has no commit in progress")
            self._commit_in_progress = False
            self._committed = succeeded

    def _ensure_mutable(self) -> None:
        if self._committed:
            raise TimelineError("interaction is already committed")
        if self._commit_in_progress:
            raise TimelineError("interaction commit is in progress")


class InteractionCSVLogger:
    """Serialize complete interaction rows and isolate persistence failures."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        file_boundary: FileBoundary | None = None,
        failure_callback: FailureCallback | None = None,
    ) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if failure_callback is not None and not callable(failure_callback):
            raise TypeError("failure_callback must be callable")
        self._path = Path(path)
        self._clock = clock
        self._files = file_boundary or LocalCSVFileBoundary()
        self._failure_callback = failure_callback
        self._write_lock = threading.Lock()
        self._rows_written = 0
        self._write_failures = 0
        self._last_failure_type: str | None = None

    def new_interaction(self) -> InteractionTimeline:
        """Create an in-memory timeline without touching the filesystem."""

        return InteractionTimeline(clock=self._clock)

    def commit(self, timeline: InteractionTimeline) -> InteractionRecord | None:
        """Write one complete row, returning ``None`` on isolated I/O failure."""

        if not isinstance(timeline, InteractionTimeline):
            raise TypeError("timeline must be an InteractionTimeline")
        failure: Exception | None = None
        record: InteractionRecord
        with self._write_lock:
            record = timeline._claim_record()
            succeeded = False
            try:
                existing = self._files.read(self._path)
                payload = _append_payload(existing, record)
                self._files.append(self._path, payload)
                succeeded = True
                self._rows_written += 1
                self._last_failure_type = None
            except Exception as error:
                failure = error
                self._write_failures += 1
                self._last_failure_type = type(error).__name__
            finally:
                timeline._finish_commit(succeeded)

        if failure is not None:
            self._notify_failure(failure)
            return None
        return record

    def status(self) -> LoggerStatus:
        with self._write_lock:
            return LoggerStatus(
                self._rows_written,
                self._write_failures,
                self._last_failure_type,
            )

    def _notify_failure(self, error: Exception) -> None:
        if self._failure_callback is None:
            return
        try:
            self._failure_callback(error)
        except Exception:
            pass


def _record(
    times: tuple[float, ...],
    model: str,
    profile: str,
    tokens_in: int,
    tokens_out: int,
) -> InteractionRecord:
    vad, pcm, transcript, request, response, audio = times
    return InteractionRecord(
        vad,
        pcm,
        transcript,
        request,
        response,
        audio,
        (pcm - vad) * 1000.0,
        (transcript - pcm) * 1000.0,
        (request - transcript) * 1000.0,
        (response - request) * 1000.0,
        (audio - response) * 1000.0,
        (audio - vad) * 1000.0,
        model,
        profile,
        tokens_in,
        tokens_out,
    )


def _append_payload(existing: bytes, record: InteractionRecord) -> bytes:
    if not isinstance(existing, bytes):
        raise CSVFileError("file boundary read must return bytes")
    header = _csv_line(CSV_COLUMNS)
    row = _csv_line(_record_values(record))
    if not existing:
        return header + row
    if not existing.endswith(b"\n"):
        raise CSVFileError("existing latency CSV has an incomplete final row")
    first_line = existing.splitlines(keepends=True)[0]
    if first_line != header or existing.count(header) != 1:
        raise CSVFileError("existing latency CSV header differs from locked columns")
    return row


def _record_values(record: InteractionRecord) -> tuple[object, ...]:
    timestamps = (
        record.t_vad_end,
        record.t_pcm_received,
        record.t_transcript,
        record.t_request_sent,
        record.t_response,
        record.t_first_audio_chunk,
    )
    stages = (
        record.stage_ms_vad_to_pcm,
        record.stage_ms_transcription,
        record.stage_ms_transcript_to_request,
        record.stage_ms_model_response,
        record.stage_ms_response_to_first_audio,
        record.stage_ms_end_to_end,
    )
    return (
        *(format(value, ".9f") for value in timestamps),
        *(format(value, ".3f") for value in stages),
        record.model,
        record.profile,
        record.tokens_in,
        record.tokens_out,
    )


def _csv_line(values: tuple[object, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue().encode("utf-8")


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timestamp must be a finite nonnegative number")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("timestamp must be a finite nonnegative number")
    return timestamp


def _model_id(value: object) -> str:
    if not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None:
        raise ValueError("model must be a compact provider model id")
    return value


def _profile(value: object) -> str:
    if value not in ("free", "private") or not isinstance(value, str):
        raise ValueError("profile must be 'free' or 'private'")
    return value


def _token_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


__all__ = [
    "CSV_COLUMNS",
    "CSVFileError",
    "FileBoundary",
    "InteractionCSVLogger",
    "InteractionRecord",
    "InteractionStatus",
    "InteractionTimeline",
    "LocalCSVFileBoundary",
    "LoggerStatus",
    "MILESTONE_ORDER",
    "Milestone",
    "STAGE_COLUMNS",
    "TimelineError",
]
