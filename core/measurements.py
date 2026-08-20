"""Runtime engagement and process-resource measurements.

The latency path has its own stricter interaction timeline in
``core.instrumentation``.  This module owns the two lower-rate CSVs needed by
the technical note:

* ``engagement.csv`` records one row for every valid-gaze acquisition attempt;
* ``resources.csv`` samples the Python core's CPU and resident memory.

CSV writes never run on the behavior or animation thread.  Engagement rows are
submitted to a single worker and resource samples are taken by their own
one-second monitor thread.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import resource
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .blackboard import BlackboardSnapshot
from .fsm import FSMStatus, BehaviorState, Transition
from .instrumentation import CSVFileError, LocalCSVFileBoundary

LOGGER = logging.getLogger(__name__)


ENGAGEMENT_COLUMNS: Final = (
    "trial_id",
    "t_started",
    "t_completed",
    "outcome",
    "latency_ms",
    "reason",
    "peak_confidence",
)

RESOURCE_COLUMNS: Final = (
    "t",
    "elapsed_s",
    "core_cpu_percent",
    "core_rss_mb",
    "core_peak_rss_mb",
    "logical_cpus",
    "rss_source",
)


class MeasurementCSV:
    """Append rows while enforcing a stable, restart-safe CSV header."""

    def __init__(self, path: str | Path, columns: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.columns = columns
        self._file = LocalCSVFileBoundary()
        self._lock = threading.Lock()

    def append(self, values: tuple[object, ...]) -> None:
        if len(values) != len(self.columns):
            raise ValueError("measurement row does not match its CSV columns")
        row = _csv_line(values)
        header = _csv_line(self.columns)
        with self._lock:
            existing = self._file.read(self.path)
            if not existing:
                payload = header + row
            else:
                if not existing.endswith(b"\n"):
                    raise CSVFileError(
                        f"existing measurement CSV has an incomplete final row: {self.path}"
                    )
                first_line = existing.splitlines(keepends=True)[0]
                if first_line != header or existing.count(header) != 1:
                    raise CSVFileError(
                        f"existing measurement CSV header differs from locked columns: {self.path}"
                    )
                payload = row
            self._file.append(self.path, payload)


@dataclass(slots=True)
class _EngagementAttempt:
    trial_id: int
    started_at: float
    peak_confidence: float


class EngagementRecorder:
    """Record operational gaze-to-engagement acquisition reliability.

    An attempt begins when the FSM first accepts a valid on-target gaze while
    dormant.  It succeeds on ``NOTICING -> ENGAGED`` and fails if gaze is lost
    before that transition.  These are acquisition outcomes, not human-labeled
    detector precision/recall; the CSV names and README keep that distinction
    explicit.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        executor: Executor | None = None,
    ) -> None:
        self._csv = MeasurementCSV(path, ENGAGEMENT_COLUMNS)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="luxo-engagement"
        )
        self._owns_executor = executor is None
        self._lock = threading.RLock()
        self._attempt: _EngagementAttempt | None = None
        self._next_trial_id = 1
        self._closed = False
        self.rows_submitted = 0
        self.rows_dropped = 0

    def observe(
        self,
        snapshot: BlackboardSnapshot,
        status: FSMStatus,
        transition: Transition | None,
        now: float,
    ) -> None:
        """Observe one completed behavior tick using only immutable inputs."""

        instant = _finite_nonnegative(now, "now")
        with self._lock:
            if self._closed:
                return

            attempt = self._attempt
            began_from_dormant = (
                status.state is BehaviorState.DORMANT
                or (
                    transition is not None
                    and transition.previous is BehaviorState.DORMANT
                )
            )
            if attempt is None and status.gaze_on and began_from_dormant:
                started_at = status.gaze_on_since
                if started_at is None or not math.isfinite(started_at):
                    started_at = instant
                attempt = _EngagementAttempt(
                    trial_id=self._next_trial_id,
                    started_at=max(0.0, min(instant, started_at)),
                    peak_confidence=max(0.0, min(1.0, float(snapshot.gaze.conf))),
                )
                self._next_trial_id += 1
                self._attempt = attempt

            if attempt is None:
                return
            attempt.peak_confidence = max(
                attempt.peak_confidence,
                max(0.0, min(1.0, float(snapshot.gaze.conf))),
            )

            if (
                transition is not None
                and transition.previous is BehaviorState.NOTICING
                and transition.current is BehaviorState.ENGAGED
            ):
                self._finish_locked(instant, "success", transition.reason)
            elif (
                transition is not None
                and transition.current is BehaviorState.DORMANT
                and transition.reason == "gaze_lost_during_notice"
            ):
                self._finish_locked(instant, "failure", transition.reason)
            elif status.state is BehaviorState.DORMANT and not status.gaze_on:
                self._finish_locked(instant, "failure", "gaze_lost_before_dwell")

    def cancel(self, now: float, reason: str) -> None:
        """End an in-flight attempt without counting it as success or failure."""

        instant = _finite_nonnegative(now, "now")
        with self._lock:
            if self._attempt is not None and not self._closed:
                self._finish_locked(instant, "aborted", reason)

    def close(self, now: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if self._attempt is not None:
                self._finish_locked(
                    time.time() if now is None else _finite_nonnegative(now, "now"),
                    "aborted",
                    "shutdown",
                )
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _finish_locked(self, completed_at: float, outcome: str, reason: str) -> None:
        attempt = self._attempt
        if attempt is None:
            return
        self._attempt = None
        completed = max(completed_at, attempt.started_at)
        row = (
            attempt.trial_id,
            format(attempt.started_at, ".6f"),
            format(completed, ".6f"),
            outcome,
            format((completed - attempt.started_at) * 1000.0, ".3f"),
            reason,
            format(attempt.peak_confidence, ".4f"),
        )
        try:
            self._executor.submit(self._append, row)
            self.rows_submitted += 1
        except Exception as error:
            self.rows_dropped += 1
            LOGGER.warning(
                "engagement row submission failed (%s)", type(error).__name__
            )

    def _append(self, row: tuple[object, ...]) -> None:
        try:
            self._csv.append(row)
        except Exception as error:
            with self._lock:
                self.rows_dropped += 1
            LOGGER.warning("engagement row write failed (%s)", type(error).__name__)


class ResourceRecorder:
    """Sample core CPU and resident memory on a dedicated monitor thread."""

    def __init__(
        self,
        path: str | Path,
        *,
        interval_s: float = 1.0,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        process_time: Callable[[], float] = time.process_time,
        memory_sample: Callable[[], tuple[float, float, str]] | None = None,
    ) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("interval_s must be positive and finite")
        self._csv = MeasurementCSV(path, RESOURCE_COLUMNS)
        self._interval_s = float(interval_s)
        self._clock = clock
        self._monotonic = monotonic
        self._process_time = process_time
        self._memory_sample = memory_sample or _process_memory_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._previous_wall: float | None = None
        self._previous_cpu: float | None = None
        self.rows_written = 0
        self.rows_dropped = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._previous_wall = self._monotonic()
            self._previous_cpu = self._process_time()
            self._thread = threading.Thread(
                target=self._run,
                name="luxo-resources",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=max(0.0, timeout))

    def sample_once(self) -> tuple[object, ...] | None:
        """Take and persist one sample; public to support deterministic checks."""

        wall = self._monotonic()
        cpu = self._process_time()
        with self._lock:
            previous_wall = self._previous_wall
            previous_cpu = self._previous_cpu
            self._previous_wall = wall
            self._previous_cpu = cpu
        if previous_wall is None or previous_cpu is None:
            return None
        elapsed = wall - previous_wall
        if elapsed <= 0.0 or not math.isfinite(elapsed):
            return None
        cpu_percent = max(0.0, (cpu - previous_cpu) / elapsed * 100.0)
        rss_mb, peak_rss_mb, source = self._memory_sample()
        row: tuple[object, ...] = (
            format(_finite_nonnegative(self._clock(), "clock"), ".6f"),
            format(elapsed, ".3f"),
            format(cpu_percent, ".2f"),
            format(rss_mb, ".2f"),
            format(peak_rss_mb, ".2f"),
            os.cpu_count() or 1,
            source,
        )
        try:
            self._csv.append(row)
        except Exception as error:
            self.rows_dropped += 1
            LOGGER.warning("resource row write failed (%s)", type(error).__name__)
            return None
        self.rows_written += 1
        return row

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.sample_once()
            except Exception as error:
                self.rows_dropped += 1
                LOGGER.warning(
                    "resource sampling failed (%s)", type(error).__name__
                )


def _process_memory_mb() -> tuple[float, float, str]:
    """Return current RSS, peak RSS, and the source of the current value."""

    peak_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_bytes = peak_raw if sys.platform == "darwin" else peak_raw * 1024.0
    peak_mb = peak_bytes / (1024.0 * 1024.0)

    # Linux's procfs exposes current resident pages without another dependency
    # or child process.  On macOS, ru_maxrss is the only stdlib measurement, so
    # it is deliberately labeled as a peak fallback rather than called current.
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        rss_mb = resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
        return rss_mb, max(rss_mb, peak_mb), "procfs_current"
    except (OSError, ValueError, IndexError):
        return peak_mb, peak_mb, "getrusage_peak_fallback"


def _csv_line(values: tuple[object, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue().encode("utf-8")


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return number


__all__ = [
    "ENGAGEMENT_COLUMNS",
    "RESOURCE_COLUMNS",
    "EngagementRecorder",
    "MeasurementCSV",
    "ResourceRecorder",
]
