#!/usr/bin/env python3
"""Core-side preflight for Luxo (PRD 12.2).

Run ``python3 doctor.py`` before ``./run.sh``. Every check reports PASS, FAIL,
or SKIP with actionable remediation, and the process exits non-zero when any
check fails. The browser half of preflight lives in the ``/selftest`` page and
is deliberately not duplicated here.

Design: every check takes its environment probe as a parameter with a real
default, so the whole module runs offline against injected fakes -- no network,
no sockets, no subprocesses, no model files on disk.

Security, non-negotiable:

* The root ``.env`` is never opened, read, copied, or referenced as a data
  source. Only the already-exported process environment is consulted.
* ``OPENROUTER_API_KEY`` is probed for *presence only*. ``key_is_present``
  returns a bool, so the secret never enters the checking or reporting layer
  and cannot be printed -- not as a prefix, suffix, length, or masked form.
* The optional live reachability probe reads the key inside itself, sends it
  only as an ``Authorization`` header to OpenRouter, and returns a closed-enum
  outcome carrying a category and an HTTP status. It never reads or returns a
  request or response body, so no credential can travel back into output.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parent

# PRD 2: the core language is locked to Python 3.12.
PYTHON_FLOOR = (3, 12)

# PRD 10 and 12.3: the service is loopback only. The doctor refuses to probe
# any other bind address, so it can never open a LAN-reachable socket itself.
LOOPBACK_HOST = "127.0.0.1"
SERVICE_PORT = 8765

# PRD 11.4: expected core RSS is about 700 MB on the 8 GB deploy target. The
# requirement adds headroom for model-load transients; the browser needs its
# own memory on top of this.
MEBIBYTE = 1024 * 1024
CORE_RSS_BYTES = 700 * MEBIBYTE
REQUIRED_AVAILABLE_BYTES = CORE_RSS_BYTES * 3 // 2

API_KEY_VARIABLE = "OPENROUTER_API_KEY"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
REACHABILITY_TIMEOUT_S = 10.0

# eSpeak NG is the Piper phonemiser. The owner approved it as a system
# dependency; setup.sh installs it, and this check reports it when absent.
ESPEAK_BINARY = "espeak-ng"

DEFAULT_MANIFEST_PATH = REPO_ROOT / "config" / "models.yaml"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMAND_TIMEOUT_S = 10.0
_HASH_CHUNK_BYTES = 1 << 16


class Status(str, Enum):
    """Outcome of a single check. Only FAIL affects the exit code."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    remediation: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL


def _passed(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Status.PASS, detail)


def _failed(name: str, detail: str, remediation: Sequence[str] = ()) -> CheckResult:
    return CheckResult(name, Status.FAIL, detail, tuple(remediation))


def _skipped(name: str, detail: str, remediation: Sequence[str] = ()) -> CheckResult:
    return CheckResult(name, Status.SKIP, detail, tuple(remediation))


# --------------------------------------------------------------------------
# Injectable probe types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterpreterInfo:
    """The interpreter facts the doctor needs, captured as plain data."""

    version: tuple[int, ...]
    prefix: str
    base_prefix: str
    executable: str

    @classmethod
    def real(cls) -> "InterpreterInfo":
        return cls(
            version=tuple(sys.version_info[:3]),
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
            executable=sys.executable,
        )


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    available_bytes: int | None
    total_bytes: int | None
    source: str


@dataclass(frozen=True, slots=True)
class CommandOutput:
    returncode: int
    stdout: str


REACHABILITY_CATEGORIES = (
    "ok",
    "unauthorized",
    "unexpected_status",
    "unreachable",
    "timeout",
    "missing_key",
    "probe_error",
)


@dataclass(frozen=True, slots=True)
class ReachabilityOutcome:
    """A closed-enum result. It structurally cannot carry credential material."""

    category: str
    http_status: int | None = None

    def __post_init__(self) -> None:
        if self.category not in REACHABILITY_CATEGORIES:
            raise ValueError(f"unknown reachability category: {self.category!r}")


class Filesystem(Protocol):
    """Read-only filesystem access, injected so checks need no real files."""

    def expand(self, path: str) -> str: ...

    def is_file(self, path: str) -> bool: ...

    def size_bytes(self, path: str) -> int: ...

    def sha256(self, path: str) -> str: ...

    def read_text(self, path: str) -> str: ...


class RealFilesystem:
    """The default :class:`Filesystem`, backed by the standard library."""

    def expand(self, path: str) -> str:
        # expanduser only, deliberately not expandvars: expanding environment
        # variables inside a path could interpolate a secret into printed text.
        return os.path.expanduser(path)

    def is_file(self, path: str) -> bool:
        return Path(path).is_file()

    def size_bytes(self, path: str) -> int:
        return Path(path).stat().st_size

    def sha256(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")


def run_command(command: Sequence[str], *, timeout_s: float = _COMMAND_TIMEOUT_S) -> CommandOutput:
    """Default subprocess probe. Failures degrade to a non-zero return code."""

    try:
        completed = subprocess.run(
            tuple(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return CommandOutput(returncode=-1, stdout="")
    return CommandOutput(returncode=completed.returncode, stdout=completed.stdout or "")


def bind_loopback(host: str, port: int) -> None:
    """Default port probe: bind and listen on loopback, then close.

    ``SO_REUSEADDR`` is deliberately not set, so an existing listener surfaces
    as ``EADDRINUSE`` instead of being masked.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
        sock.listen(1)


def key_is_present(variable: str = API_KEY_VARIABLE) -> bool:
    """Presence-only probe. The value is discarded inside this function."""

    value = os.environ.get(variable)
    return bool(value and value.strip())


def read_memory_snapshot(
    filesystem: Filesystem,
    runner: Callable[[Sequence[str]], CommandOutput],
) -> MemorySnapshot:
    """Read available/total memory from ``/proc/meminfo``, else macOS tools."""

    if filesystem.is_file("/proc/meminfo"):
        try:
            return parse_meminfo(filesystem.read_text("/proc/meminfo"))
        except OSError:
            return MemorySnapshot(None, None, "unreadable /proc/meminfo")
    return read_darwin_memory(runner)


def parse_meminfo(text: str) -> MemorySnapshot:
    """Parse the ``MemAvailable``/``MemTotal`` kB fields of ``/proc/meminfo``."""

    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "b"
        fields[key.strip()] = value * 1024 if unit == "kb" else value
    return MemorySnapshot(
        available_bytes=fields.get("MemAvailable"),
        total_bytes=fields.get("MemTotal"),
        source="/proc/meminfo",
    )


def read_darwin_memory(runner: Callable[[Sequence[str]], CommandOutput]) -> MemorySnapshot:
    """Approximate available memory on macOS from ``vm_stat`` and ``sysctl``."""

    stats = runner(("vm_stat",))
    if stats.returncode != 0:
        return MemorySnapshot(None, None, "no supported memory source")

    page_size = 4096
    page_match = re.search(r"page size of (\d+) bytes", stats.stdout)
    if page_match:
        page_size = int(page_match.group(1))

    pages = 0
    found = False
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{label}:\s+(\d+)\.", stats.stdout, re.MULTILINE)
        if match:
            pages += int(match.group(1))
            found = True
    if not found:
        return MemorySnapshot(None, None, "unparsable vm_stat output")

    total: int | None = None
    memsize = runner(("sysctl", "-n", "hw.memsize"))
    if memsize.returncode == 0:
        try:
            total = int(memsize.stdout.strip())
        except ValueError:
            total = None
    return MemorySnapshot(available_bytes=pages * page_size, total_bytes=total, source="vm_stat")


def probe_openrouter(
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    env_get: Callable[[str], str | None] = os.environ.get,
    url: str = OPENROUTER_KEY_URL,
    timeout_s: float = REACHABILITY_TIMEOUT_S,
) -> ReachabilityOutcome:
    """Optional live probe of OpenRouter, opt-in only.

    The key is read here and leaves only inside the ``Authorization`` header.
    No request or response body is ever read, stored, or returned: the caller
    receives a category from a closed set plus an HTTP status code.
    """

    secret = env_get(API_KEY_VARIABLE)
    if not secret or not secret.strip():
        return ReachabilityOutcome("missing_key")

    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {secret.strip()}",
            "Accept": "application/json",
        },
    )
    del secret

    try:
        with opener(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", 0) or 0)
    except urllib.error.HTTPError as exc:
        # exc holds the response body; it is deliberately never read.
        return _classify_status(int(exc.code))
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return ReachabilityOutcome("timeout")
        return ReachabilityOutcome("unreachable")
    except TimeoutError:
        return ReachabilityOutcome("timeout")
    except OSError:
        return ReachabilityOutcome("unreachable")
    except Exception:  # noqa: BLE001 - a probe must never raise into the report
        return ReachabilityOutcome("probe_error")
    return _classify_status(status)


def _classify_status(status: int) -> ReachabilityOutcome:
    if 200 <= status < 300:
        return ReachabilityOutcome("ok", status)
    if status in (401, 403):
        return ReachabilityOutcome("unauthorized", status)
    return ReachabilityOutcome("unexpected_status", status)


@dataclass(frozen=True)
class Probes:
    """The complete injectable environment surface of the doctor."""

    interpreter: InterpreterInfo
    filesystem: Filesystem
    memory: Callable[[], MemorySnapshot]
    bind: Callable[[str, int], None]
    key_present: Callable[[str], bool]
    which: Callable[[str], str | None]
    run_command: Callable[[Sequence[str]], CommandOutput]
    reachability: Callable[[], ReachabilityOutcome]

    @classmethod
    def real(cls) -> "Probes":
        filesystem = RealFilesystem()
        return cls(
            interpreter=InterpreterInfo.real(),
            filesystem=filesystem,
            memory=lambda: read_memory_snapshot(filesystem, run_command),
            bind=bind_loopback,
            key_present=key_is_present,
            which=shutil.which,
            run_command=run_command,
            reachability=probe_openrouter,
        )


# --------------------------------------------------------------------------
# Asset manifest: shape-tolerant, because config/models.yaml is authored
# elsewhere and its final schema is not fixed.
# --------------------------------------------------------------------------

MANIFEST_WRAPPER_KEYS = ("assets", "models", "entries", "files")
MANIFEST_PATH_KEYS = ("path", "dest", "destination", "target", "file", "filename", "local_path")
MANIFEST_HASH_KEYS = ("sha256", "sha256sum", "sha256_hex", "hash", "digest", "checksum")
MANIFEST_NAME_KEYS = ("name", "id", "key", "label")
MANIFEST_SIZE_KEYS = ("size_bytes", "bytes", "size")


@dataclass(frozen=True, slots=True)
class AssetEntry:
    name: str
    path: str
    sha256: str
    size_bytes: int | None = None


class ManifestShapeError(ValueError):
    """The injected manifest is not in a shape the doctor can read."""


def _is_entry_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _manifest_pairs(document: Any) -> list[tuple[str | None, Any]]:
    if isinstance(document, Mapping):
        for wrapper in MANIFEST_WRAPPER_KEYS:
            if wrapper in document:
                inner = document[wrapper]
                if isinstance(inner, Mapping) or _is_entry_sequence(inner):
                    return _manifest_pairs(inner)
                raise ManifestShapeError(
                    f"manifest key '{wrapper}' must hold a list or mapping of assets"
                )
        if any(key in document for key in (*MANIFEST_PATH_KEYS, *MANIFEST_HASH_KEYS)):
            # A bare single-asset mapping, not a mapping of named assets.
            return [(None, document)]
        return [(str(key), value) for key, value in document.items()]
    if _is_entry_sequence(document):
        return [(None, item) for item in document]
    raise ManifestShapeError(
        "manifest must be a list of assets, a mapping of assets, or a mapping "
        f"with one of {list(MANIFEST_WRAPPER_KEYS)}"
    )


def _first_string(entry: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in entry:
            value = entry[key]
            if isinstance(value, str) and value.strip():
                return value.strip()
            raise ManifestShapeError(f"manifest field '{key}' must be a non-empty string")
    return None


def _first_size(entry: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in entry:
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ManifestShapeError(f"manifest field '{key}' must be a non-negative integer")
            return value
    return None


def _asset_entry(name_hint: str | None, raw: Any) -> AssetEntry:
    label = name_hint or "<unnamed>"
    if isinstance(raw, str):
        if name_hint is None:
            raise ManifestShapeError("a bare sha256 string needs its path as the mapping key")
        return AssetEntry(name=name_hint, path=name_hint, sha256=raw.strip())
    if not isinstance(raw, Mapping):
        raise ManifestShapeError(f"asset '{label}' must be a mapping or a sha256 string")

    path = _first_string(raw, MANIFEST_PATH_KEYS) or name_hint
    if path is None:
        raise ManifestShapeError(
            f"asset '{label}' has no path field (any of {list(MANIFEST_PATH_KEYS)})"
        )
    digest = _first_string(raw, MANIFEST_HASH_KEYS)
    if digest is None:
        raise ManifestShapeError(
            f"asset '{label}' has no sha256 field (any of {list(MANIFEST_HASH_KEYS)})"
        )
    name = _first_string(raw, MANIFEST_NAME_KEYS) or name_hint or path
    return AssetEntry(
        name=name,
        path=path,
        sha256=digest,
        size_bytes=_first_size(raw, MANIFEST_SIZE_KEYS),
    )


def normalize_manifest(document: Any) -> list[AssetEntry]:
    """Turn an injected manifest document into entries, tolerating its shape."""

    return [_asset_entry(name, raw) for name, raw in _manifest_pairs(document)]


def load_manifest_document(
    path: str,
    filesystem: Filesystem,
) -> tuple[Any | None, str | None]:
    """Read a manifest file as JSON-compatible YAML, as core/config.py does."""

    resolved = filesystem.expand(path)
    if not filesystem.is_file(resolved):
        return None, f"no asset manifest at {resolved}"
    try:
        text = filesystem.read_text(resolved)
    except OSError as exc:
        return None, f"cannot read {resolved}: {exc.strerror or 'unreadable'}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{resolved} is not JSON-compatible YAML: {exc.msg} (line {exc.lineno})"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_python_version(
    interpreter: InterpreterInfo,
    *,
    floor: tuple[int, ...] = PYTHON_FLOOR,
) -> CheckResult:
    actual = ".".join(str(part) for part in interpreter.version)
    wanted = ".".join(str(part) for part in floor)
    if tuple(interpreter.version[: len(floor)]) >= tuple(floor):
        return _passed("python", f"Python {actual} meets the {wanted} floor")
    return _failed(
        "python",
        f"Python {actual} is below the locked {wanted} floor",
        (
            "sudo apt-get install -y python3.12 python3.12-venv   # Ubuntu 24.04",
            "rebuild the environment with it: python3.12 -m venv .venv",
        ),
    )


def check_virtualenv(interpreter: InterpreterInfo) -> CheckResult:
    if interpreter.prefix != interpreter.base_prefix:
        return _passed("venv", f"running inside a virtualenv at {interpreter.prefix}")
    return _failed(
        "venv",
        f"running from the base interpreter at {interpreter.prefix}",
        (
            "Ubuntu 24.04 marks system Python externally managed (PEP 668), so pip is blocked",
            "python3 -m venv .venv",
            ". .venv/bin/activate",
            "pip install -r requirements.txt",
        ),
    )


def _resolve_asset_path(path: str, filesystem: Filesystem, base_dir: Path) -> str:
    expanded = filesystem.expand(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(str(base_dir), expanded))


def check_asset(
    entry: AssetEntry,
    filesystem: Filesystem,
    *,
    base_dir: Path = REPO_ROOT,
) -> CheckResult:
    name = f"asset {entry.name}"
    expected = entry.sha256.strip().lower()
    if not _HEX64.fullmatch(expected):
        return _failed(
            name,
            "the manifest sha256 is not 64 hexadecimal characters",
            ("fix the manifest entry; hashes are lowercase sha256 hex digests",),
        )

    resolved = _resolve_asset_path(entry.path, filesystem, base_dir)
    if not filesystem.is_file(resolved):
        return _failed(
            name,
            f"missing: {resolved}",
            (
                "./setup.sh downloads model assets and verifies their sha256",
                "weights live under ~/.cache/lumen and are never committed",
            ),
        )

    try:
        actual_size = filesystem.size_bytes(resolved)
        actual = filesystem.sha256(resolved).lower()
    except OSError as exc:
        return _failed(
            name,
            f"cannot read {resolved}: {exc.strerror or 'unreadable'}",
            ("check file permissions, then re-run ./setup.sh",),
        )

    if entry.size_bytes is not None and actual_size != entry.size_bytes:
        return _failed(
            name,
            f"size mismatch at {resolved}: expected {entry.size_bytes} bytes, found {actual_size}",
            ("delete the file and re-run ./setup.sh to download it again",),
        )
    if actual != expected:
        return _failed(
            name,
            (
                f"sha256 mismatch at {resolved}: expected {expected[:16]}..., "
                f"found {actual[:16]}..."
            ),
            (
                "the download is truncated, corrupt, or the wrong revision",
                "delete the file and re-run ./setup.sh to download it again",
            ),
        )
    return _passed(name, f"{resolved} matches sha256 {expected[:16]}... ({actual_size} bytes)")


def check_assets(
    document: Any,
    filesystem: Filesystem,
    *,
    base_dir: Path = REPO_ROOT,
    source: str = "the asset manifest",
    load_error: str | None = None,
) -> list[CheckResult]:
    """One result per manifest entry, or one result describing a manifest fault."""

    manifest_help = (
        "./setup.sh writes the manifest and downloads the assets it names",
        "or point the doctor at one: python3 doctor.py --manifest PATH",
        "the manifest must be JSON-compatible YAML, as config/default.yaml is",
    )
    if load_error is not None:
        return [_failed("assets", load_error, manifest_help)]
    try:
        entries = normalize_manifest(document)
    except ManifestShapeError as exc:
        return [_failed("assets", f"{source}: {exc}", manifest_help)]
    if not entries:
        return [_failed("assets", f"{source} lists no assets", manifest_help)]
    return [check_asset(entry, filesystem, base_dir=base_dir) for entry in entries]


def _mib(value: int) -> int:
    return value // MEBIBYTE


def check_memory(
    snapshot: MemorySnapshot,
    *,
    required_bytes: int = REQUIRED_AVAILABLE_BYTES,
) -> CheckResult:
    if snapshot.available_bytes is None:
        return _skipped(
            "memory",
            f"cannot determine available memory ({snapshot.source})",
            (
                f"check free RAM by hand: the core needs about {_mib(CORE_RSS_BYTES)} MiB "
                "resident (PRD 11.4)",
            ),
        )
    total = f"{_mib(snapshot.total_bytes)} MiB total" if snapshot.total_bytes else "unknown total"
    detail = f"{_mib(snapshot.available_bytes)} MiB available of {total} ({snapshot.source})"
    if snapshot.available_bytes < required_bytes:
        return _failed(
            "memory",
            f"{detail}; the core wants at least {_mib(required_bytes)} MiB free",
            (
                f"core RSS is about {_mib(CORE_RSS_BYTES)} MiB (PRD 11.4) and the browser "
                "needs its own on top",
                "close other applications, or run the core on the 8 GB target with nothing else",
            ),
        )
    return _passed("memory", detail)


def check_port(
    bind: Callable[[str, int], None],
    *,
    host: str = LOOPBACK_HOST,
    port: int = SERVICE_PORT,
) -> CheckResult:
    if host != LOOPBACK_HOST:
        return _failed(
            "port",
            f"refusing to probe {host}:{port}: the core binds {LOOPBACK_HOST} only",
            (
                "the WebSocket server is loopback only and must never bind 0.0.0.0 "
                "(PRD 10, 12.3)",
            ),
        )
    try:
        bind(host, port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return _failed(
                "port",
                f"{host}:{port} is already in use",
                (
                    "ss -ltnp 'sport = :8765'                  # Ubuntu",
                    "lsof -nP -iTCP:8765 -sTCP:LISTEN          # macOS",
                    "stop the listener, or the core left running from a previous session",
                ),
            )
        return _failed(
            "port",
            f"cannot bind {host}:{port}: {exc.strerror or errno.errorcode.get(exc.errno or 0, 'error')}",
            ("confirm the loopback interface is up, then re-run the doctor",),
        )
    return _passed("port", f"{host}:{port} is free; the core binds loopback only, never 0.0.0.0")


def check_api_key(
    key_present: Callable[[str], bool],
    *,
    variable: str = API_KEY_VARIABLE,
) -> CheckResult:
    if key_present(variable):
        return _passed("api key", f"{variable} is set (presence only; the value is never read)")
    return _failed(
        "api key",
        f"{variable} is not set in this environment",
        (
            f"export {variable} in the shell that launches the core",
            "the root .env holds it and run.sh loads it; the doctor never opens that file",
            "the doctor never prints, logs, or transmits the value",
        ),
    )


def _clean_line(text: str, limit: int = 90) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    printable = "".join(char for char in first if char.isprintable())
    return printable[:limit]


def check_espeak(
    which: Callable[[str], str | None],
    runner: Callable[[Sequence[str]], CommandOutput],
    *,
    binary: str = ESPEAK_BINARY,
) -> CheckResult:
    install_help = (
        f"sudo apt-get install -y {binary}     # Ubuntu 24.04",
        f"brew install {binary}                # macOS",
        "Piper phonemisation needs it; the owner approved it as a system dependency",
    )
    path = which(binary)
    if not path:
        return _failed("espeak-ng", f"{binary} is not on PATH", install_help)
    output = runner((path, "--version"))
    if output.returncode != 0:
        return _failed(
            "espeak-ng",
            f"{path} is present but '--version' exited {output.returncode}",
            (f"sudo apt-get install --reinstall -y {binary}", *install_help[1:]),
        )
    version = _clean_line(output.stdout)
    detail = f"{path} ({version})" if version else path
    return _passed("espeak-ng", detail)


_REACHABILITY_DETAIL = {
    "ok": "OpenRouter accepted the key",
    "unauthorized": "OpenRouter rejected the key",
    "unexpected_status": "OpenRouter answered with an unexpected status",
    "unreachable": "OpenRouter could not be reached",
    "timeout": "the OpenRouter request timed out",
    "missing_key": f"{API_KEY_VARIABLE} is not set, so the live probe was not sent",
    "probe_error": "the reachability probe failed before it got an answer",
}

_REACHABILITY_HELP: dict[str, tuple[str, ...]] = {
    "unauthorized": (
        f"issue a new OpenRouter key and export it as {API_KEY_VARIABLE}",
        "a free-profile key still has to exist and be active (PRD 8.1)",
    ),
    "unexpected_status": (
        "retry shortly; if it persists, check OpenRouter status before the demo",
    ),
    "unreachable": (
        "check network access and any proxy; the core needs HTTPS to openrouter.ai",
    ),
    "timeout": ("check network access; the probe waits 10 s before giving up",),
    "missing_key": (f"export {API_KEY_VARIABLE}, then re-run with --check-openrouter",),
    "probe_error": ("re-run with --check-openrouter; report the failure if it repeats",),
}


def check_openrouter(
    reachability: Callable[[], ReachabilityOutcome],
    *,
    enabled: bool = False,
) -> CheckResult:
    """Live reachability. OFF unless ``enabled``; the probe is not even called."""

    if not enabled:
        return _skipped(
            "openrouter",
            "live reachability is off by default",
            ("pass --check-openrouter to send one authenticated request to OpenRouter",),
        )
    outcome = reachability()
    detail = _REACHABILITY_DETAIL.get(outcome.category, outcome.category)
    if outcome.http_status is not None:
        detail = f"{detail} (HTTP {outcome.http_status})"
    if outcome.category == "ok":
        return _passed("openrouter", detail)
    return _failed("openrouter", detail, _REACHABILITY_HELP.get(outcome.category, ()))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_checks(
    probes: Probes,
    *,
    manifest_path: str = str(DEFAULT_MANIFEST_PATH),
    check_openrouter_live: bool = False,
    base_dir: Path = REPO_ROOT,
) -> list[CheckResult]:
    """Run every check in report order against the injected probes."""

    results = [
        check_python_version(probes.interpreter),
        check_virtualenv(probes.interpreter),
    ]
    document, load_error = load_manifest_document(manifest_path, probes.filesystem)
    results.extend(
        check_assets(
            document,
            probes.filesystem,
            base_dir=base_dir,
            source=manifest_path,
            load_error=load_error,
        )
    )
    results.append(check_memory(probes.memory()))
    results.append(check_port(probes.bind))
    results.append(check_api_key(probes.key_present))
    results.append(check_espeak(probes.which, probes.run_command))
    results.append(check_openrouter(probes.reachability, enabled=check_openrouter_live))
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    lines = ["Luxo core preflight (PRD 12.2)", ""]
    width = max((len(result.name) for result in results), default=0)
    for result in results:
        lines.append(f"[{result.status.value}] {result.name.ljust(width)}  {result.detail}")
        for hint in result.remediation:
            lines.append(f"{' ' * 8}-> {hint}")
    passed = sum(1 for result in results if result.status is Status.PASS)
    failed = sum(1 for result in results if result.status is Status.FAIL)
    skipped = sum(1 for result in results if result.status is Status.SKIP)
    lines.append("")
    lines.append(f"{passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        lines.append("preflight failed; fix the items above before running ./run.sh")
    else:
        lines.append("core preflight is clean; run the browser /selftest page next")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doctor.py",
        description="Core-side preflight for Luxo (PRD 12.2). Offline unless asked otherwise.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        metavar="PATH",
        help="asset manifest to verify (JSON-compatible YAML); default: %(default)s",
    )
    parser.add_argument(
        "--check-openrouter",
        action="store_true",
        help="opt in to one live authenticated request to OpenRouter (off by default)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    probes: Probes | None = None,
    stdout: Any | None = None,
) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    resolved_probes = probes if probes is not None else Probes.real()
    stream = stdout if stdout is not None else sys.stdout
    results = run_checks(
        resolved_probes,
        manifest_path=args.manifest,
        check_openrouter_live=args.check_openrouter,
    )
    print(format_report(results), file=stream)
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
