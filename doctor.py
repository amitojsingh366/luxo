#!/usr/bin/env python3
"""Core-side preflight for Luxo (PRD 12.2).

Run ``python3 doctor.py`` before ``./run.sh``. Every check reports PASS, FAIL,
WARN, or SKIP with actionable remediation, and the process exits non-zero when
any check fails. Only FAIL affects the exit code: a WARN records a real but
accepted gap -- an asset that is present but that the manifest cannot pin to a
digest -- and must not block the demo. The browser half of preflight lives in
the ``/selftest`` page and is deliberately not duplicated here.

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

# PRD 2: the core language is locked to Python 3.12 exactly -- not a floor.
# requirements.txt pins onnxruntime 1.20.1, which publishes no sdist and no
# pure-Python wheel, so pip must take an ABI-tagged binary wheel and only the
# cp312 ones are hashed. numpy 2.5.2 requires >= 3.12, which sets the other
# side. A newer Python is as broken as an older one, so this is an equality.
REQUIRED_PYTHON = (3, 12)

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
_HEX32 = re.compile(r"[0-9a-f]{32}")
_COMMAND_TIMEOUT_S = 10.0
_HASH_CHUNK_BYTES = 1 << 16


class Status(str, Enum):
    """Outcome of a single check. Only FAIL affects the exit code."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
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

    @property
    def warned(self) -> bool:
        return self.status is Status.WARN


def _passed(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Status.PASS, detail)


def _failed(name: str, detail: str, remediation: Sequence[str] = ()) -> CheckResult:
    return CheckResult(name, Status.FAIL, detail, tuple(remediation))


def _warned(name: str, detail: str, remediation: Sequence[str] = ()) -> CheckResult:
    return CheckResult(name, Status.WARN, detail, tuple(remediation))


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
    """Read-only filesystem access, injected so checks need no real files.

    This is one shared probe contract: an implementation provides every member,
    because any check may reach for any of them. ``is_dir`` and ``list_dir``
    exist for directory assets, whose destination is a staged directory of
    required files rather than a single downloaded file. ``md5`` exists only to
    cross-check an asset whose publisher ships no SHA-256; it is never a
    supply-chain pin.
    """

    def expand(self, path: str) -> str: ...

    def is_file(self, path: str) -> bool: ...

    def is_dir(self, path: str) -> bool: ...

    def list_dir(self, path: str) -> tuple[str, ...]: ...

    def size_bytes(self, path: str) -> int: ...

    def sha256(self, path: str) -> str: ...

    def md5(self, path: str) -> str: ...

    def read_text(self, path: str) -> str: ...


class RealFilesystem:
    """The default :class:`Filesystem`, backed by the standard library."""

    def expand(self, path: str) -> str:
        # expanduser only, deliberately not expandvars: expanding environment
        # variables inside a path could interpolate a secret into printed text.
        return os.path.expanduser(path)

    def is_file(self, path: str) -> bool:
        return Path(path).is_file()

    def is_dir(self, path: str) -> bool:
        return Path(path).is_dir()

    def list_dir(self, path: str) -> tuple[str, ...]:
        """Immediate child names, sorted. An unreadable directory lists empty.

        Only used to name files a directory holds but the manifest does not
        require, which is reported and never fails a check, so a listing that
        cannot be read degrades to saying nothing rather than to an error.
        """

        try:
            with os.scandir(path) as entries:
                return tuple(sorted(entry.name for entry in entries))
        except OSError:
            return ()

    def size_bytes(self, path: str) -> int:
        return Path(path).stat().st_size

    def sha256(self, path: str) -> str:
        return self._digest(path, hashlib.sha256)

    def md5(self, path: str) -> str:
        # usedforsecurity=False states the intent and keeps this available on a
        # FIPS build: md5 here is a corruption cross-check, never a pin.
        return self._digest(path, lambda: hashlib.md5(usedforsecurity=False))

    def read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    @staticmethod
    def _digest(path: str, factory: Callable[[], Any]) -> str:
        digest = factory()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


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

# config/models.yaml nests each destination as {layout, path, served_at}, so a
# path field may be a bare string or a mapping that carries the path under this
# key. Both shapes are accepted; the layout roots in the manifest's own
# 'destinations' table are a lookup for setup.sh, not part of the asset path.
MANIFEST_NESTED_PATH_KEY = "path"
MANIFEST_VERIFY_KEY = "verify_command"

# An asset is a single file unless it declares otherwise. A directory asset has
# no single digest, so it carries a required_files table instead: one record per
# file, each path relative to that entry's own destination directory. That
# rooting is stated in the manifest's about block and enforced by
# the validation tooling
MANIFEST_KIND_KEYS = ("kind", "type")
MANIFEST_REQUIRED_FILES_KEY = "required_files"
FILE_KIND = "file"
DIRECTORY_KIND = "directory"
DIRECTORY_KINDS = (DIRECTORY_KIND, "dir")

# A publisher digest in an algorithm that is not the pin. It is read only to
# catch a corrupted download of an asset whose sha256 cannot be pinned, and the
# report says so wherever it appears.
MANIFEST_ADDITIONAL_DIGESTS_KEY = "additional_digests"
MANIFEST_MD5_KEY = "md5"

# The manifest may record a literal marker in place of a digest for an asset
# whose publisher ships no SHA-256. It declares the marker itself; this is only
# the fallback for a manifest that uses one without naming it.
MANIFEST_MARKER_KEY = "unverified_marker"
DEFAULT_UNVERIFIED_MARKER = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class RequiredFile:
    """One file a directory asset must contain.

    ``path`` is relative to the owning asset's own destination directory, never
    to the upstream package root: joining the two gives the staged file.
    """

    path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class AssetEntry:
    name: str
    path: str
    sha256: str
    size_bytes: int | None = None
    verify_command: str | None = None
    kind: str = FILE_KIND
    required_files: tuple[RequiredFile, ...] = ()
    md5: str | None = None

    @property
    def is_directory(self) -> bool:
        return self.kind.strip().casefold() in DIRECTORY_KINDS


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


def _entry_path(entry: Mapping[str, Any], keys: Sequence[str] = MANIFEST_PATH_KEYS) -> str | None:
    """Read the destination path, accepting a bare string or a nested mapping.

    ``{"destination": "~/.cache/luxo/x.bin"}`` and
    ``{"destination": {"layout": "core_cache", "path": "~/.cache/luxo/x.bin"}}``
    both yield the same path. Only the ``path`` member of the mapping is read;
    ``layout`` and ``served_at`` are consumed by setup.sh and the renderer.
    """

    for key in keys:
        if key not in entry:
            continue
        value = entry[key]
        if isinstance(value, Mapping):
            nested = value.get(MANIFEST_NESTED_PATH_KEY)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
            raise ManifestShapeError(
                f"manifest field '{key}' is a mapping, so it must carry a non-empty "
                f"string '{MANIFEST_NESTED_PATH_KEY}'"
            )
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ManifestShapeError(f"manifest field '{key}' must be a non-empty string")
    return None


def _optional_string(entry: Mapping[str, Any], key: str) -> str | None:
    """Read a purely informational field. A bad value is dropped, never fatal."""

    value = entry.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_size(entry: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in entry:
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ManifestShapeError(f"manifest field '{key}' must be a non-negative integer")
            return value
    return None


def _required_files(label: str, entry: Mapping[str, Any]) -> tuple[RequiredFile, ...]:
    """Read the per-file table a directory asset is verified through.

    Each record is read with the same helpers as a top-level asset, so a
    required file and a whole asset are described in the same vocabulary.
    """

    raw = entry.get(MANIFEST_REQUIRED_FILES_KEY)
    if raw is None:
        return ()
    if not _is_entry_sequence(raw):
        raise ManifestShapeError(
            f"asset '{label}' has a '{MANIFEST_REQUIRED_FILES_KEY}' that is not a list"
        )
    files: list[RequiredFile] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ManifestShapeError(
                f"asset '{label}' has a required file that is not a mapping"
            )
        path = _entry_path(item)
        if path is None:
            raise ManifestShapeError(
                f"asset '{label}' has a required file with no path "
                f"(any of {list(MANIFEST_PATH_KEYS)})"
            )
        digest = _first_string(item, MANIFEST_HASH_KEYS)
        if digest is None:
            raise ManifestShapeError(
                f"asset '{label}' has a required file '{path}' with no sha256 "
                f"(any of {list(MANIFEST_HASH_KEYS)})"
            )
        files.append(
            RequiredFile(
                path=path,
                sha256=digest,
                size_bytes=_first_size(item, MANIFEST_SIZE_KEYS),
            )
        )
    return tuple(files)


def _declared_kind(entry: Mapping[str, Any]) -> str | None:
    for key in MANIFEST_KIND_KEYS:
        declared = _optional_string(entry, key)
        if declared is not None:
            return declared
    return None


def _additional_md5(entry: Mapping[str, Any]) -> str | None:
    """Read the optional publisher md5. Unusable values are simply dropped.

    md5 is not collision-resistant and is never the pin. It is recorded by a
    publisher that ships no SHA-256, and reading it lets the doctor still catch
    a corrupted download of that asset.
    """

    extra = entry.get(MANIFEST_ADDITIONAL_DIGESTS_KEY)
    if not isinstance(extra, Mapping):
        return None
    value = extra.get(MANIFEST_MD5_KEY)
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if _HEX32.fullmatch(candidate) else None


def _asset_entry(name_hint: str | None, raw: Any) -> AssetEntry:
    label = name_hint or "<unnamed>"
    if isinstance(raw, str):
        if name_hint is None:
            raise ManifestShapeError("a bare sha256 string needs its path as the mapping key")
        return AssetEntry(name=name_hint, path=name_hint, sha256=raw.strip())
    if not isinstance(raw, Mapping):
        raise ManifestShapeError(f"asset '{label}' must be a mapping or a sha256 string")

    path = _entry_path(raw) or name_hint
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
    required = _required_files(label, raw)
    # An entry that names required files is a directory even if it forgot to
    # say so; anything else is a single file unless it declares a kind.
    kind = _declared_kind(raw) or (DIRECTORY_KIND if required else FILE_KIND)
    return AssetEntry(
        name=name,
        path=path,
        sha256=digest,
        size_bytes=_first_size(raw, MANIFEST_SIZE_KEYS),
        verify_command=_optional_string(raw, MANIFEST_VERIFY_KEY),
        kind=kind,
        required_files=required,
        md5=_additional_md5(raw),
    )


def normalize_manifest(document: Any) -> list[AssetEntry]:
    """Turn an injected manifest document into entries, tolerating its shape."""

    return [_asset_entry(name, raw) for name, raw in _manifest_pairs(document)]


def manifest_unverified_marker(document: Any) -> str:
    """Read the manifest's own declaration of its UNVERIFIED marker."""

    if isinstance(document, Mapping):
        declared = document.get(MANIFEST_MARKER_KEY)
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    return DEFAULT_UNVERIFIED_MARKER


def _is_unverified(digest: str, marker: str) -> bool:
    """True when the manifest recorded the marker instead of a real digest."""

    if not marker or not marker.strip():
        return False
    return digest.strip().casefold() == marker.strip().casefold()


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


# Both failure branches share the install steps; only the leading sentence
# differs, because "your Python is too new" is the counter-intuitive half.
_PYTHON_INSTALL_HELP = (
    "sudo apt-get install -y python3.12 python3.12-venv   # Ubuntu 24.04",
    "rebuild the environment with it: python3.12 -m venv .venv",
)


def check_python_version(
    interpreter: InterpreterInfo,
    *,
    required: tuple[int, ...] = REQUIRED_PYTHON,
) -> CheckResult:
    actual = ".".join(str(part) for part in interpreter.version)
    wanted = ".".join(str(part) for part in required)
    found = tuple(interpreter.version[: len(required)])
    if found == tuple(required):
        return _passed("python", f"Python {actual} matches the locked {wanted}")
    if found > tuple(required):
        return _failed(
            "python",
            f"Python {actual} is newer than the locked {wanted}",
            (
                f"the project is locked to {wanted} exactly, so a newer Python is not an upgrade",
                "requirements.txt pins cp312-only wheels, so pip picks one for this ABI instead,",
                "matches none of the pinned hashes, and aborts with a hash error, not a version error",
                *_PYTHON_INSTALL_HELP,
            ),
        )
    return _failed(
        "python",
        f"Python {actual} is older than the locked {wanted}",
        (
            f"the project is locked to {wanted} exactly, and this interpreter is below it",
            "requirements.txt pins cp312-only wheels and numpy 2.5.2 needs 3.12+, so pip cannot install",
            *_PYTHON_INSTALL_HELP,
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


# One file's verdict against one manifest record. Both a single-file asset and
# every member of a directory asset are decided here, so their outcomes cannot
# drift apart.
VERDICT_OK = "ok"
VERDICT_MISSING = "missing"
VERDICT_UNREADABLE = "unreadable"
VERDICT_SIZE = "size"
VERDICT_MISMATCH = "mismatch"
VERDICT_MD5_MISMATCH = "md5_mismatch"
VERDICT_UNVERIFIED = "unverified"
VERDICT_MALFORMED = "malformed"

_MISSING_FILE_HELP = (
    "./setup.sh downloads model assets and verifies their sha256",
    "weights live under ~/.cache/luxo and are never committed",
)
_MISSING_DIRECTORY_HELP = (
    "./setup.sh stages this directory and verifies every file the manifest requires",
    "browser assets are staged under renderer/public and are never committed",
)
_REDOWNLOAD_HELP = ("delete the file and re-run ./setup.sh to download it again",)


@dataclass(frozen=True, slots=True)
class _FileVerdict:
    outcome: str
    detail: str
    size_bytes: int | None = None
    md5_confirmed: bool = False


def _digest_is_unusable(digest: str, marker: str) -> bool:
    """True when a manifest digest is neither hex nor the declared marker."""

    return not _is_unverified(digest, marker) and not _HEX64.fullmatch(digest.strip().lower())


def _cross_check_md5(resolved: str, expected_md5: str | None, filesystem: Filesystem) -> str | None:
    """Hash md5 only to catch a corrupted download of an unpinnable asset.

    Returns None when there is nothing to compare or the probe cannot answer,
    so this can add a failure but can never invent one.
    """

    if not expected_md5:
        return None
    try:
        return filesystem.md5(resolved).strip().lower()
    except (OSError, ValueError):
        return None


def _verify_file(
    resolved: str,
    expected_sha256: str,
    expected_size: int | None,
    filesystem: Filesystem,
    *,
    unverified_marker: str,
    expected_md5: str | None = None,
) -> _FileVerdict:
    """Check one concrete file against one manifest record.

    The caller turns a verdict into a CheckResult, because the wording of a
    whole asset and of one file inside a directory differ; the decision does
    not.
    """

    expected = expected_sha256.strip().lower()
    unverified = _is_unverified(expected_sha256, unverified_marker)
    if _digest_is_unusable(expected_sha256, unverified_marker):
        return _FileVerdict(
            VERDICT_MALFORMED, "the manifest sha256 is not 64 hexadecimal characters"
        )
    if not filesystem.is_file(resolved):
        return _FileVerdict(VERDICT_MISSING, f"missing: {resolved}")

    try:
        actual_size = filesystem.size_bytes(resolved)
        # An unverified entry has no digest to compare against, so the file is
        # deliberately not hashed here.
        actual = None if unverified else filesystem.sha256(resolved).lower()
    except OSError as exc:
        return _FileVerdict(
            VERDICT_UNREADABLE, f"cannot read {resolved}: {exc.strerror or 'unreadable'}"
        )

    if expected_size is not None and actual_size != expected_size:
        return _FileVerdict(
            VERDICT_SIZE,
            f"size mismatch at {resolved}: expected {expected_size} bytes, found {actual_size}",
            actual_size,
        )
    if unverified:
        published = _cross_check_md5(resolved, expected_md5, filesystem)
        if published is not None and expected_md5 is not None and published != expected_md5:
            return _FileVerdict(
                VERDICT_MD5_MISMATCH,
                (
                    f"md5 mismatch at {resolved}: the publisher records "
                    f"{expected_md5[:16]}..., these bytes are {published[:16]}..."
                ),
                actual_size,
            )
        return _FileVerdict(
            VERDICT_UNVERIFIED,
            f"present at {resolved} ({actual_size} bytes)",
            actual_size,
            md5_confirmed=published is not None,
        )
    if actual != expected:
        return _FileVerdict(
            VERDICT_MISMATCH,
            (
                f"sha256 mismatch at {resolved}: expected {expected[:16]}..., "
                f"found {actual[:16]}..."
            ),
            actual_size,
        )
    return _FileVerdict(
        VERDICT_OK,
        f"{resolved} matches sha256 {expected[:16]}... ({actual_size} bytes)",
        actual_size,
    )


def _resolve_member_path(directory: str, relative: str) -> str | None:
    """Join a required_files path onto its own destination directory.

    None means the manifest path would land outside the destination, which
    breaks the rooting rule config/models.yaml states: a required_files path is
    relative to destination.path, never absolute and never with a '..' segment.
    """

    candidate = os.path.normpath(os.path.join(directory, relative))
    prefix = directory.rstrip(os.sep) + os.sep
    if candidate == directory or not candidate.startswith(prefix):
        return None
    return candidate


def _join_names(names: Sequence[str], limit: int = 6) -> str:
    shown = list(names[:limit])
    remaining = len(names) - len(shown)
    if remaining > 0:
        shown.append(f"+{remaining} more")
    return ", ".join(shown)


def _unrequired_names(
    resolved: str,
    required_files: Sequence[RequiredFile],
    filesystem: Filesystem,
) -> tuple[str, ...]:
    """Name what the directory holds that no required_files record claims.

    Reported, never failed: the manifest says setup.sh may copy a whole upstream
    directory, and records that both wasm bundles ship files the renderer never
    fetches, so an unclaimed file is information rather than an error.
    """

    claimed = {
        required.path.replace(os.sep, "/").split("/", 1)[0]
        for required in required_files
        if required.path
    }
    return tuple(name for name in filesystem.list_dir(resolved) if name not in claimed)


def check_asset(
    entry: AssetEntry,
    filesystem: Filesystem,
    *,
    base_dir: Path = REPO_ROOT,
    unverified_marker: str = DEFAULT_UNVERIFIED_MARKER,
) -> CheckResult:
    """Verify one asset. Missing or corrupt is a FAIL; unpinnable is a WARN.

    Three outcomes stay distinguishable on purpose. A missing file blocks the
    run. A digest that disagrees with the manifest is the corruption and
    supply-chain signal and must stay loud. A file that is present but whose
    manifest entry carries the UNVERIFIED marker instead of a digest is a
    recorded, deliberate gap: it warns, and it does not fail preflight.

    A ``kind: directory`` asset is verified through the files the manifest
    requires inside it, by exactly those rules, one file at a time. Its own
    sha256 is UNVERIFIED by construction -- a directory has no single digest --
    so that marker is not what decides the outcome there; the per-file digests
    are, and a directory whose required files are all present and matching
    passes.
    """

    name = f"asset {entry.name}"
    # Checked before anything is resolved or read, so a manifest typo is
    # reported without touching the disk at all.
    if _digest_is_unusable(entry.sha256, unverified_marker):
        return _failed(
            name,
            "the manifest sha256 is not 64 hexadecimal characters",
            (
                "fix the manifest entry; hashes are lowercase sha256 hex digests",
                "an asset with no published digest is recorded as "
                f"{unverified_marker or DEFAULT_UNVERIFIED_MARKER}",
            ),
        )
    if entry.is_directory:
        return _check_directory_asset(
            entry, filesystem, base_dir=base_dir, unverified_marker=unverified_marker
        )
    return _check_file_asset(
        entry, filesystem, base_dir=base_dir, unverified_marker=unverified_marker
    )


def _check_file_asset(
    entry: AssetEntry,
    filesystem: Filesystem,
    *,
    base_dir: Path,
    unverified_marker: str,
) -> CheckResult:
    name = f"asset {entry.name}"
    resolved = _resolve_asset_path(entry.path, filesystem, base_dir)
    verdict = _verify_file(
        resolved,
        entry.sha256,
        entry.size_bytes,
        filesystem,
        unverified_marker=unverified_marker,
        expected_md5=entry.md5,
    )

    if verdict.outcome == VERDICT_MISSING:
        return _failed(name, verdict.detail, _MISSING_FILE_HELP)
    if verdict.outcome == VERDICT_UNREADABLE:
        return _failed(name, verdict.detail, ("check file permissions, then re-run ./setup.sh",))
    if verdict.outcome == VERDICT_SIZE:
        return _failed(name, verdict.detail, _REDOWNLOAD_HELP)
    if verdict.outcome == VERDICT_MD5_MISMATCH:
        return _failed(
            name,
            verdict.detail,
            (
                "the publisher's own md5 disagrees with these bytes, so the download "
                "is corrupt or is not the file the manifest names",
                *_REDOWNLOAD_HELP,
            ),
        )
    if verdict.outcome == VERDICT_UNVERIFIED:
        detail = (
            f"present at {resolved} ({verdict.size_bytes} bytes) but the manifest records "
            f"sha256 {unverified_marker}, so these bytes cannot be hash-checked"
        )
        if verdict.md5_confirmed:
            detail += (
                "; the publisher's md5 does match, which rules out a corrupted download "
                "but is not a supply-chain pin"
            )
        verify_hint = entry.verify_command or (
            f"hash it yourself, then paste the digest over {unverified_marker} in the manifest"
        )
        return _warned(
            name,
            detail,
            (
                f"verify it by hand: {verify_hint}",
                "the publisher ships no sha256 for this asset; the manifest records "
                "why, and this is a known gap rather than a corrupt download",
            ),
        )
    if verdict.outcome == VERDICT_MISMATCH:
        return _failed(
            name,
            verdict.detail,
            (
                "the download is truncated, corrupt, or the wrong revision",
                *_REDOWNLOAD_HELP,
            ),
        )
    return _passed(name, verdict.detail)


def _check_directory_asset(
    entry: AssetEntry,
    filesystem: Filesystem,
    *,
    base_dir: Path,
    unverified_marker: str,
) -> CheckResult:
    """Verify a directory asset file by file, naming whichever files are wrong."""

    name = f"asset {entry.name}"
    resolved = _resolve_asset_path(entry.path, filesystem, base_dir)
    if not filesystem.is_dir(resolved):
        if filesystem.is_file(resolved):
            return _failed(
                name,
                f"expected a directory at {resolved}, found a file",
                (
                    "this asset is a staged directory of files, not one download",
                    "delete it, then re-run ./setup.sh to stage the directory",
                ),
            )
        return _failed(name, f"missing directory: {resolved}", _MISSING_DIRECTORY_HELP)
    if not entry.required_files:
        return _warned(
            name,
            (
                f"present at {resolved} but the manifest names no required_files, "
                "so nothing inside it can be checked"
            ),
            (
                "list the files this directory must contain under required_files, "
                "each with its own path, size_bytes and sha256",
                "a directory has no single sha256, so required_files is the only "
                "thing that can verify one",
            ),
        )

    failures: list[tuple[str, str]] = []
    unpinned: list[str] = []
    matched = 0
    matched_bytes = 0
    for required in entry.required_files:
        member = _resolve_member_path(resolved, required.path)
        if member is None:
            failures.append(
                (
                    required.path,
                    f"{required.path!r} lands outside {resolved}; a required_files path "
                    "is relative to the destination directory",
                )
            )
            continue
        verdict = _verify_file(
            member,
            required.sha256,
            required.size_bytes,
            filesystem,
            unverified_marker=unverified_marker,
        )
        if verdict.outcome == VERDICT_OK:
            matched += 1
            matched_bytes += verdict.size_bytes or 0
        elif verdict.outcome == VERDICT_UNVERIFIED:
            unpinned.append(required.path)
        elif verdict.outcome == VERDICT_MALFORMED:
            # The only verdict whose wording is about the manifest rather than
            # about a path, so it needs the file named onto it.
            failures.append((required.path, f"{required.path}: {verdict.detail}"))
        else:
            failures.append((required.path, verdict.detail))

    total = len(entry.required_files)
    unrequired = _unrequired_names(resolved, entry.required_files, filesystem)
    extra_note = (
        f"; also present, not required: {_join_names(unrequired)}" if unrequired else ""
    )
    verify_hint = entry.verify_command or (
        "hash each required file and compare it against required_files in the manifest"
    )

    if failures:
        return _failed(
            name,
            (
                f"{resolved}: {len(failures)} of {total} required files did not verify "
                f"({_join_names([path for path, _ in failures])}){extra_note}"
            ),
            (
                *(detail for _, detail in failures),
                "re-run ./setup.sh to stage this directory again; it copies exactly the "
                "files the manifest requires",
                f"check by hand: {verify_hint}",
            ),
        )
    if unpinned:
        return _warned(
            name,
            (
                f"{resolved}: {matched} of {total} required files match their recorded "
                f"sha256, {len(unpinned)} cannot be hash-checked "
                f"({_join_names(unpinned)}){extra_note}"
            ),
            (
                f"verify those by hand: {verify_hint}",
                f"their manifest records sha256 {unverified_marker} rather than a digest; "
                "that is a known gap, not a corrupt download",
            ),
        )
    return _passed(
        name,
        (
            f"{resolved}: all {total} required files match their recorded sha256 "
            f"({matched_bytes} bytes){extra_note}"
        ),
    )


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
    marker = manifest_unverified_marker(document)
    return [
        check_asset(entry, filesystem, base_dir=base_dir, unverified_marker=marker)
        for entry in entries
    ]


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
            f"export {variable} in the shell that launches the core, before ./run.sh",
            "nothing loads it for you today: run.sh does not read the root .env, so a "
            "key that lives only in that file will not reach the core",
            "the doctor never opens .env, and never prints, logs, or transmits the value",
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
    warned = sum(1 for result in results if result.status is Status.WARN)
    lines.append("")
    lines.append(f"{passed} passed, {failed} failed, {skipped} skipped, {warned} warned")
    if failed:
        lines.append("preflight failed; fix the items above before running ./run.sh")
    elif warned:
        lines.append(
            "core preflight passed with warnings; they do not block ./run.sh. "
            "Run the browser /selftest page next"
        )
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
