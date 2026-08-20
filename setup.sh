#!/usr/bin/env bash
#
# Luxo setup — clean Ubuntu 24.04 LTS (4 cores, 8 GB, no GPU) is the target;
# macOS is supported as the development platform and degrades where apt cannot
# apply.  PRD 12.1 and 12.3.
#
# What this script does, in order:
#
#   1. system packages   build-essential cmake python3-venv espeak-ng (+ the
#                        tools the later steps need: git, curl, ca-certificates)
#   2. node              a Node that actually satisfies Vite 8 — see ensure_node
#   3. python            a MANDATORY venv at .venv (PEP 668 marks Noble's system
#                        Python externally managed, so system pip is blocked)
#   4. whisper.cpp       built from source at a pinned commit, CPU-only flags
#   5. model assets      parsed out of config/models.yaml, downloaded to
#                        ~/.cache/luxo and renderer/public, SHA-256 verified
#   6. renderer          npm ci, stage the browser-side assets, vite build
#
# What this script deliberately never does:
#
#   * It never reads, prints, copies, or commits the root .env or any API key.
#     Loading .env is run.sh's job.  There is not one reference to it below.
#   * It never commits model weights.  Core weights live in ~/.cache/luxo,
#     outside the repo entirely.  Browser weights are staged into
#     renderer/public and are checked against .gitignore at the end of the run
#     (see check_ignored) rather than assumed to be covered.
#   * It never binds anything to a network interface.  PRD 12.3: localhost is a
#     secure context, so no TLS and no LAN bind — and setup starts no server.
#   * It never writes a SHA-256 into config/models.yaml.  Four of the seven
#     assets are honestly marked UNVERIFIED; this script prints what it computed
#     and leaves the manifest to the owner.
#
# Safe to re-run.  Downloads, the whisper.cpp build, and the venv are all
# skipped when they are already correct.
#
# Environment overrides (all optional):
#   LUXO_PYTHON            path to the CPython 3.12 interpreter to build .venv
#   LUXO_NODE_MAJOR        Node major line to install on Linux (default 22)
#   LUXO_WHISPER_REF       whisper.cpp tag to build (default below)
#   LUXO_WHISPER_COMMIT    commit that tag must resolve to; "" disables the check
#   LUXO_SKIP_APT=1        skip the system-package step entirely
#   LUXO_FORCE_DOWNLOAD=1  re-download every asset even if it already verifies
#   LUXO_GGML_NATIVE=OFF   build whisper.cpp without -march=native

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

# whisper.cpp.  The tag is what we fetch; the commit is what we assert we got,
# which makes a mutable tag behave like an immutable pin without needing the
# server to support fetching a bare SHA.
whisper_ref="${LUXO_WHISPER_REF:-v1.9.2}"
whisper_commit="${LUXO_WHISPER_COMMIT-306c88f4d1286aec1bf96e544632897886af5501}"
whisper_repo="https://github.com/ggml-org/whisper.cpp.git"

# Vite 8 declares engines.node "^20.19.0 || >=22.12.0" (renderer/package-lock.json).
# Ubuntu 24.04's apt nodejs is 18.x and does NOT satisfy that — see ensure_node.
node_major="${LUXO_NODE_MAJOR:-22}"
nodesource_key_url="https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key"
nodesource_keyring="/usr/share/keyrings/nodesource.gpg"
nodesource_list="/etc/apt/sources.list.d/nodesource.list"

# requirements.txt pins onnxruntime 1.20.1 by hash, and those two hashes are the
# cp312 macOS universal2 wheel and the cp312 manylinux x86_64 wheel.  A 3.13 or
# 3.11 venv therefore cannot satisfy the pin at all — pip would fail with an
# opaque "no matching distribution" instead of an explanation.  Require 3.12.
python_series="3.12"

manifest_path="$repo_dir/config/models.yaml"
venv_dir="$repo_dir/.venv"
cache_dir="${HOME}/.cache/luxo"
whisper_src_dir="$cache_dir/whisper.cpp"
whisper_bin_dir="$cache_dir/bin"
whisper_bin="$whisper_bin_dir/whisper-cli"
renderer_dir="$repo_dir/renderer"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

work_dir=""
notes_file=""
step_number=0

cleanup() {
  if [[ -n "$work_dir" && -d "$work_dir" ]]; then
    rm -rf -- "$work_dir"
  fi
}

on_error() {
  local code="$1" line="$2"
  printf '\nsetup failed at line %s (exit %s)\n' "$line" "$code" >&2
  printf 'nothing was rolled back; this script is safe to re-run once fixed\n' >&2
}

trap cleanup EXIT
trap 'on_error "$?" "$LINENO"' ERR

step() {
  step_number=$((step_number + 1))
  printf '\n==> [%d] %s\n' "$step_number" "$*"
}

info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }

die() {
  printf '\nsetup: %s\n' "$*" >&2
  exit 1
}

# A note is something the owner must read after a successful run: an unpinned
# digest, an unhashed dependency, a path .gitignore does not cover.  Collected
# in a file rather than an array because macOS ships bash 3.2, where expanding
# an empty array under `set -u` is an error.
note() { printf '%s\n' "$*" >>"$notes_file"; }

# ---------------------------------------------------------------------------
# Small platform helpers
# ---------------------------------------------------------------------------

os_name="$(uname -s)"
sudo_cmd=""

is_linux() { [[ "$os_name" == "Linux" ]]; }
is_macos() { [[ "$os_name" == "Darwin" ]]; }

have() { command -v "$1" >/dev/null 2>&1; }

cpu_count() {
  if have nproc; then
    nproc
  elif have sysctl; then
    sysctl -n hw.ncpu 2>/dev/null || echo 2
  else
    echo 2
  fi
}

resolve_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    sudo_cmd=""
  elif have sudo; then
    sudo_cmd="sudo"
  else
    die "system packages need root and sudo is not installed; re-run as root"
  fi
}

# Digest helpers.  sha256sum on Linux, shasum on macOS, python as the backstop.
sha256_of() {
  if have sha256sum; then
    sha256sum "$1" | cut -d' ' -f1
  elif have shasum; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    "$python_bin" -c 'import hashlib,sys;h=hashlib.sha256()
with open(sys.argv[1],"rb") as f:
    for b in iter(lambda: f.read(1<<16), b""): h.update(b)
print(h.hexdigest())' "$1"
  fi
}

md5_of() {
  if have md5sum; then
    md5sum "$1" | cut -d' ' -f1
  elif have md5; then
    md5 -q "$1"
  else
    "$python_bin" -c 'import hashlib,sys;h=hashlib.md5()
with open(sys.argv[1],"rb") as f:
    for b in iter(lambda: f.read(1<<16), b""): h.update(b)
print(h.hexdigest())' "$1"
  fi
}

size_of() { wc -c <"$1" | tr -d ' '; }

download() {
  local url="$1" dest="$2"
  if have curl; then
    # --progress-bar rather than the default meter: these are 60 MB downloads
    # and a single bar is readable in a setup log. Errors still print.
    curl -fL --progress-bar --retry 3 --retry-delay 2 --connect-timeout 20 -o "$dest" "$url"
  elif have wget; then
    wget -q -O "$dest" "$url"
  else
    die "neither curl nor wget is available to download $url"
  fi
}

# renderer/public holds staged weights.  .gitignore has a bare "models/" rule,
# which covers renderer/public/models/ but NOT renderer/public/mediapipe/ or
# renderer/public/onnxruntime/.  This script does not own .gitignore, so it
# checks each staged path against git rather than claiming coverage it cannot
# see, and reports anything git would happily commit.
check_ignored() {
  local path="$1"
  have git || return 0
  git -C "$repo_dir" rev-parse --git-dir >/dev/null 2>&1 || return 0
  if ! git -C "$repo_dir" check-ignore -q "$path" 2>/dev/null; then
    note "NOT GITIGNORED: ${path#"$repo_dir"/} — staged weights must never be committed; add the directory to .gitignore"
  fi
}

# ---------------------------------------------------------------------------
# Step 1 — system packages
# ---------------------------------------------------------------------------

# PRD 12.1 names: build-essential cmake python3-venv nodejs npm.
#
# nodejs and npm are handled by ensure_node instead of being listed here.
# Installing apt's nodejs (18.x on Noble) only to replace it minutes later
# wastes a download and creates a package conflict with the Node we actually
# need.  See ensure_node for the full reasoning.
#
# git, curl and ca-certificates are added because PRD 12.1 itself requires
# cloning whisper.cpp from source and downloading model weights over HTTPS;
# they are not optional extras.  espeak-ng is the Piper phonemiser and was
# approved by the owner as a system dependency (doctor.py checks for it).
apt_packages="build-essential cmake git curl ca-certificates gnupg python3-venv python3-dev espeak-ng"

apt_missing() {
  local pkg missing=""
  # shellcheck disable=SC2086  # apt_packages is a deliberate whitespace list
  for pkg in $apt_packages; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "^install ok installed$"; then
      missing="$missing $pkg"
    fi
  done
  printf '%s' "${missing# }"
}

install_system_packages_linux() {
  local missing
  missing="$(apt_missing)"
  if [[ -z "$missing" ]]; then
    info "all apt packages already installed"
    return 0
  fi
  # shellcheck disable=SC2086
  info "installing:$(printf ' %s' $missing)"
  # shellcheck disable=SC2086  # sudo_cmd is deliberately unquoted: empty as root
  DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get update
  # shellcheck disable=SC2086
  DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get install -y $missing
}

install_system_packages_macos() {
  # No apt on macOS.  Xcode command line tools supply the compiler, so the only
  # thing that must be installed is espeak-ng, which the owner approved.
  info "macOS: skipping apt; expecting Xcode command line tools for the toolchain"
  have cc || die "no C compiler; run: xcode-select --install"
  have cmake || die "cmake is required; run: brew install cmake"

  if have espeak-ng; then
    info "espeak-ng already present"
    return 0
  fi
  have brew || die "espeak-ng is missing and Homebrew is not installed; install brew, then: brew install espeak-ng"
  info "installing espeak-ng via Homebrew"
  brew install espeak-ng
}

install_system_packages() {
  if [[ "${LUXO_SKIP_APT:-0}" == "1" ]]; then
    info "LUXO_SKIP_APT=1, skipping system packages"
    return 0
  fi
  if is_linux; then
    have apt-get || die "this script's Linux path targets Ubuntu 24.04 and needs apt-get"
    resolve_sudo
    install_system_packages_linux
  elif is_macos; then
    install_system_packages_macos
  else
    die "unsupported platform '$os_name'; the deploy target is Ubuntu 24.04 and dev is macOS"
  fi
}

# ---------------------------------------------------------------------------
# Step 2 — Node
# ---------------------------------------------------------------------------

# renderer/ builds with Vite 8, whose engines field is "^20.19.0 || >=22.12.0".
# Ubuntu 24.04 ships nodejs 18.19.1, which does not satisfy either branch, so
# `apt install nodejs npm` produces a box that cannot build the renderer at all.
# The failure surfaces deep inside vite as an unrelated-looking syntax or API
# error, which is exactly the confusing outcome to avoid.
#
# Mechanism chosen: NodeSource's apt repository, pinned to the 22.x line, added
# as a signed keyring plus an explicit sources.list entry.
#
# Why NodeSource and not nvm:
#   * nvm is per-user and lives in a shell rc file.  A non-interactive shell —
#     which is how run.sh, a service unit, cron, or an IDE terminal invokes
#     things — does not source ~/.bashrc, so `node` silently vanishes and
#     run.sh dies on "npm is required to start the renderer".  A deploy box
#     wants node on PATH for every process, not for one login shell.
#   * NodeSource installs to /usr/bin through apt, so it is visible everywhere,
#     upgradable with the rest of the system, and inspectable with dpkg.
# Cost, stated plainly: it needs root and it trusts a third-party apt repo.
# That is acceptable on a dedicated deploy box and is the mechanism NodeSource
# documents.  We add the repo by hand (keyring + sources.list) rather than
# piping their install script into a root shell, so what gets added is
# inspectable and every package is signature-verified by apt.

node_version_string() {
  node -v 2>/dev/null | sed 's/^v//'
}

# Vite 8: ^20.19.0 || >=22.12.0.  Written out branch by branch on purpose —
# note that 21.x and 22.0-22.11 both fail, which a naive ">= 20" check misses.
node_satisfies() {
  local version="$1" major minor
  [[ -n "$version" ]] || return 1
  major="${version%%.*}"
  minor="${version#*.}"
  minor="${minor%%.*}"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 1

  if [[ "$major" -eq 20 && "$minor" -ge 19 ]]; then return 0; fi
  if [[ "$major" -gt 22 ]]; then return 0; fi
  if [[ "$major" -eq 22 && "$minor" -ge 12 ]]; then return 0; fi
  return 1
}

install_node_nodesource() {
  local arch
  resolve_sudo
  arch="$(dpkg --print-architecture)"

  # Ubuntu's own nodejs/npm packages conflict with NodeSource's nodejs, which
  # bundles its own npm.  Remove them first, loudly, rather than letting apt
  # abort halfway through with a file-conflict error.
  local conflicting="" pkg
  for pkg in npm libnode-dev; do
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "^install ok installed$"; then
      conflicting="$conflicting $pkg"
    fi
  done
  if [[ -n "$conflicting" ]]; then
    info "removing distro packages that conflict with NodeSource:$conflicting"
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get remove -y $conflicting
  fi

  info "adding the NodeSource ${node_major}.x apt repository"
  # --batch --yes so a re-run overwrites the keyring instead of prompting.
  curl -fsSL "$nodesource_key_url" | $sudo_cmd gpg --batch --yes --dearmor -o "$nodesource_keyring"
  printf 'deb [arch=%s signed-by=%s] https://deb.nodesource.com/node_%s.x nodistro main\n' \
    "$arch" "$nodesource_keyring" "$node_major" \
    | $sudo_cmd tee "$nodesource_list" >/dev/null

  # shellcheck disable=SC2086
  DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get update
  # shellcheck disable=SC2086
  DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get install -y nodejs
}

ensure_node() {
  local version
  version="$(node_version_string || true)"

  if [[ -n "$version" ]] && node_satisfies "$version"; then
    info "node $version satisfies Vite 8's ^20.19 || >=22.12"
  else
    if [[ -n "$version" ]]; then
      warn "node $version does not satisfy Vite 8's ^20.19 || >=22.12"
    else
      info "node is not installed"
    fi

    if is_linux; then
      install_node_nodesource
    else
      # A dev Mac's node is the developer's business, and Homebrew's node@22 is
      # keg-only (installing it does not change what `node` resolves to), so a
      # silent brew install would look like it worked and change nothing.  Say
      # exactly what to run and stop.
      die "node ${version:-(absent)} cannot build Vite 8.
    Install a satisfying Node, for example:
      brew install node          # current line, >= 22.12
      # or, to keep other projects on their own version:
      brew install nvm && nvm install 22 && nvm use 22
    Then re-run ./setup.sh"
    fi

    hash -r 2>/dev/null || true
    version="$(node_version_string || true)"
    node_satisfies "$version" || die "installed node reports '${version:-nothing}', which still does not satisfy Vite 8's ^20.19 || >=22.12; install Node ${node_major}.x manually and re-run"
    info "node $version installed and verified"
  fi

  have npm || die "npm is missing even though node $version is present; install the npm that ships with Node ${node_major}.x"
  info "npm $(npm --version)"
}

# ---------------------------------------------------------------------------
# Step 3 — Python venv
# ---------------------------------------------------------------------------

python_bin=""

interpreter_series() {
  "$1" -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

select_python() {
  local candidate series
  for candidate in "${LUXO_PYTHON:-}" "python${python_series}" python3; do
    [[ -n "$candidate" ]] || continue
    have "$candidate" || continue
    series="$(interpreter_series "$candidate")"
    if [[ "$series" == "$python_series" ]]; then
      python_bin="$(command -v "$candidate")"
      return 0
    fi
  done

  die "no CPython $python_series interpreter found.
    Python $python_series is not a floor here, it is exact: requirements.txt pins
    onnxruntime 1.20.1 by hash and the two hashes are its cp312 wheels, so any
    other series has nothing to match and pip fails obscurely.
      sudo apt-get install -y python${python_series} python${python_series}-venv   # Ubuntu 24.04 ships it
      brew install python@${python_series}                                        # macOS
    Or point at one directly: LUXO_PYTHON=/path/to/python${python_series} ./setup.sh"
}

create_venv() {
  # PEP 668: Ubuntu 24.04 marks the system Python externally managed, so pip
  # into it is blocked outright.  The venv is not a convenience here.
  if [[ -x "$venv_dir/bin/python" ]]; then
    local series
    series="$(interpreter_series "$venv_dir/bin/python")"
    if [[ "$series" == "$python_series" ]]; then
      info "reusing the existing venv at $venv_dir (Python $series)"
      return 0
    fi
    die "the existing venv at $venv_dir is Python ${series:-unknown}, not $python_series; remove it and re-run: rm -rf '$venv_dir'"
  fi

  info "creating the venv at $venv_dir with $python_bin"
  "$python_bin" -m venv "$venv_dir" \
    || die "python -m venv failed; on Ubuntu install it with: sudo apt-get install -y python${python_series}-venv"
}

install_python_requirements() {
  local venv_python="$venv_dir/bin/python"

  # pip is used as the venv ships it.  Upgrading pip first would mean an
  # unpinned, unhashed network install ahead of a deliberately hash-pinned one,
  # which undercuts the point; Noble's bundled pip 24.0 checks hashes fine.

  # requirements.txt carries --hash lines, so pip switches itself into
  # hash-checking mode automatically.  Try the strict path first: if the
  # manifest of hashes is ever completed, this branch simply starts succeeding
  # and the fallback below never runs again.
  info "installing pinned requirements (hash-checked)"
  if "$venv_python" -m pip install --require-hashes -r "$repo_dir/requirements.txt"; then
    info "all requirements installed with verified hashes"
    return 0
  fi

  # Why the strict path fails today: pip's hash-checking mode requires a hash
  # for *every* distribution it installs, transitive ones included.
  # requirements.txt hashes onnxruntime and websockets but not onnxruntime's six
  # dependencies (coloredlogs, flatbuffers, numpy, packaging, protobuf, sympy),
  # which carry no hashes anywhere in the repo.
  #
  # This script does not own requirements.txt and will not invent hashes for it.
  # It splits the install instead, and says out loud which half is unverified.
  warn "hash-checked install of the full dependency tree failed"
  info "falling back: hash-verify the pinned distributions, then resolve their dependencies"

  # Phase 1 — still hash-checked (pip infers the mode from the file). --no-deps
  # means the only things installed are the two distributions that carry pins,
  # so a corrupt or substituted onnxruntime/websockets still fails hard here.
  "$venv_python" -m pip install --no-deps -r "$repo_dir/requirements.txt" \
    || die "the pinned distributions themselves failed to install or failed hash verification; this is a real integrity failure, not the transitive-dependency gap"

  # Phase 2 — the same requirement lines with the --hash tokens removed, so pip
  # will resolve the transitive dependencies.  Derived from requirements.txt at
  # runtime into a temp file; requirements.txt is never modified.
  local stripped="$work_dir/requirements-no-hashes.txt"
  "$python_bin" "$work_dir/manifest.py" strip-hashes "$repo_dir/requirements.txt" >"$stripped"
  "$venv_python" -m pip install -r "$stripped" \
    || die "could not resolve the transitive dependencies of the pinned requirements"

  note "UNHASHED DEPENDENCIES: requirements.txt hashes onnxruntime and websockets but not their transitive dependencies, so pip's hash-checking mode cannot cover the whole tree. Those dependencies were installed unverified. To close this, regenerate the file with hashes for everything (pip-compile --generate-hashes, or pip hash on each wheel) — setup.sh will then take the strict path automatically."
}

# ---------------------------------------------------------------------------
# Step 4 — whisper.cpp
# ---------------------------------------------------------------------------

build_whisper() {
  if [[ -x "$whisper_bin" && -d "$whisper_src_dir/.git" ]]; then
    local head
    head="$(git -C "$whisper_src_dir" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "$whisper_commit" && "$head" == "$whisper_commit" ]]; then
      info "whisper-cli already built from $whisper_ref ($head)"
      return 0
    fi
    if [[ -z "$whisper_commit" && -n "$head" ]]; then
      info "whisper-cli already built from $head (commit assertion disabled)"
      return 0
    fi
    info "rebuilding: the checkout is at ${head:-unknown}, not the pinned commit"
  fi

  mkdir -p "$whisper_src_dir"
  if [[ ! -d "$whisper_src_dir/.git" ]]; then
    info "cloning whisper.cpp $whisper_ref"
    git clone --depth 1 --branch "$whisper_ref" "$whisper_repo" "$whisper_src_dir"
  else
    info "fetching whisper.cpp $whisper_ref"
    git -C "$whisper_src_dir" fetch --depth 1 origin "refs/tags/$whisper_ref:refs/tags/$whisper_ref" --force
    git -C "$whisper_src_dir" checkout --detach "refs/tags/$whisper_ref"
  fi

  # A tag can be moved upstream.  Assert the commit it resolved to, so the pin
  # is really a pin and a moved tag fails here rather than shipping quietly.
  if [[ -n "$whisper_commit" ]]; then
    local head
    head="$(git -C "$whisper_src_dir" rev-parse HEAD)"
    [[ "$head" == "$whisper_commit" ]] \
      || die "whisper.cpp $whisper_ref resolved to $head but the pin expects $whisper_commit;
    the tag moved, or the pin is stale. Verify upstream, then set LUXO_WHISPER_COMMIT
    (or LUXO_WHISPER_COMMIT= to disable the assertion) before re-running."
  fi

  # CPU-only build.  PRD 12.3: provide a CPU-only flag path and do not assume
  # the macOS accelerators exist on Linux.
  #   BUILD_SHARED_LIBS=OFF  so whisper-cli can be copied out of the build tree
  #                          without dragging libwhisper.so along behind it
  #   GGML_CUDA / VULKAN     off: the deploy box has no GPU
  #   GGML_BLAS              off: no external BLAS is installed and none is needed
  #   GGML_NATIVE            on: this builds on the machine that will run it.
  #                          Set LUXO_GGML_NATIVE=OFF when building elsewhere,
  #                          or the binary can die on an illegal instruction.
  local -a cmake_flags
  cmake_flags=(
    -S "$whisper_src_dir"
    -B "$whisper_src_dir/build"
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_NATIVE="${LUXO_GGML_NATIVE:-ON}"
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DGGML_BLAS=OFF
    -DWHISPER_BUILD_TESTS=OFF
    -DWHISPER_BUILD_SERVER=OFF
    -DWHISPER_BUILD_EXAMPLES=ON
    -DWHISPER_SDL2=OFF
    -DWHISPER_CURL=OFF
  )

  if is_macos; then
    # Accelerate is fine to keep on the dev Mac and is what the measurement
    # notes assume.  Metal is turned off so the mac build stays CPU-shaped like
    # the deploy target, and so the copied binary has no GPU-side dependency.
    cmake_flags+=(-DGGML_METAL=OFF -DGGML_ACCELERATE=ON -DWHISPER_COREML=OFF)
  else
    cmake_flags+=(-DGGML_METAL=OFF -DGGML_ACCELERATE=OFF)
  fi

  info "configuring whisper.cpp (CPU-only)"
  cmake "${cmake_flags[@]}"
  info "building whisper.cpp with $(cpu_count) jobs"
  cmake --build "$whisper_src_dir/build" --config Release --target whisper-cli -j "$(cpu_count)"

  local built="$whisper_src_dir/build/bin/whisper-cli"
  [[ -x "$built" ]] || die "the build finished but $built is missing; inspect the cmake output above"

  mkdir -p "$whisper_bin_dir"
  install -m 0755 "$built" "$whisper_bin"
  info "installed $whisper_bin"
}

# ---------------------------------------------------------------------------
# Step 5 — model assets from config/models.yaml
# ---------------------------------------------------------------------------

# The manifest is the authority for every URL, digest and destination.  Nothing
# below hardcodes any of them.  It is JSON-compatible YAML for the same reason
# config/default.yaml is — JSON is a strict YAML subset, so no YAML dependency
# is needed — which is exactly how core/config.py and doctor.py read it.
write_manifest_reader() {
  cat >"$work_dir/manifest.py" <<'PYTHON'
"""Read config/models.yaml (JSON-compatible YAML) and emit TSV rows for setup.sh.

Modes:
  marker <manifest>            print the unverified marker
  files <manifest> <repo>      one row per kind=file asset:
                               name, dest, url, sha256, size_bytes, md5
  dirfiles <manifest> <repo>   one row per required file of a kind=directory
                               asset: name, package, destdir, relpath, sha256,
                               size_bytes
  strip-hashes <requirements>  the requirement lines with --hash tokens removed
"""

import json
import os
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve(path, repo):
    """Same rule doctor.py uses: expand ~, then anchor relatives at the repo."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(repo, expanded))


def destination_of(asset, document, repo):
    destination = asset["destination"]
    layout = destination["layout"]
    roots = document["destinations"]
    if layout not in roots:
        raise SystemExit(f"{asset['name']}: unknown destination layout {layout!r}")
    root = resolve(roots[layout]["root"], repo)
    target = resolve(destination["path"], repo)
    # Guard a malformed manifest: a path must sit under the root its layout names.
    if not (target == root or target.startswith(root + os.sep)):
        raise SystemExit(
            f"{asset['name']}: destination {target} is outside the {layout} root {root}"
        )
    return target


def row(*fields):
    print("\t".join("" if field is None else str(field) for field in fields))


def main(argv):
    if len(argv) < 3:
        raise SystemExit("usage: manifest.py MODE PATH [REPO]")
    mode, path = argv[1], argv[2]

    if mode == "strip-hashes":
        with open(path, encoding="utf-8") as handle:
            joined = handle.read().replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.split("#", 1)[0]
            kept = [tok for tok in line.split() if not tok.startswith("--hash=")]
            if kept:
                print(" ".join(kept))
        return

    document = load(path)
    for key in ("manifest_version", "unverified_marker", "destinations", "assets"):
        if key not in document:
            raise SystemExit(f"{path}: manifest is missing the {key!r} key")

    if mode == "marker":
        print(document["unverified_marker"])
        return

    repo = argv[3]
    for asset in document["assets"]:
        kind = asset.get("kind")
        if mode == "files" and kind == "file":
            digests = asset.get("additional_digests") or {}
            row(
                asset["name"],
                destination_of(asset, document, repo),
                asset["url"],
                asset["sha256"],
                asset.get("size_bytes", ""),
                digests.get("md5", ""),
            )
        elif mode == "dirfiles" and kind == "directory":
            npm = asset.get("npm") or {}
            package = npm.get("package")
            if not package:
                raise SystemExit(
                    f"{asset['name']}: directory assets are staged from node_modules "
                    "and need an npm.package field"
                )
            destdir = destination_of(asset, document, repo)
            for required in asset.get("required_files") or []:
                # required_files paths are destination-relative and are used
                # verbatim.  No leading segment is ever stripped.
                row(
                    asset["name"],
                    package,
                    destdir,
                    required["path"],
                    required.get("sha256", ""),
                    required.get("size_bytes", ""),
                )


main(sys.argv)
PYTHON
}

unverified_marker=""

# Report a digest we computed for an asset the manifest does not pin, in a form
# the owner can paste straight into config/models.yaml.  Never written back
# automatically: an unverified digest that this script invented and then
# "verified" against itself would be worth nothing.
report_unpinned_digest() {
  local name="$1" path="$2" digest="$3"
  info "$name: sha256 is $unverified_marker in the manifest — NOT a supply-chain pin"
  info "  computed: $digest"
  note "UNPINNED: $name ($path)
      computed sha256: $digest
      The manifest records $unverified_marker for this asset with a documented reason.
      Nothing verified these bytes against a publisher digest. Review the source,
      then paste the line below into config/models.yaml to pin it:
        \"sha256\": \"$digest\","
}

fetch_file_asset() {
  local name="$1" dest="$2" url="$3" want_sha="$4" want_size="$5" want_md5="$6"
  local dest_dir tmp actual_sha actual_size actual_md5

  dest_dir="$(dirname -- "$dest")"
  mkdir -p "$dest_dir"

  # Idempotence: a file already on disk that matches its pin is left alone.
  if [[ -f "$dest" && "${LUXO_FORCE_DOWNLOAD:-0}" != "1" ]]; then
    actual_size="$(size_of "$dest")"
    actual_sha="$(sha256_of "$dest")"
    if [[ "$want_sha" != "$unverified_marker" ]]; then
      if [[ "$actual_sha" == "$want_sha" ]]; then
        info "$name: present and verified"
        return 0
      fi
      die "$name: $dest is on disk but its sha256 does not match the manifest
    expected $want_sha
    found    $actual_sha
    Delete the file and re-run, or set LUXO_FORCE_DOWNLOAD=1."
    fi
    if [[ -z "$want_size" || "$actual_size" == "$want_size" ]]; then
      info "$name: present (size matches; digest unpinned)"
      report_unpinned_digest "$name" "$dest" "$actual_sha"
      return 0
    fi
    warn "$name: on-disk size $actual_size does not match the manifest's $want_size; re-downloading"
  fi

  info "$name: downloading"
  tmp="$dest.part"
  rm -f -- "$tmp"
  download "$url" "$tmp" || die "$name: download failed from $url"

  actual_size="$(size_of "$tmp")"
  if [[ -n "$want_size" && "$actual_size" != "$want_size" ]]; then
    rm -f -- "$tmp"
    die "$name: downloaded $actual_size bytes, manifest says $want_size; the download is truncated or the URL moved"
  fi

  actual_sha="$(sha256_of "$tmp")"

  if [[ "$want_sha" != "$unverified_marker" ]]; then
    # A real pin. Any mismatch is fatal, always.
    if [[ "$actual_sha" != "$want_sha" ]]; then
      rm -f -- "$tmp"
      die "$name: SHA-256 MISMATCH — refusing to install these bytes
    expected $want_sha
    found    $actual_sha
    The download is corrupt, truncated, or the wrong revision. Do not edit the
    manifest to make this pass; find out why the bytes changed."
    fi
    mv -f -- "$tmp" "$dest"
    info "$name: verified sha256 ${actual_sha:0:16}... ($actual_size bytes)"
    return 0
  fi

  # Unpinned. The manifest may still carry a publisher digest in another
  # algorithm; MediaPipe publishes MD5 via x-goog-hash. MD5 is not
  # collision-resistant and is not the supply-chain pin, but it does catch a
  # corrupted or truncated download, so a mismatch is still fatal.
  if [[ -n "$want_md5" ]]; then
    actual_md5="$(md5_of "$tmp")"
    if [[ "$actual_md5" != "$want_md5" ]]; then
      rm -f -- "$tmp"
      die "$name: publisher MD5 mismatch (expected $want_md5, found $actual_md5); the download is corrupt"
    fi
    info "$name: publisher MD5 matches (integrity check only, not a pin)"
  fi

  mv -f -- "$tmp" "$dest"
  report_unpinned_digest "$name" "$dest" "$actual_sha"
}

fetch_file_assets() {
  local rows="$work_dir/files.tsv"
  "$python_bin" "$work_dir/manifest.py" files "$manifest_path" "$repo_dir" >"$rows"

  local name dest url sha size md5
  while IFS=$'\t' read -r name dest url sha size md5; do
    [[ -n "$name" ]] || continue
    fetch_file_asset "$name" "$dest" "$url" "$sha" "$size" "$md5"
    case "$dest" in
      "$repo_dir"/*) check_ignored "$dest" ;;
    esac
  done <"$rows"
}

# ---------------------------------------------------------------------------
# Step 6 — renderer
# ---------------------------------------------------------------------------

# Directory assets are not downloaded.  renderer/package.json already pins
# @mediapipe/tasks-vision and onnxruntime-web, and `npm ci` verifies their
# tarball integrity against package-lock.json — that check is the real
# supply-chain pin, which is why those manifest entries carry UNVERIFIED at the
# top level.  We copy the required files out of node_modules afterwards.
#
# The manifest's per-file digests are labelled MIRROR: they were read from a CDN
# index of the npm tarball, not from the publisher.  The manifest says so and
# says what to do on disagreement — "trust npm ci and investigate rather than
# editing this file" — so a mismatch here is reported loudly and carried into
# the summary, but it does not override npm's own verified install.
stage_directory_assets() {
  local rows="$work_dir/dirfiles.tsv"
  "$python_bin" "$work_dir/manifest.py" dirfiles "$manifest_path" "$repo_dir" >"$rows"

  local name package destdir relpath sha size
  local pkgdir source candidate root found actual_sha actual_size dest
  while IFS=$'\t' read -r name package destdir relpath sha size; do
    [[ -n "$name" ]] || continue

    pkgdir="$renderer_dir/node_modules/$package"
    [[ -d "$pkgdir" ]] || die "$name: $pkgdir is missing; npm ci did not install $package"

    # relpath is destination-relative and is used verbatim for the destination.
    # Inside the package the same file may live at the package root or under a
    # conventional subdirectory, so try those roots in order.  Nothing is
    # stripped from relpath.
    found=""
    for root in "" "wasm" "dist"; do
      if [[ -z "$root" ]]; then
        candidate="$pkgdir/$relpath"
      else
        candidate="$pkgdir/$root/$relpath"
      fi
      if [[ -f "$candidate" ]]; then
        found="$candidate"
        break
      fi
    done
    [[ -n "$found" ]] || die "$name: cannot find '$relpath' in $pkgdir (looked in the package root, wasm/ and dist/)"
    source="$found"

    case "$relpath" in
      */*)
        note "MANIFEST SHAPE: $name lists required file '$relpath' with a directory component. required_files paths are destination-relative, so this stages to $destdir/$relpath — check that against the entry's served_at."
        ;;
    esac

    dest="$destdir/$relpath"
    mkdir -p "$(dirname -- "$dest")"
    cp -f -- "$source" "$dest"

    actual_size="$(size_of "$dest")"
    if [[ -n "$size" && "$actual_size" != "$size" ]]; then
      warn "$name/$relpath: size $actual_size, manifest says $size"
      note "MIRROR SIZE MISMATCH: $name/$relpath is $actual_size bytes, manifest says $size. npm ci verified the tarball; investigate before editing config/models.yaml."
    fi
    if [[ -n "$sha" && "$sha" != "$unverified_marker" ]]; then
      actual_sha="$(sha256_of "$dest")"
      if [[ "$actual_sha" == "$sha" ]]; then
        info "$name/$relpath: staged, matches the mirror digest"
      else
        warn "$name/$relpath: sha256 $actual_sha does not match the manifest's mirror digest $sha"
        note "MIRROR DIGEST MISMATCH: $name/$relpath
      staged   $actual_sha
      manifest $sha
      These digests come from a CDN index of the npm tarball, not the publisher.
      npm ci already verified the tarball against renderer/package-lock.json, so
      the manifest says to trust npm ci and investigate rather than edit the file.
      Do that before shipping."
      fi
    else
      info "$name/$relpath: staged (no mirror digest recorded)"
    fi

    check_ignored "$dest"
  done <"$rows"
}

build_renderer() {
  [[ -d "$renderer_dir" ]] || die "renderer/ is missing"
  [[ -f "$renderer_dir/package-lock.json" ]] || die "renderer/package-lock.json is missing; npm ci needs it"

  info "npm ci in renderer/"
  npm ci --prefix "$renderer_dir"

  step "Staging browser-side assets from node_modules"
  stage_directory_assets

  # `npm run build` is the project's own entry point: tsc --noEmit && vite build.
  # Nothing here starts a server or binds a port.
  step "Building the renderer"
  npm run build --prefix "$renderer_dir"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
  printf '\n'
  printf 'setup complete\n'
  printf '\n'
  printf '  venv          %s\n' "$venv_dir"
  printf '  whisper-cli   %s\n' "$whisper_bin"
  printf '  core weights  %s\n' "$cache_dir"
  printf '  browser files %s\n' "$renderer_dir/public"
  printf '\n'
  printf '  Browser minimum: Chromium >= 107 or Firefox >= 104, from the default\n'
  printf '  baseline-widely-available output target that Vite emits for. The\n'
  printf '  WebAssembly SIMD onnxruntime-web and MediaPipe need landed earlier\n'
  printf '  (Chromium 91, Firefox 89), so the build target is the binding constraint.\n'
  printf '\n'
  printf '  Next: python doctor.py    then    ./run.sh\n'

  if [[ -s "$notes_file" ]]; then
    printf '\n'
    printf -- '--- read these before you trust this install ---\n\n'
    cat "$notes_file"
    printf '\n'
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  work_dir="$(mktemp -d)"
  notes_file="$work_dir/notes"
  : >"$notes_file"

  printf 'Luxo setup — %s, repo at %s\n' "$os_name" "$repo_dir"
  [[ -f "$manifest_path" ]] || die "config/models.yaml is missing; it is the asset manifest this script reads"
  [[ -f "$repo_dir/requirements.txt" ]] || die "requirements.txt is missing"

  # Written before anything uses it: the pip step reads requirements.txt
  # through it, and the asset steps read the manifest through it.
  write_manifest_reader

  step "Installing system packages"
  install_system_packages

  step "Resolving Node"
  ensure_node

  step "Creating the Python virtual environment"
  select_python
  info "using $python_bin (Python $python_series)"
  create_venv
  install_python_requirements

  unverified_marker="$("$python_bin" "$work_dir/manifest.py" marker "$manifest_path")"
  [[ -n "$unverified_marker" ]] || die "config/models.yaml does not declare unverified_marker"

  step "Building whisper.cpp from source"
  build_whisper

  step "Downloading model assets"
  info "manifest: config/models.yaml, unverified marker '$unverified_marker'"
  mkdir -p "$cache_dir"
  fetch_file_assets

  step "Installing renderer dependencies"
  build_renderer

  print_summary
}

main "$@"
