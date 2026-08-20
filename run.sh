#!/usr/bin/env bash
#
# Luxo launcher — starts the two halves of the running system together and
# stops both as a unit: the Python core (WebSocket on ws://127.0.0.1:8765,
# PRD 10) and the renderer (Vite on loopback, PRD 12.3).
#
# What this script does, in order:
#
#   1. options       --dev / --prod / --skip-doctor  (see usage below)
#   2. .env          parse the root .env WITHOUT executing it and export what
#                    it declares.  See load_env_file: the file is data, never
#                    code, and no value out of it is ever printed.
#   3. preflight     python doctor.py (PRD 12.2).  A non-zero exit stops the
#                    launch; --skip-doctor is the development escape hatch.
#   4. bind check    confirm the renderer is still configured for loopback.
#                    PRD 12.3: localhost is a secure context, so getUserMedia
#                    works without TLS — and the renderer must not be bound to
#                    a LAN address.
#   5. launch        core and renderer, each as its own process group, with a
#                    trap that stops the survivor when either one exits.
#
# What this script deliberately never does:
#
#   * It never sources, dots, or evals .env, and never prints, logs, copies or
#     commits a value out of it.  A key that reaches a terminal, a CI log or a
#     shell history is a leaked key.
#   * It never passes --host to vite.  The bind address belongs to
#     renderer/vite.config.ts; this script reads and verifies it rather than
#     restating it, so the two cannot silently disagree.
#   * It never installs anything and adds no dependency.  Missing pieces are
#     setup.sh's job and this script says so instead of papering over them.
#
# Usage:
#   ./run.sh                  dev: the Vite dev server, from renderer sources
#   ./run.sh --prod           serve the already-built renderer/dist
#   ./run.sh --skip-doctor    launch even if the core preflight fails

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
renderer_dir="$repo_dir/renderer"
vite_config="$renderer_dir/vite.config.ts"
vite_bin="$renderer_dir/node_modules/.bin/vite"
env_file="$repo_dir/.env"

mode="dev"
skip_doctor=0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

info() { printf 'run.sh: %s\n' "$*"; }
warn() { printf 'run.sh: WARNING: %s\n' "$*" >&2; }
die() { printf '\nrun.sh: %s\n' "$*" >&2; exit 1; }

# Missing tooling is a different kind of failure from a misconfiguration, and
# 127 is the conventional "command not found" status the original script used.
die_missing() { printf '\nrun.sh: %s\n' "$*" >&2; exit 127; }

usage() {
  cat <<'USAGE'
usage: ./run.sh [--dev | --prod] [--skip-doctor]

  --dev           start the Vite dev server on loopback (default)
  --prod          serve the already-built renderer/dist with `vite preview`
  --skip-doctor   launch even if the core preflight (doctor.py) fails.
                  Development only: the doctor checks model hashes, the venv,
                  port 8765 and the API key, and a demo that skips it usually
                  fails later and less clearly.
  -h, --help      show this message

The root .env, if present, is read for KEY=VALUE lines before launch. It is
parsed as data and never executed, no value from it is ever printed, and a
variable already exported in the environment always wins over the file.
USAGE
}

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) mode="dev" ;;
    --prod | --production) mode="prod" ;;
    --skip-doctor) skip_doctor=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option '$1'"
      ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Tooling
# ---------------------------------------------------------------------------

if [[ -x "$repo_dir/.venv/bin/python" ]]; then
  python_bin="$repo_dir/.venv/bin/python"
elif ! python_bin="$(command -v python3)"; then
  die_missing "python3 is required to start Luxo"
fi

if [[ ! -d "$renderer_dir" ]]; then
  die "renderer directory is missing"
fi

if [[ "$mode" == "dev" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    die_missing "npm is required to start the renderer"
  fi
elif [[ ! -x "$vite_bin" ]]; then
  # --prod runs the vite that `npm ci` already installed. Nothing new is added.
  die_missing "$vite_bin is missing; run ./setup.sh (npm ci) before ./run.sh --prod"
fi

# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

# The root .env carries OPENROUTER_API_KEY and may carry OPENROUTER_MODEL. It is
# gitignored, contains secrets, and is DATA — so it is treated as data here.
#
# Why not `source .env`, `. .env`, or `eval`:
#   each of those hands the file to the shell parser, which means every line
#   becomes a shell command. A stray backtick, a `$(...)`, a `;` or a `>` in
#   what merely looks like a value then executes with this script's privileges
#   — and .env is precisely the file people paste into, copy between machines,
#   and generate with other tools. A secrets file must not be able to run code.
#
# What happens instead: `read -r` takes one line at a time and the line is
# taken apart with bash parameter expansion and pattern matching only. Bash
# expands a word once and does not rescan the RESULT of an expansion for
# metacharacters, so `$(id)` read out of the file stays five literal characters
# all the way into `export "$name=$value"` — a single, already-expanded, quoted
# word that the export builtin only assigns. There is no second parse anywhere
# on this path, which is exactly why file contents cannot become commands.
#
# Nothing from the file is ever printed: not a value, not a prefix, not a
# length, not a masked form. Malformed lines are reported by line number alone,
# because the malformed part may itself be the secret.
#
# Deliberate non-goals, so the behaviour is not mistaken for a full dotenv:
#   * backslash escapes inside quotes are NOT interpreted; a value is taken
#     literally after one layer of matching quotes is removed.
#   * an inline `#` is part of the value, not a comment. Comments must be on
#     their own line. Truncating a value at a `#` would silently corrupt any
#     secret containing one, which is the worse failure of the two.

# Result of trim_edges, which writes to a global rather than returning through
# a command substitution: no value ever becomes an argument to another process.
_trimmed=""

trim_edges() {
  # Strip leading and trailing spaces and tabs. A loop rather than extglob or
  # `${var%%*( )}` because macOS ships bash 3.2, which this script targets.
  _trimmed="$1"
  while [[ "$_trimmed" == [$' \t']* ]]; do _trimmed="${_trimmed#?}"; done
  while [[ "$_trimmed" == *[$' \t'] ]]; do _trimmed="${_trimmed%?}"; done
}

check_env_permissions() {
  local mode_bits="" other_bits
  mode_bits="$(stat -f '%Lp' "$env_file" 2>/dev/null || stat -c '%a' "$env_file" 2>/dev/null || true)"
  [[ -n "$mode_bits" ]] || return 0
  other_bits="${mode_bits: -2}"
  if [[ "$other_bits" != "00" ]]; then
    warn ".env is mode $mode_bits, so it is readable beyond its owner; chmod 600 '$env_file'"
  fi
}

load_env_file() {
  # Absent is normal and silent: the file is optional and the user may export
  # OPENROUTER_API_KEY by hand instead, which is what doctor.py documents.
  [[ -e "$env_file" ]] || return 0
  if [[ ! -f "$env_file" ]]; then
    warn "$env_file exists but is not a regular file; ignoring it"
    return 0
  fi
  if [[ ! -r "$env_file" ]]; then
    warn "$env_file is not readable; ignoring it"
    return 0
  fi

  check_env_permissions

  local line line_number=0 name value first last
  local exported=0 kept=0 ignored=0

  # `|| [[ -n "$line" ]]` so a final line with no trailing newline is not lost.
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))

    line="${line%$'\r'}" # tolerate a CRLF file
    trim_edges "$line"
    line="$_trimmed"

    [[ -n "$line" ]] || continue
    [[ "$line" != '#'* ]] || continue

    # `export KEY=VALUE` is accepted because that is what people paste.
    if [[ "$line" == export[$' \t']* ]]; then
      line="${line#export}"
      trim_edges "$line"
      line="$_trimmed"
    fi

    name="${line%%=*}"
    if [[ "$name" == "$line" ]]; then
      ignored=$((ignored + 1))
      warn ".env line $line_number is not KEY=VALUE; ignored (contents not shown)"
      continue
    fi
    value="${line#*=}"

    trim_edges "$name"
    name="$_trimmed"
    if [[ ! "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      ignored=$((ignored + 1))
      warn ".env line $line_number does not begin with a valid variable name; ignored (contents not shown)"
      continue
    fi

    # Refusing to execute the file would be pointless if the file could still
    # hijack how the processes launched below find and load their code.
    case "$name" in
      PATH | IFS | ENV | BASH_ENV | LD_PRELOAD | LD_LIBRARY_PATH | LD_AUDIT | \
        DYLD_INSERT_LIBRARIES | DYLD_LIBRARY_PATH | DYLD_FRAMEWORK_PATH)
        ignored=$((ignored + 1))
        warn ".env line $line_number sets $name; ignored (this file may not change how programs are found or loaded)"
        continue
        ;;
    esac

    trim_edges "$value"
    value="$_trimmed"

    # One layer of matching quotes, removed only when both ends agree, so that
    # a value containing a lone quote survives intact. Whitespace inside the
    # quotes is preserved; this is the way to keep leading or trailing spaces.
    if [[ "${#value}" -ge 2 ]]; then
      first="${value:0:1}"
      last="${value: -1}"
      if [[ "$first" == "$last" && ("$first" == '"' || "$first" == "'") ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    # An explicit export by the user wins over the file. Empty counts as unset,
    # matching doctor.py's key_is_present, so a stale `export KEY=` in the
    # shell cannot quietly block the file and then fail preflight.
    if [[ -n "${!name:-}" ]]; then
      kept=$((kept + 1))
      continue
    fi

    # A readonly or otherwise unsettable name must not abort the launch.
    if export "$name=$value" 2>/dev/null; then
      exported=$((exported + 1))
    else
      ignored=$((ignored + 1))
      warn ".env line $line_number declares a variable this shell cannot set; ignored"
    fi
  done <"$env_file"

  # Counts only. Which variables a secrets file defines is itself a hint about
  # its contents, and doctor.py reports on OPENROUTER_API_KEY by name anyway.
  info ".env read: $exported variable(s) exported, $kept already set and left alone, $ignored ignored"
}

# ---------------------------------------------------------------------------
# Preflight (PRD 12.2)
# ---------------------------------------------------------------------------

run_preflight() {
  if [[ "$skip_doctor" -eq 1 ]]; then
    warn "--skip-doctor: the core preflight was not run"
    return 0
  fi

  info "running the core preflight (doctor.py)"
  # Foreground and synchronous, so it cannot outlive this script: a Ctrl-C here
  # reaches the doctor directly and the EXIT trap below has no children to
  # reap. The doctor is offline by default and exits non-zero on any FAIL.
  if (
    cd -- "$repo_dir"
    exec "$python_bin" doctor.py
  ); then
    info "preflight passed"
    return 0
  fi

  die "preflight failed — fix the FAIL lines above before launching.
    The browser half of preflight is the /selftest page, after this starts.
    To launch anyway during development: ./run.sh --skip-doctor"
}

# ---------------------------------------------------------------------------
# Renderer bind address (PRD 12.3)
# ---------------------------------------------------------------------------

# renderer/vite.config.ts is the authority for the renderer's host and port, so
# this script reads it instead of guessing or overriding it. Today it declares
# server 127.0.0.1:5173 and preview 127.0.0.1:4173, both strictPort, and
# package.json's dev script repeats `--host 127.0.0.1` on the command line.
#
# This is a deliberately small scan, not a TypeScript parser: find the
# `server: {` or `preview: {` block and return the first `host:` or `port:`
# inside it. When it cannot find a value it returns nothing and the caller
# warns rather than inventing one.
vite_setting() {
  local block="$1" key="$2"
  [[ -f "$vite_config" ]] || return 0
  awk -v block="$block" -v key="$key" '
    index($0, block ": {") { inside = 1; next }
    inside && /^  \}/ { inside = 0 }
    inside {
      pos = index($0, key ":")
      if (pos > 1 && substr($0, 1, pos - 1) ~ /^[ \t]+$/) {
        value = substr($0, pos + length(key) + 1)
        gsub(/[ \t",;]/, "", value)
        print value
        exit
      }
    }
  ' "$vite_config"
}

is_loopback() {
  case "$1" in
    127.0.0.1 | localhost | ::1 | "[::1]") return 0 ;;
    *) return 1 ;;
  esac
}

# The dev server is started through `npm run dev`, and a --host on that script's
# command line overrides the config file, so the script line is checked too.
check_dev_script_host() {
  local script host_arg shown
  script="$(awk -F'"' '/"dev"[[:space:]]*:/ { print $4; exit }' "$renderer_dir/package.json" 2>/dev/null || true)"
  case "$script" in
    *"--host "*)
      host_arg="${script##*--host }"
      host_arg="${host_arg%% *}"
      shown="--host $host_arg"
      ;;
    *--host)
      # A bare --host tells vite to listen on every interface.
      host_arg="0.0.0.0"
      shown="a bare --host, which means every interface"
      ;;
    *)
      return 0
      ;;
  esac
  is_loopback "$host_arg" && return 0
  die "renderer/package.json runs the dev server with $shown.
    PRD 12.3: the renderer must stay on loopback — localhost is already a
    secure context, so getUserMedia needs no TLS and no LAN bind. Fix the dev
    script, or start the renderer yourself if you really mean to expose it."
}

check_renderer_bind() {
  local block="$1" host port
  host="$(vite_setting "$block" host || true)"
  port="$(vite_setting "$block" port || true)"
  renderer_url=""

  if [[ -z "$host" ]]; then
    warn "could not read the $block host out of renderer/vite.config.ts.
    PRD 12.3 requires the renderer to stay on loopback; verify it by hand."
    return 0
  fi

  if ! is_loopback "$host"; then
    die "renderer/vite.config.ts binds the $block host to '$host'.
    PRD 12.3: the renderer must not be exposed on a LAN address. localhost is
    a secure context on its own, so nothing is gained by binding wider."
  fi

  if [[ -n "$port" ]]; then
    renderer_url="http://$host:$port"
  else
    renderer_url="http://$host"
  fi
  info "renderer binds $host${port:+:$port} (loopback only)"
}

# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------

core_pid=""
renderer_pid=""
renderer_url=""

# Both children are started under `set -m`, so each leads its own process
# group and can be signalled as a tree. That matters for the renderer: `npm run
# dev` spawns vite as a grandchild and does not reliably forward SIGTERM to it,
# so signalling only the npm pid can leave a vite holding port 5173 — which,
# with strictPort, breaks the next run.
signal_group() {
  local pid="$1" signal="$2"
  [[ -n "$pid" ]] || return 0
  kill -"$signal" -"$pid" 2>/dev/null || kill -"$signal" "$pid" 2>/dev/null || true
}

group_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 -"$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null
}

terminate_children() {
  local child_pid waited=0

  signal_group "$core_pid" TERM
  signal_group "$renderer_pid" TERM

  # Up to five seconds to shut down cleanly, then stop being polite.
  while [[ "$waited" -lt 50 ]] && { group_alive "$core_pid" || group_alive "$renderer_pid"; }; do
    sleep 0.1
    waited=$((waited + 1))
  done
  if group_alive "$core_pid" || group_alive "$renderer_pid"; then
    warn "a child did not stop within 5s; sending SIGKILL"
    signal_group "$core_pid" KILL
    signal_group "$renderer_pid" KILL
  fi

  for child_pid in "$core_pid" "$renderer_pid"; do
    if [[ -n "$child_pid" ]]; then
      wait "$child_pid" 2>/dev/null || true
    fi
  done
}

handle_signal() {
  local exit_code="$1"
  trap - EXIT INT TERM
  terminate_children
  exit "$exit_code"
}

start_renderer() {
  # stdin comes from /dev/null explicitly. Without job control bash already
  # does this for background commands; under `set -m` it does not, and vite
  # binds keyboard shortcuts when stdin is a TTY — from a background process
  # group that read would raise SIGTTIN and stop the dev server.
  if [[ "$mode" == "prod" ]]; then
    (
      cd -- "$renderer_dir"
      exec "$vite_bin" preview
    ) </dev/null &
  else
    npm run dev --prefix "$renderer_dir" </dev/null &
  fi
  renderer_pid="$!"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Installed before anything can fail, and before any child exists, so a failure
# between here and the launch still unwinds through one path. With both pids
# still empty, terminate_children is a no-op and the exit status is untouched.
trap terminate_children EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

load_env_file

if [[ "$mode" == "prod" ]]; then
  # Production serving without adding a dependency: vite is already a
  # devDependency installed by `npm ci`, and vite.config.ts already configures
  # a preview server (127.0.0.1:4173, strictPort) and wires the /selftest route
  # into it. So `vite preview` serves the real build with no new package.
  #
  # Limitation, stated plainly: this is still vite's own server, not a
  # hardened static server, and renderer/package.json has no `preview` script,
  # so the local binary is invoked directly. Adding that script would be
  # tidier, but package.json is not this script's file to edit.
  [[ -f "$renderer_dir/dist/index.html" ]] \
    || die "renderer/dist is not built; run: npm run build --prefix '$renderer_dir'"
  check_renderer_bind preview
else
  check_renderer_bind server
  check_dev_script_host
fi

run_preflight

set -m
(
  cd -- "$repo_dir"
  exec "$python_bin" -m core.main
) </dev/null &
core_pid="$!"

start_renderer
set +m

info "core on ws://127.0.0.1:8765, renderer on ${renderer_url:-loopback}"
info "browser preflight: ${renderer_url:-http://127.0.0.1:5173}/selftest"

while kill -0 "$core_pid" 2>/dev/null && kill -0 "$renderer_pid" 2>/dev/null; do
  sleep 0.2
done

set +e
if ! kill -0 "$core_pid" 2>/dev/null; then
  wait "$core_pid"
  exit_code="$?"
else
  wait "$renderer_pid"
  exit_code="$?"
fi
set -e

exit "$exit_code"
