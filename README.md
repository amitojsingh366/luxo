# Luxo

Luxo is an articulated five-degree-of-freedom desk lamp that behaves as one live
character rather than a set of separate AI demos. It watches through the laptop
camera, listens through the laptop microphone, and expresses itself through
motion, light, voice, sound effects, and music. The personality is eager and
puppyish: it perks up when someone looks at it, leans in to inspect things, and
droops when attention moves elsewhere. The supplied URDF under `robot/` is the
body; everything else in this repository is the character built around it.

## Status: not demo-ready

Every claim in this README was checked against commit `d61cdfd`.

The subsystems are built and unit-tested in isolation. They are **not assembled
into a running character yet**, and this repository will not produce a
demonstration today.

- **`core/main.py` is a scaffold.** It parses arguments, loads configuration,
  and serves the WebSocket protocol stream. It does not construct the behavior
  FSM, the animation loop, the blackboard, the brain wiring, or scene memory.
  Starting the core gives you a protocol server, not a character.
- **`setup.sh` does not exist in this repository yet.** The quickstart below is
  the intended flow, and its first step is currently missing. Until it lands,
  the virtualenv, npm install, model download, and whisper.cpp build are manual.
- **No model assets have been downloaded.** `python3 doctor.py` reports all
  seven entries in `config/models.yaml` as missing.
- **Four of those seven entries carry `sha256: UNVERIFIED`.** That is a
  deliberate marker for assets whose publisher ships no SHA-256; each entry
  records why and the exact command that closes the gap. `doctor.py` warns on an
  unverified-but-present asset instead of failing, so the gap stays visible.
- **No measurement has been taken.** `measurements/` contains no data, and this
  README states no latency, throughput, or resource figure.

## Architecture in brief

**The browser is the body. Python is the mind.** One WebSocket at
`ws://127.0.0.1:8765` connects them, and nothing else crosses that line.

- The browser owns the camera, microphone, gaze landmarking, voice-activity
  detection, the three.js render of the URDF, the point light and bloom, and all
  audio output. It makes no behavioral decisions.
- Python owns speech-to-text, text-to-speech, the behavior state machine, the
  language model boundary, scene memory, and the animation layers that turn
  semantic intent into joint values.
- The protocol is defined once in `schema/messages.schema.json`. The renderer's
  TypeScript types are generated from it, and a check fails if the checked-in
  file has drifted.

The design argument — why the split falls there, what the model is and is not
allowed to emit, how the missing-object comparison stays in Python, and the
rejected alternatives — is in [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md). This
README does not repeat it.

## Prerequisites

| Requirement | Notes |
|---|---|
| Ubuntu 24.04 LTS | The deployment target: 4 cores, 8 GB RAM, no GPU. macOS is supported for development. |
| Python 3.12 | `doctor.py` enforces 3.12 as a floor. A virtualenv is mandatory on Ubuntu, where PEP 668 blocks pip against the system interpreter. |
| Node.js `^20.19 \|\| >=22.12` | The `engines` requirement of the pinned Vite 8. |
| Chromium or Firefox, recent | Needs WebGL2, `getUserMedia`, Web Audio, and WebAssembly. `localhost` is a secure context, so no TLS is involved. |
| `espeak-ng` | A system package, not a Python one. `sudo apt-get install -y espeak-ng` on Ubuntu, `brew install espeak-ng` on macOS. It is Piper's phonemizer. |
| An OpenRouter API key | Exported as `OPENROUTER_API_KEY` in the shell that launches the core. `doctor.py` probes for presence only and never reads, prints, or transmits the value. |

whisper.cpp is built from source, CPU-only, and is not vendored here. Core-side
model weights live outside the repository, in `~/.cache/lumen`. Browser-side
assets are staged into `renderer/public/` at setup time. Neither set is
committed — see the note under [Repository layout](#repository-layout) about a
gap in `.gitignore` coverage.

## Quickstart

This is the intended sequence. Step 1 is not yet available — see
[Status](#status-not-demo-ready).

```sh
./setup.sh                      # NOT PRESENT YET: venv, npm ci, models, whisper.cpp
export OPENROUTER_API_KEY=...   # run.sh does not read .env; export it yourself
python3 doctor.py               # core-side preflight; non-zero exit means stop
./run.sh                        # starts the Python core and the Vite dev server
```

Then open two pages:

- `http://127.0.0.1:5173` — the character.
- `http://127.0.0.1:5173/selftest` — the browser half of preflight. It asks for
  camera and microphone permission and exercises the real sensor and audio path.
  `doctor.py` deliberately does not duplicate these checks, because they cannot
  be done outside a browser.

`run.sh` runs `python -m core.main` and `npm run dev` together and shuts both
down when either exits. The core binds loopback only, never `0.0.0.0`. The Vite
dev server uses `strictPort`, so port 5173 must be free.

Two smaller commands are useful on their own:

```sh
python3 -m core.main --check    # validate config and exit without opening a socket
python3 doctor.py --check-openrouter   # opt into one live authenticated API request
```

`doctor.py` checks the interpreter version, virtualenv, every asset in the
manifest, available memory, port 8765, the presence of the API key, and
`espeak-ng`. The live OpenRouter probe is off by default.

## Repository layout

```
.
├── CHALLENGE.md                 the brief this repository answers
├── SUBMISSION.md                what the assignment asks to be submitted
├── TECHNICAL_NOTE.md            architecture, tradeoffs, and known limitations
├── doctor.py                    core-side preflight; run before ./run.sh
├── run.sh                       starts the core and the Vite dev server together
├── requirements.txt             pinned, hash-locked Python runtime dependencies
├── config/
│   ├── default.yaml             rates, joint limits, springs, closed vocabularies
│   ├── models.yaml              model asset manifest: URLs, digests, licences
│   └── poses.yaml               the five canonical postures
├── core/                        the mind: one Python process
│   ├── main.py                  CLI entry point (a scaffold — see Status)
│   ├── config.py                immutable configuration loading
│   ├── logging_setup.py         process-wide logging
│   ├── blackboard.py            thread-safe exchange between workers and the tick
│   ├── fsm.py                   10 Hz behavior state machine
│   ├── plan_executor.py         sequences validated semantic plans
│   ├── wake_sequence.py         non-blocking warm-up and the waking sequence
│   ├── instrumentation.py       per-interaction latency timelines, written as CSV
│   ├── protocol/
│   │   ├── messages.py          parsing and serialization for JSON and binary frames
│   │   └── ws_server.py         reconnect-safe WebSocket transport
│   ├── runtime/
│   │   └── interactions.py      conversation staging: VAD → STT → brain → speech
│   ├── animation/               120 Hz motion, entirely body-owned
│   │   ├── runtime.py           coordinates the additive layers
│   │   ├── director.py          routes semantic intent to animation
│   │   ├── layers.py            the additive-layer boundary
│   │   ├── gestures.py          authored gestures and posture transitions
│   │   ├── poses.py             the canonical pose library
│   │   ├── lookat.py            analytic aiming, not IK
│   │   ├── springs.py           per-joint spring-damper integration
│   │   └── output_stage.py      velocity and soft-limit clamps, the last stage
│   ├── brain/
│   │   ├── client.py            OpenRouter boundary; worker threads only
│   │   ├── prompts.py           fixed instructions and privacy-limited payloads
│   │   ├── schema.py            validated plans over the closed verb enum
│   │   ├── observe.py           the only path from a captured frame to the model
│   │   ├── memory.py            flat scene memory
│   │   └── missing.py           the Python-side missing-object comparison
│   └── speech/
│       ├── stt.py               whisper.cpp transcription boundary
│       ├── tts.py               Piper synthesis through ONNX Runtime
│       └── phonemes.py          text to Piper phoneme IDs via eSpeak NG
├── renderer/                    the body: Vite, TypeScript, three.js, Tone.js
│   ├── README.md                the model contract and manual browser checks
│   ├── index.html               the character page
│   ├── selftest.html            browser preflight, served at /selftest
│   ├── src/
│   │   ├── main.ts              browser entry point
│   │   ├── runtime.ts           wires scene, audio, sensors, and protocol client
│   │   ├── app.ts               mounts the three.js scene
│   │   ├── degraded.ts          fallback behavior when the core disconnects
│   │   ├── selftest.ts          the checks the /selftest page runs
│   │   ├── protocol/            client.ts validates every core message;
│   │   │                        types.ts is generated — do not edit
│   │   ├── scene/               urdf.ts, lighting.ts, camera.ts
│   │   ├── sensors/             camera.ts, gaze.ts, mic.ts, vad.ts
│   │   ├── audio/               mixer.ts, music.ts, sfx.ts, ttsPlayer.ts
│   │   └── ui/overlay.ts        on-screen status and telemetry
│   └── validation tooling/                   Node validation suites
├── schema/
│   ├── messages.schema.json     the single source of truth for the protocol
│   ├── generate_types.py        emits renderer/src/protocol/types.ts
│   └── check_generated.py       fails when the checked-in types have drifted
├── robot/                       supplied and unmodified: URDF, mesh, reference image
├── validation tooling/                       Python unit checks (validation)
└── measurements/                where interaction CSVs will be written; empty today
```

`setup.sh` will also create `.venv/`, `renderer/node_modules/`, and
`renderer/public/`. The first two are gitignored, and so is any `models/`
directory. `renderer/public/mediapipe/` and `renderer/public/onnxruntime/` are
staged from `node_modules` but are **not** covered by `.gitignore` today; do not
commit them.

## Development

Python, from the repository root — 555 checks at `d61cdfd`:

```sh
```

Renderer, from `renderer/` — 113 checks at `d61cdfd`:

```sh
npm ci
npm run typecheck
npm run build      # runs the typecheck, then the Vite build
```

Both suites run offline. They cover subsystems in isolation, not the assembled
character, and passing them is not evidence that the demonstration works.

If you change `schema/messages.schema.json`, regenerate the renderer types and
confirm they are current:

```sh
python3 schema/generate_types.py
python3 schema/check_generated.py
```

`the validation tooling` asserts the same thing, so core and renderer
cannot drift apart silently.

## Privacy

This is an architectural property of the split, not a policy statement.

- **Gaze processing never leaves the browser.** Face landmarking runs locally.
  The only gaze fields the protocol carries are presence, yaw, pitch, azimuth,
  elevation, and a confidence scalar — no landmarks, no image data. No face ever
  reaches the core or a cloud service.
- **No camera frame is transmitted except on an explicit `observe`.** There is
  no other code path that uploads a frame. The renderer validates every message
  from the core before dispatch, so a malformed capture request cannot trigger a
  capture.
- **Only one utterance of audio is sent per turn.** The browser detects
  end-of-speech locally and sends that single utterance as PCM. There is no open
  microphone stream to the core.
- The interaction CSV written by `core/instrumentation.py` records timestamps,
  stage durations, model name, profile, and token counts. It cannot contain
  transcripts, prompts, model text, keys, audio, images, gaze, joint values, or
  FSM state.

What does leave the machine, and the retention consequences of the shipped
OpenRouter profile, are described in [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md)
§5 and §9. Read that before recording anything you care about.

## Limitations

Current status is covered under [Status](#status-not-demo-ready) above. Beyond
that:

- **The body has no roll degree of freedom**, so the classic sideways head-tilt
  is physically impossible. Curiosity is expressed as whole-body lean and crane
  instead. This is a property of the supplied URDF, not a bug.
- No full inverse kinematics and no physics dynamics; motion is authored and
  spring-damped.
- One subject at a time, no wake word (gaze is the signal), no barge-in, and no
  cross-frame tracking — `observe` is discrete.
- Localhost only. There is no LAN bind and no TLS path.
- `doctor.py`'s remediation text says `run.sh` loads a root `.env`. It does not;
  export `OPENROUTER_API_KEY` yourself.
- No Linux smoke check has been run. The target is Ubuntu 24.04, but everything
  so far has been exercised on macOS.

[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) §10 carries the full list.
