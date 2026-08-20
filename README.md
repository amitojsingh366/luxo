# Luxo

Luxo is an articulated five-degree-of-freedom desk lamp that behaves as a single,
live character. It watches through the laptop camera, listens through the laptop
microphone, and expresses itself through
motion, light, voice, sound effects, and music. The personality is eager and
puppyish: it perks up when someone looks at it, leans in to inspect things, and
droops when attention moves elsewhere. The supplied URDF under `robot/` is the
body; everything else in this repository is the character built around it.

## Current state

The character is assembled as one application. `core/main.py` constructs the
speech, OpenRouter, memory, behaviour, observation, animation, and protocol
boundaries; `run.sh` starts that core together with the Vite renderer. Live use
still depends on the locally staged model assets, browser camera and microphone
permissions, and a paid OpenRouter API key with a vision-capable model. The free
models do not support the image input used by `observe`. Testing used
`google/gemini-2.5-flash-lite:nitro`, which is the recommended model for running
and testing Luxo.

Core preflight checks and the renderer typecheck and build run offline. Camera,
microphone, browser GPU, voice quality, OpenRouter latency, and the clean Ubuntu
installation remain environment-dependent checks; use `doctor.py` and
`/selftest` before a take.

## Architecture in brief

The browser renderer and Python core communicate over a local WebSocket at
`ws://127.0.0.1:8765`. All messages between the two use this connection.

- The browser owns the camera, microphone, gaze landmarking, voice-activity
  detection, the three.js render of the URDF, the point light and bloom, and all
  audio output. It makes no behavioral decisions.
- Python owns speech-to-text, text-to-speech, the behavior state machine, the
  language model boundary, scene memory, and the animation layers that turn
  semantic intent into joint values.
- OpenRouter owns language interpretation, relevance, wording, and semantic
  plan selection. Python sends each conversation the transcript, recent turns,
  compact memory, and a filtered snapshot of what is currently visible.
- A visual dialogue turn has exactly three successful cloud calls:
  `converse` selects one terminal `observe`, `observe` receives one JPEG, and
  `resolve_observation` produces the grounded reply and plan.
- Vision returns at most 10 meaningful objects nearest-to-camera first, plus
  stable prior matches, a focused-object index, and presence evidence for
  lower-priority prior objects. Python keeps the current nearest objects first,
  fills remaining memory slots from recent history, and computes missing
  objects by stable ID.
- The protocol is defined once in `schema/messages.schema.json`. The renderer's
  TypeScript types are generated from it, and a check fails if the checked-in
  file has drifted.

[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) explains why the split falls there,
what the model may emit, why Python handles the missing-object comparison, and
which alternatives were rejected.

## Prerequisites

| Requirement | Notes |
|---|---|
| Ubuntu 24.04 LTS | The deployment target: 4 cores, 8 GB RAM, no GPU. macOS is supported for development. |
| Python 3.12 | `doctor.py` enforces 3.12 as a floor. A virtualenv is mandatory on Ubuntu, where PEP 668 blocks pip against the system interpreter. |
| Node.js `^20.19 \|\| >=22.12` | The `engines` requirement of the pinned Vite 8. |
| Chromium or Firefox, recent | Needs WebGL2, `getUserMedia`, Web Audio, and WebAssembly. `localhost` is a secure context, so no TLS is involved. |
| `espeak-ng` | A system package, not a Python one. `sudo apt-get install -y espeak-ng` on Ubuntu, `brew install espeak-ng` on macOS. It is Piper's phonemizer. |
| OpenRouter configuration | Required. Put a paid API key in `OPENROUTER_API_KEY` and set `OPENROUTER_MODEL=google/gemini-2.5-flash-lite:nitro` in the root `.env`. This is the model used during development and is recommended for testing. The free models do not support image input. `run.sh` parses `.env` as data without executing or printing its contents. `doctor.py` probes only for key presence. |

`OPENROUTER_API_KEY` is required. Without it, conversation, image observation,
and model-directed behaviour do not work, so Luxo will not function as intended.

whisper.cpp is built from source, CPU-only, and is not vendored here. Core-side
model weights live outside the repository, in `~/.cache/luxo`. Browser-side
assets are staged into `renderer/public/` at setup time. Neither set is
committed; the generated asset destinations are covered by `.gitignore`.

## Quickstart

```sh
./setup.sh                      # venv, pip, npm ci, models, whisper.cpp, espeak-ng
# In .env: OPENROUTER_API_KEY=... and OPENROUTER_MODEL=google/gemini-2.5-flash-lite:nitro
python3 doctor.py               # core-side preflight; non-zero exit means stop
./run.sh                        # safely loads .env, then starts core and renderer
```

Then open two pages:

- `http://127.0.0.1:5173` — the character.
- `http://127.0.0.1:5173/selftest` — the browser half of preflight. It asks for
  camera and microphone permission and exercises the real sensor and audio path.
  Those checks require a browser, so `doctor.py` covers the core preflight.

`run.sh` runs `python -m core.main` and `npm run dev` together and shuts both
down when either exits. The core binds to `127.0.0.1`. The Vite dev server uses
`strictPort`, so port 5173 must be free.

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
│   ├── main.py                  assembles and launches the character core
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
│   │   ├── app.py               assembled character and serialized tick wiring
│   │   ├── interactions.py      conversation staging: VAD → STT → brain → speech
│   │   └── observations.py      one-frame observation and cloud resolution flow
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
│   │   ├── memory.py            bounded v2 scene memory and stable ID allocator
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
├── schema/
│   ├── messages.schema.json     the single source of truth for the protocol
│   ├── generate_types.py        emits renderer/src/protocol/types.ts
│   └── check_generated.py       fails when the checked-in types have drifted
├── robot/                       supplied body assets and hand-authored animation JSON
└── measurements/                ignored runtime CSV output, created as needed
```

`setup.sh` also creates `.venv/`, `renderer/node_modules/`, and staged assets
under `renderer/public/`. Those generated paths, model weights, recordings,
environment files, and measurement outputs are gitignored and must not be
committed.

## Development

Renderer, from `renderer/`:

```sh
npm ci
npm run typecheck
npm run build      # runs the typecheck, then the Vite build
```

These checks do not replace a live browser, sensor, audio, OpenRouter, or Linux
smoke check.

If you change `schema/messages.schema.json`, regenerate the renderer types and
confirm they are current:

```sh
python3 schema/generate_types.py
python3 schema/check_generated.py
```

## Privacy

The split enforces these privacy properties.

- Continuous gaze processing stays local. Face and hand landmarking run in
  the browser. Only derived measurements cross the local WebSocket; landmarks
  and the continuous camera stream do not.
- A camera frame leaves the machine only for an explicit `observe` action.
  Dialogue observations are selected by the cloud plan; the bounded hand-dwell
  scene event can also issue one. The renderer captures one JPEG for the
  validated request and no other code path uploads frames. That scene JPEG may
  contain a person or face, so observation is not anonymous even though
  continuous gaze stays local.
- OpenRouter text calls receive the transcript, up to three recent exchanges,
  compact memory, and filtered current scene facts. Post-observation resolution
  also receives the typed origin, filtered fresh facts, and Python-computed
  missing objects as stable IDs with canonical names. Bounding boxes,
  timestamps, raw labels, gaze, telemetry, and joint state are excluded from
  text payloads.
- Only one utterance of audio is sent per turn. The browser detects
  end-of-speech locally and sends that single utterance as PCM. There is no open
  microphone stream to the core, and OpenRouter receives the transcript rather
  than the audio.
- The interaction CSV written by `core/instrumentation.py` records timestamps,
  stage durations, model name, profile, and token counts. Its schema excludes
  transcripts, prompts, model text, keys, audio, images, gaze, joint values, or
  FSM state.

What does leave the machine, and the retention consequences of the shipped
OpenRouter profile, are described in [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md)
§5. Read that before recording anything you care about.

## Limitations

- The body has no roll degree of freedom, so the classic sideways head-tilt
  is physically impossible. Curiosity is expressed as whole-body lean and crane
  instead. The supplied URDF imposes this constraint.
- No full inverse kinematics and no physics dynamics; motion is authored and
  spring-damped.
- One subject at a time, no wake word (gaze is the signal), no barge-in, and no
  cross-frame tracking. `observe` is discrete.
- Localhost only. There is no LAN bind and no TLS path.
- `run.sh` safely loads the root `.env` as data. A non-empty value already
  exported in the parent shell takes precedence over the value in `.env`.
- No Linux smoke check has been run. The target is Ubuntu 24.04, but everything
  so far has been exercised on macOS.

[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) §10 carries the full list.
