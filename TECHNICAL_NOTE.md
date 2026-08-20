# Luxo — Technical Note

## 0. Status

Luxo is assembled as one working application: `core/main.py` constructs the
local speech models, OpenRouter client, scene memory, behaviour and observation
runtimes, animation loop, and protocol server; `run.sh` starts it with the Vite
renderer. Offline suites cover the Python and renderer boundaries. A live take
still requires staged assets, browser permissions, an available OpenRouter
model, and environment-specific camera, microphone, audio, and latency checks.

## 1. Architecture and data flow

**The browser is the body. Python is the mind.** Sensors sit with the light and
speakers because `getUserMedia` is identical on macOS and Ubuntu: it deletes the
AVFoundation/V4L2/PortAudio layer, gives free echo cancellation, and puts face
landmarking on the GPU rather than one of four scarce CPU cores.

```
┌─────────────── LAPTOP (localhost only) ─────────────────┐
│ BODY — browser (Vite · TS · three.js · Tone.js)         │
│  camera ─► MediaPipe ─► gaze @10Hz; JPEG ONLY on observe │
│  mic ───► Silero VAD (ONNX Web) ─► 1 PCM per utterance  │
│  URDF · PointLight · bloom · 8 SFX · music · TTS out    │
│  client.ts validates EVERY core message before dispatch │
└────────┬────────────────────────────────────────────────┘
   up   gaze·vad·tts_done·0x01 PCM·0x02 JPEG
   down body_state@60Hz·cue·capture_frame·0x03 PCM
┌────────▼────────────────────────────────────────────────┐
│ MIND — core (Python 3.12, one process)                  │
│  BLACKBOARD gaze · utterance · scene_memory · plan       │
│  10Hz   BehaviorFSM ─► conversation/observation ─► plans│
│  workers whisper.cpp · Piper · OpenRouter ──────────────┼─► cloud
│  120Hz  idle⊕gaze⊕gesture⊕light⊕bob ─► springs ─► clamps│
└─────────────────────────────────────────────────────────┘
```

The browser never chooses a gesture, light preset, or joint value, and gaze
*dwell* is evaluated in the core, so behavioural logic sits on one side of the
socket. A worker completion callback only enqueues a generation-tagged result;
publication happens on the serialized tick, so animation never blocks on I/O.

Cloud reasoning has three typed boundaries. `converse` receives the transcript,
up to three recent exchanges, compact historical memory, and the latest
privacy-filtered `currently_visible` facts. The model decides relevance,
wording, and whether the semantic plan should observe. `observe` alone carries
one JPEG and returns object facts, never dialogue. After Python commits those
facts and computes the deterministic missing set, `resolve_observation` receives
the exact typed origin, fresh facts, missing labels, current visible facts,
compact memory, and recent dialogue; it owns the final line and semantic plan.

## 2. Protocol

One socket at `ws://127.0.0.1:8765`. JSON on text frames; binary frames carry a
**1-byte type prefix** (`0x01` utterance PCM, `0x02` JPEG, `0x03` TTS PCM), and
`ProtocolServer` solely owns `0x03`, so framing has one author. The schema is
defined once in `schema/messages.schema.json`, TypeScript types are generated
from it, and a check asserts the generated file matches, so **core and renderer
cannot drift**.

## 3. The model-to-action boundary

**The model never emits joint angles.** It emits a plan over a closed eight-verb
enum: `gesture`, `look_at`, `light`, `sfx`, `scan`, `observe`, `posture`, `wait`.
The cloud model owns language interpretation, relevance, wording, and the
choice and order of those semantic actions. There is no parallel local phrase,
colour, object, or intent classifier rewriting that decision.
Every duration, easing curve, joint split, overshoot magnitude, velocity clamp,
and limit check lives in the animation layer, unreachable by the model.

- A hallucinated verb is dropped by the schema validator and logged.
- A hallucinated joint angle **cannot reach the body, because that path does not
  exist.**
- Physical safety is *structural*, not prompted. No prompt asks the model to
  respect joint limits, because it is never in a position to violate one.

## 4. The missing-object comparison

VLMs invent differences when asked to compare two scenes, so absence detection is
not theirs to perform:

> **The model performs perception and response. Python performs the
> comparison.**

An `observe` prompt includes the prior canonical list and asks which known
objects remain present and which objects are new, stabilising names across
frames. **Python computes `missing = L₀ − present`** as a deterministic set
difference against the snapshot taken before capture. A label absent from the
baseline can never become a missing item.

Python then gives `resolve_observation` the complete result, including an empty
missing set, rather than deciding locally whether the result is relevant or
whether Luxo should stay silent. The cloud resolves the exact dialogue turn or
scene event that caused the observation and returns the final `say` and semantic
plan. Memory remains a flat bounded list: no embeddings or vector search.

## 5. Privacy

Continuous gaze and hand landmarking stay in the browser; only derived sensor
measurements cross the local WebSocket. **A frame leaves the machine only on an
explicit `observe` action.** Dialogue observations are selected by the cloud
plan; a bounded hand-presentation scene event can also issue one. The renderer
validates the request and captures one JPEG. That scene image may include a
person or face, so observation is not anonymous; the guarantee is that no
continuous or incidental camera stream is uploaded.

OpenRouter text calls receive transcript text, at most three recent exchanges,
compact memory, and present-only scene projections containing stable IDs,
canonical labels, and filtered attributes. `resolve_observation` additionally
receives the typed origin, filtered fresh facts, and Python-computed missing
labels. Raw labels, bounding boxes, timestamps, local presence metadata, gaze,
joint angles, FSM state, telemetry, clamps, and audio are excluded. The default
free profile may permit provider retention or training; use the implemented
private profile when zero-data-retention guarantees are required.

## 6. Physical reasoning and simulation

Per-joint spring-damper, `ẍ = ω²(target − x) − 2ζω·ẋ`. The constants **mirror the
URDF's own velocity limits**: the shoulder is capped at 0.95 rad/s and gets
ω=6.5/ζ=0.90 (lags visibly); the neck permits 1.60 rad/s and gets ω=14.0/ζ=0.62
(overshoots). Overlap and follow-through fall out of the URDF's constraints
rather than being decorated on top. Output stage order is mandatory: sum →
integrate → **velocity clamp** → **soft-limit clamp** → emit. Commands clamp to
*soft* limits; hard limits do not appear in that module, and every clamp event is
counted and exposed as telemetry.

> On real hardware this behaviour layer would emit setpoints to a deterministic
> C++ or Rust servo loop at 500 Hz with hard deadlines. In simulation the
> animation layer is the final stage before the transport, so Python is
> sufficient and the model/body boundary is unchanged. Soft joint limits and
> per-joint velocity limits from the URDF are enforced in the output stage and
> clamp events are counted, so the same constraint contract would hold against a
> physical actuator.

Look-at is analytic, not IK: `base_yaw + neck_yaw = α`, the neck leads by up to
0.5 rad and recentres on a 0.9 s constant, so overshoot lands on the neck.

## 7. Deployment

Target: clean Ubuntu 24.04, 4 cores, 8 GB, no GPU. Native `setup.sh` + venv +
npm, **no Docker at runtime**. Venv is mandatory (PEP 668 blocks system pip on
Noble), requirements pinned with hashes, whisper.cpp built CPU-only from source,
models fetched to `~/.cache/luxo` with SHA256 verification, never committed.
Preflight splits in two: `doctor.py` checks core assets, configuration, local
resources, and optional OpenRouter reachability; `/selftest` exercises the
actual browser camera, microphone, WebGL, MediaPipe, audio, and WebSocket paths.
`localhost` is a secure context: no TLS and no LAN bind. `setup.sh`, `run.sh`,
the root README, and both preflight surfaces are present. A clean Ubuntu smoke
check remains a release check rather than something an offline unit suite proves.

## 8. Rejected alternatives

| Rejected | Reason |
|---|---|
| Local LLM/VLM | 4 cores, no GPU: 20–60 s/frame prefill, ~8–12 tok/s. |
| Python-side sensors | AVFoundation vs V4L2; reduces the browser to a display. |
| All-browser, no Python | WASM whisper/Piper slower; weakens Linux evidence. |
| Electron / Tauri | A build system for a tidiness problem. |
| Physics dynamics | Fights authored animation; authored overshoot is directable. |
| Next.js / shadcn | SSR and hydration add failure surface to one canvas. |
| Docker at runtime | Device passthrough is the dominant failure mode; costs RAM. |
| C++/Rust core | ~200 flops/tick; heavy compute already in compiled kernels. |

## 9. Measurements

Runtime measurement artifacts are written under the ignored `measurements/`
directory and are intentionally not committed. The submission should report
figures from the selected demo run rather than preserving workstation-specific
CSV output in source control.

| Group | Metrics | Value |
|---|---|---|
| Latency (§11.1) | end-of-speech → first audio p50 / p95; per-stage breakdown | `TBD — owner-measured` |
| Engagement (§11.3) | engage and disengage precision / recall; median time-to-engage; trials per condition (4) | `TBD — owner-measured` |
| Resources (§11.4) | peak RSS core / browser; CPU% per thread; tick jitter p99; dropped render frames | `TBD — owner-measured` |
| Effort | actual project hours | `TBD — owner-measured` |

**Required disclosures.**

1. **Measurements will be taken on macOS and do not transfer to the Ubuntu
   target.** whisper.cpp uses Accelerate/Metal on Apple Silicon and single cores
   far outpace four x86 cores, so any macOS figure is optimistic by a large
   multiple.
2. The Ubuntu component benchmark or smoke-check conditions must be stated
   explicitly. whisper.cpp and Piper are the two CPU-bound core components and
   will be materially slower on four x86 cores without Accelerate.
3. The recording's OpenRouter profile and model must be named. Free endpoints
   may permit provider retention and training; the private profile requests
   zero data retention and denies data collection. §5 describes payload scope,
   not a provider's downstream retention policy.

## 10. Known limitations

- **No roll DOF — the classic sideways curious head-tilt is physically impossible
  on this body.** The shade cannot rotate about its optical axis, and faking it
  with `neck_yaw` + `head_pitch` reads as a broken gimbal. Curiosity is expressed
  as **whole-body lean and crane** — the same constraint that makes the lamp
  stoop to inspect something low. The character design accommodates this; it is
  not a bug.
- No full IK; no physics dynamics; single subject; no barge-in; no wake word
  (gaze is the signal); no cross-frame tracking (`observe` is discrete);
  localhost only.
