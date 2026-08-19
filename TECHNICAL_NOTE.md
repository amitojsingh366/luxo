# Luxo — Technical Note

## 0. Status

As of commit `0eeb863`, 420 Python and 113 renderer unit checks pass offline.
They check subsystems in isolation, **not** the assembled character.
`core/main.py` loads config and serves the protocol stream; it does not yet
construct the character loop. **The system is not demo-ready as of this
commit.** No recording and no measurement exist.

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
│  10Hz   BehaviorFSM ─► ConversationCoordinator ─► Plans │
│  workers whisper.cpp · Piper · OpenRouter ──────────────┼─► cloud
│  120Hz  idle⊕gaze⊕gesture⊕light⊕bob ─► springs ─► clamps│
└─────────────────────────────────────────────────────────┘
```

The browser never chooses a gesture, light preset, or joint value, and gaze
*dwell* is evaluated in the core, so behavioural logic sits on one side of the
socket. A worker completion callback only enqueues a generation-tagged result;
publication happens on the serialized tick, so animation never blocks on I/O.

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

> **The model performs perception and narration. Python performs the
> comparison.**

The first `observe` stores canonical list `L₀`. On the goal the lamp scans and
observes again; the second prompt includes `L₀` and asks only what is still
present and what is new, which stabilises naming ("mug" vs "cup"). **Python
computes `missing = L₀ − present`** as a deterministic set difference. Only then
is the model given `missing`, and asked only to narrate it — `observe` returns
objects, never dialogue. A label reported present but absent from the baseline is
local evidence only and **can never add an item to `missing`**. The model cannot
hallucinate a missing object because it never performs the comparison. Memory is
a flat list: no embeddings, N < 20.

## 5. Privacy

Gaze never leaves the browser; only derived yaw/pitch/azimuth cross the socket.
**No frame is ever sent except on an explicit `observe`**, so faces never reach a
cloud vision service. This is architectural, not policy: no code path uploads a
frame outside `observe`, and the renderer validates every core message before
dispatch, so a malformed `capture_frame` cannot trigger a capture. Outbound
payloads carry only transcript text, memory as one compact line of canonical
labels, the last three `say` exchanges, and on `observe` one JPEG. **Never
sent:** joint angles, FSM state, telemetry, gaze, clamps, audio, faces.

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
models fetched to `~/.cache/lumen` with SHA256 verification, never committed.
Preflight splits in two: `doctor.py` (913 lines, 131 offline unit checks, injected
probes, presence-only key handling) and a `/selftest` page exercising the real
runtime path. `localhost` is a secure context: no TLS, no LAN bind.

**Gap:** `doctor.py`, `run.sh`, and `selftest.html` exist. **`setup.sh` and the
root `README.md` are absent, and no Linux smoke check has been run.**

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

## 9. Measurements — none taken

**No benchmark has been run.** `measurements/` contains no CSV.

| Group | Metrics | Value |
|---|---|---|
| Latency (§11.1) | end-of-speech → first audio p50 / p95; per-stage breakdown | `TBD — owner-measured` |
| Engagement (§11.3) | engage and disengage precision / recall; median time-to-engage; trials per condition (4) | `TBD — owner-measured` |
| Resources (§11.4) | peak RSS core / browser; CPU% per thread; tick jitter p99; dropped render frames | `TBD — owner-measured` |
| Effort | actual project hours | `TBD — owner-measured` |

**Three disclosures.**

1. **Measurements will be taken on macOS and do not transfer to the Ubuntu
   target.** whisper.cpp uses Accelerate/Metal on Apple Silicon and single cores
   far outpace four x86 cores, so any macOS figure is optimistic by a large
   multiple.
2. **The Ubuntu component benchmark was deliberately cut**, so no measured Linux
   figure exists. As reasoning, not data: whisper.cpp and Piper are the two
   CPU-bound core components, and both should be materially slower on four x86
   cores without Accelerate — enough that a macOS figure is not representative.
   The size of that gap is unmeasured.
3. **The recording will use the OpenRouter `free` profile.** Free endpoints
   generally permit provider retention and training, which is **incompatible**
   with the `private` profile's `zdr: true` + `data_collection: "deny"`. Both are
   implemented; the shipped default is `free`. Transcripts and `observe` frames
   sent under it may be retained and trained on. §5 describes what leaves the
   machine, not what the provider does with it afterwards.

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
- **`core/main.py` does not yet assemble the character loop** (§0) — the gap
  between "all parts built and tested" and "demo".
