import type { BodyStateMessage, JointsState } from "./protocol/types";
import { LIGHT_PRESETS } from "./scene/lighting";
import { REST_JOINTS } from "./scene/urdf";

export const CORE_DISCONNECT_DRIFT_SECONDS = 2;

export const DEGRADED_JOINT_NAMES = Object.freeze([
  "base_yaw",
  "shoulder_pitch",
  "elbow_pitch",
  "neck_yaw",
  "head_pitch",
] as const satisfies readonly (keyof JointsState)[]);

export type RendererCoreStatus = "connected" | "disconnected";

const SOFT_LIMITS: Readonly<
  Record<keyof JointsState, readonly [number, number]>
> = Object.freeze({
  base_yaw: [-2.45, 2.45] as const,
  shoulder_pitch: [-0.65, 0.95] as const,
  elbow_pitch: [-1.7, 0.3] as const,
  neck_yaw: [-1.25, 1.25] as const,
  head_pitch: [-0.8, 0.6] as const,
});

const LIGHT_PATTERNS = new Set(["steady", "pulse", "flicker", "blink"]);
const FSM_STATES = new Set([
  "BOOT",
  "DORMANT",
  "NOTICING",
  "ENGAGED",
  "LISTENING",
  "THINKING",
  "SPEAKING",
  "INSPECTING",
  "ACTING",
  "DISENGAGING",
]);

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function nonnegative(value: unknown): value is number {
  return finite(value) && value >= 0;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const keys = Object.keys(value);
  return keys.length === expected.length && expected.every((key) => key in value);
}

export function isValidRendererBodyState(
  value: unknown,
): value is BodyStateMessage {
  const state = record(value);
  if (
    state === null ||
    state.type !== "body_state" ||
    !nonnegative(state.t) ||
    !nonnegative(state.seq)
  ) {
    return false;
  }

  const joints = record(state.joints);
  if (joints === null || !exactKeys(joints, DEGRADED_JOINT_NAMES)) return false;
  for (const name of DEGRADED_JOINT_NAMES) {
    const angle = joints[name];
    const [lower, upper] = SOFT_LIMITS[name];
    if (!finite(angle) || angle < lower || angle > upper) return false;
  }

  const light = record(state.light);
  if (
    light === null ||
    !nonnegative(light.intensity) ||
    !finite(light.color_k) ||
    light.color_k <= 0 ||
    !nonnegative(light.bloom) ||
    typeof light.pattern !== "string" ||
    !LIGHT_PATTERNS.has(light.pattern)
  ) {
    return false;
  }

  const audio = record(state.audio);
  if (
    audio === null ||
    typeof audio.speaking !== "boolean" ||
    !finite(audio.arousal) ||
    audio.arousal < 0 ||
    audio.arousal > 1
  ) {
    return false;
  }

  const telemetry = record(state.telemetry);
  const clamps = record(telemetry?.clamps);
  const gaze = record(telemetry?.gaze);
  return (
    telemetry !== null &&
    typeof telemetry.state === "string" &&
    FSM_STATES.has(telemetry.state) &&
    nonnegative(telemetry.plan_depth) &&
    nonnegative(telemetry.memory_count) &&
    nonnegative(telemetry.last_latency_ms) &&
    clamps !== null &&
    nonnegative(clamps.vel) &&
    nonnegative(clamps.limit) &&
    gaze !== null &&
    typeof gaze.present === "boolean" &&
    finite(gaze.yaw_deg) &&
    finite(gaze.pitch_deg)
  );
}

function freezeBodyState(state: BodyStateMessage): BodyStateMessage {
  const joints = Object.freeze({ ...state.joints });
  const light = Object.freeze({ ...state.light });
  const audio = Object.freeze({ ...state.audio });
  const telemetry = Object.freeze({
    ...state.telemetry,
    clamps: Object.freeze({ ...state.telemetry.clamps }),
    gaze: Object.freeze({ ...state.telemetry.gaze }),
  });
  return Object.freeze({ ...state, joints, light, audio, telemetry });
}

function mix(from: number, to: number, progress: number): number {
  if (progress <= 0) return from;
  if (progress >= 1) return to;
  return from + (to - from) * progress;
}

function degradedSnapshot(
  baseline: BodyStateMessage,
  progress: number,
): BodyStateMessage {
  if (progress <= 0) return baseline;
  const joints = Object.freeze({
    base_yaw: mix(baseline.joints.base_yaw, REST_JOINTS.base_yaw, progress),
    shoulder_pitch: mix(
      baseline.joints.shoulder_pitch,
      REST_JOINTS.shoulder_pitch,
      progress,
    ),
    elbow_pitch: mix(
      baseline.joints.elbow_pitch,
      REST_JOINTS.elbow_pitch,
      progress,
    ),
    neck_yaw: mix(baseline.joints.neck_yaw, REST_JOINTS.neck_yaw, progress),
    head_pitch: mix(
      baseline.joints.head_pitch,
      REST_JOINTS.head_pitch,
      progress,
    ),
  });
  const light = Object.freeze({
    intensity: mix(
      baseline.light.intensity,
      LIGHT_PRESETS.cool_dim.intensity,
      progress,
    ),
    color_k: mix(
      baseline.light.color_k,
      LIGHT_PRESETS.cool_dim.color_k,
      progress,
    ),
    pattern: LIGHT_PRESETS.cool_dim.pattern,
    bloom: baseline.light.bloom,
  });
  return Object.freeze({
    ...baseline,
    joints,
    light,
  });
}

export class RendererDisconnectFallback {
  private statusValue: RendererCoreStatus = "connected";
  private lastCoreState: BodyStateMessage | null = null;
  private baseline: BodyStateMessage | null = null;
  private output: BodyStateMessage | null = null;
  private disconnectedAt: number | null = null;
  private lastClock: number | null = null;
  private disposed = false;

  get status(): RendererCoreStatus {
    return this.statusValue;
  }

  get lastValidState(): BodyStateMessage | null {
    return this.lastCoreState;
  }

  acceptBodyState(state: BodyStateMessage): boolean {
    this.assertUsable();
    if (this.statusValue !== "connected" || !isValidRendererBodyState(state)) {
      return false;
    }
    if (
      this.lastCoreState !== null &&
      (state.t < this.lastCoreState.t || state.seq < this.lastCoreState.seq)
    ) {
      return false;
    }
    const snapshot = freezeBodyState(state);
    this.lastCoreState = snapshot;
    this.output = snapshot;
    return true;
  }

  disconnect(atSeconds: number): boolean {
    this.assertUsable();
    if (!nonnegative(atSeconds)) return false;
    if (this.statusValue === "disconnected") return true;
    const monotonicTime =
      this.lastClock === null ? atSeconds : Math.max(atSeconds, this.lastClock);
    this.lastClock = monotonicTime;
    this.statusValue = "disconnected";
    this.disconnectedAt = monotonicTime;
    this.baseline = this.lastCoreState;
    this.output = this.baseline;
    return true;
  }

  connect(): void {
    this.assertUsable();
    if (this.statusValue === "connected") return;
    this.statusValue = "connected";
    this.disconnectedAt = null;
    this.baseline = null;
  }

  sample(atSeconds: number): BodyStateMessage | null {
    this.assertUsable();
    if (!nonnegative(atSeconds)) return this.output;
    const monotonicTime =
      this.lastClock === null ? atSeconds : Math.max(atSeconds, this.lastClock);
    this.lastClock = monotonicTime;
    if (
      this.statusValue !== "disconnected" ||
      this.disconnectedAt === null ||
      this.baseline === null
    ) {
      return this.output;
    }
    const progress = Math.min(
      1,
      (monotonicTime - this.disconnectedAt) / CORE_DISCONNECT_DRIFT_SECONDS,
    );
    this.output = degradedSnapshot(this.baseline, progress);
    return this.output;
  }

  reset(): void {
    this.assertUsable();
    this.statusValue = "connected";
    this.lastCoreState = null;
    this.baseline = null;
    this.output = null;
    this.disconnectedAt = null;
    this.lastClock = null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.reset();
    this.disposed = true;
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error("Renderer disconnect fallback has been disposed");
    }
  }
}
