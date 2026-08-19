import { Camera, Color, Material, Mesh, MeshPhongMaterial, MeshStandardMaterial, Object3D, PCFSoftShadowMap, PointLight, Scene, SRGBColorSpace, Vector2, WebGLRenderer } from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

import type { LightState } from "../protocol/types";

export const LIGHT_PATTERN_NAMES = Object.freeze([
  "steady",
  "pulse",
  "flicker",
  "blink",
] as const);
export type LightPattern = (typeof LIGHT_PATTERN_NAMES)[number];

export const LIGHT_PRESET_NAMES = Object.freeze([
  "warm_idle",
  "warm_bright",
  "excited_flash",
  "curious_focus",
  "thinking_pulse",
  "cool_dim",
  "sad_fade",
] as const);
export type LightPresetName = (typeof LIGHT_PRESET_NAMES)[number];

export interface LightPreset {
  readonly intensity: number;
  readonly color_k: number;
  readonly pattern: LightPattern;
  readonly frequency_hz?: number;
}

const preset = (
  intensity: number,
  color_k: number,
  pattern: LightPattern = "steady",
  frequency_hz?: number,
): LightPreset => Object.freeze(
  frequency_hz === undefined
    ? { intensity, color_k, pattern }
    : { intensity, color_k, pattern, frequency_hz },
);

export const LIGHT_PRESETS: Readonly<Record<LightPresetName, LightPreset>> =
  Object.freeze({
    warm_idle: preset(0.55, 2700),
    warm_bright: preset(1, 2900),
    excited_flash: preset(1.15, 3100),
    curious_focus: preset(0.95, 3400),
    thinking_pulse: preset(0.7, 2800, "pulse", 0.8),
    cool_dim: preset(0.35, 4200),
    sad_fade: preset(0.2, 4500),
  });

export const THINKING_PULSE_HZ = 0.8;
export const BLINK_DURATION_MS = 90;
export const BLINK_INTENSITY = 0.55;
const MIN_KELVIN = 1000;
const MAX_KELVIN = 40000;
const MAX_INTENSITY = 2;
const MAX_BLOOM = 2;
const POINT_LIGHT_SCALE = 22;
const EMISSIVE_SCALE = 2.4;

const clamp = (value: number, low: number, high: number): number =>
  Math.min(high, Math.max(low, Number.isFinite(value) ? value : low));

/** Deterministic pattern multiplier; elapsed time is measured from pattern entry. */
export function lightPatternMultiplier(
  pattern: LightPattern,
  elapsedMs: number,
): number {
  const safeElapsedMs = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0;
  const seconds = safeElapsedMs / 1000;
  switch (pattern) {
    case "steady":
      return 1;
    case "pulse":
      return 0.72 + 0.28 * (0.5 + 0.5 * Math.cos(2 * Math.PI * 0.8 * seconds));
    case "flicker":
      return clamp(
        0.89 +
          0.075 * Math.sin(2 * Math.PI * 17.17 * seconds) +
          0.035 * Math.sin(2 * Math.PI * 3.73 * seconds + 1.7),
        0.78,
        1,
      );
    case "blink":
      return safeElapsedMs < BLINK_DURATION_MS ? BLINK_INTENSITY : 1;
  }
}

/** Tanner Helland's black-body approximation, returned as normalized sRGB. */
export function kelvinToRgb(kelvin: number): readonly [number, number, number] {
  const temperature = clamp(kelvin, MIN_KELVIN, MAX_KELVIN) / 100;
  const red = temperature <= 66
    ? 255
    : 329.698727446 * (temperature - 60) ** -0.1332047592;
  const green = temperature <= 66
    ? 99.4708025861 * Math.log(temperature) - 161.1195681661
    : 288.1221695283 * (temperature - 60) ** -0.0755148492;
  const blue = temperature >= 66
    ? 255
    : temperature <= 19
      ? 0
      : 138.5177312231 * Math.log(temperature - 10) - 305.0447927307;
  return Object.freeze([
    clamp(red, 0, 255) / 255,
    clamp(green, 0, 255) / 255,
    clamp(blue, 0, 255) / 255,
  ]);
}

interface SemanticRobot extends Object3D {
  readonly links?: Readonly<Record<string, Object3D>>;
}

export interface LightingRigOptions {
  readonly robot: SemanticRobot;
  readonly renderer: WebGLRenderer;
  readonly scene: Scene;
  readonly camera: Camera;
}

interface ApertureEdit {
  readonly mesh: Mesh;
  readonly original: Material | Material[];
  readonly emissive: ReadonlyArray<MeshPhongMaterial | MeshStandardMaterial>;
}

const findFrame = (robot: SemanticRobot, name: string): Object3D | undefined =>
  robot.links?.[name] ?? robot.getObjectByName(name);

function findApertureEdits(robot: SemanticRobot, emitter: Object3D): ApertureEdit[] {
  const head = findFrame(robot, "lamp_head_link") ?? emitter.parent;
  if (!head) throw new Error("Luxo lighting cannot resolve lamp_head_link");
  const edits: ApertureEdit[] = [];

  head.traverse((node) => {
    if (!(node instanceof Mesh) || emitter === node || emitter.getObjectById(node.id)) return;
    const original = node.material;
    const materials = Array.isArray(original) ? original : [original];
    const emissive: Array<MeshPhongMaterial | MeshStandardMaterial> = [];
    const replacements = materials.map((material) => {
      if (
        material.name !== "fixture_light" ||
        !(material instanceof MeshPhongMaterial || material instanceof MeshStandardMaterial)
      ) return material;
      const replacement = material.clone();
      replacement.name = `${material.name}-emissive`;
      emissive.push(replacement);
      return replacement;
    });
    if (emissive.length === 0) return;
    node.material = Array.isArray(original) ? replacements : replacements[0]!;
    edits.push({ mesh: node, original, emissive });
  });

  if (edits.length === 0) {
    throw new Error("Luxo lighting cannot find the fixture_light shade aperture");
  }
  return edits;
}

export class LightingRig {
  readonly source: PointLight;
  readonly composer: EffectComposer;
  readonly bloomPass: UnrealBloomPass;
  private readonly apertureEdits: ApertureEdit[];
  private readonly color = new Color();
  private state: LightState = {
    intensity: 0,
    color_k: 2700,
    pattern: "steady",
    bloom: 0,
  };
  private patternStartedAt = 0;
  private disposed = false;

  constructor(options: LightingRigOptions) {
    const emitter = findFrame(options.robot, "light_emitter_link");
    if (!emitter) {
      throw new Error("Luxo lighting requires semantic frame light_emitter_link");
    }
    this.apertureEdits = findApertureEdits(options.robot, emitter);
    this.source = new PointLight(0xffffff, 0, 2.4, 2);
    this.source.name = "luxo-light-source";
    this.source.castShadow = true;
    this.source.shadow.mapSize.set(512, 512);
    this.source.shadow.bias = -0.0001;
    emitter.add(this.source);
    options.renderer.shadowMap.enabled = true;
    options.renderer.shadowMap.type = PCFSoftShadowMap;

    this.composer = new EffectComposer(options.renderer);
    this.composer.addPass(new RenderPass(options.scene, options.camera));
    this.bloomPass = new UnrealBloomPass(new Vector2(1, 1), 0, 0.35, 0.18);
    this.composer.addPass(this.bloomPass);
  }

  applyState(state: LightState, nowMs = performance.now()): void {
    if (this.disposed) throw new Error("Luxo LightingRig has been disposed");
    if (!LIGHT_PATTERN_NAMES.includes(state.pattern)) {
      throw new RangeError(`Unknown Luxo light pattern: ${state.pattern}`);
    }
    const sampleMs = Number.isFinite(nowMs) ? nowMs : 0;
    if (state.pattern !== this.state.pattern) this.patternStartedAt = sampleMs;
    this.state = {
      intensity: clamp(state.intensity, 0, MAX_INTENSITY),
      color_k: clamp(state.color_k, MIN_KELVIN, MAX_KELVIN),
      pattern: state.pattern,
      bloom: clamp(state.bloom, 0, MAX_BLOOM),
    };
    this.update(sampleMs);
  }

  private update(nowMs: number): void {
    const multiplier = lightPatternMultiplier(
      this.state.pattern,
      nowMs - this.patternStartedAt,
    );
    const level = this.state.intensity * multiplier;
    const [red, green, blue] = kelvinToRgb(this.state.color_k);
    this.color.setRGB(red, green, blue, SRGBColorSpace);
    this.source.color.copy(this.color);
    this.source.intensity = level * POINT_LIGHT_SCALE;
    this.bloomPass.strength = this.state.bloom;
    for (const edit of this.apertureEdits) {
      for (const material of edit.emissive) {
        material.emissive.copy(this.color);
        material.emissiveIntensity = level * EMISSIVE_SCALE;
      }
    }
  }

  resize(width: number, height: number, pixelRatio = 1): void {
    this.composer.setPixelRatio(clamp(pixelRatio, 0.5, 2));
    this.composer.setSize(clamp(width, 1, 16384), clamp(height, 1, 16384));
  }

  render(nowMs = performance.now()): void {
    if (this.disposed) return;
    this.update(nowMs);
    this.composer.render();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.source.removeFromParent();
    this.source.dispose();
    for (const edit of this.apertureEdits) {
      edit.mesh.material = edit.original;
      for (const material of edit.emissive) material.dispose();
    }
    this.bloomPass.dispose();
    this.composer.dispose();
  }
}
