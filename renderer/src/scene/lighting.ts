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
const POINT_LIGHT_SCALE = 0.06;
const POINT_LIGHT_FORWARD_OFFSET_M = 0.025;
const EMISSIVE_SCALE = 2.2;
const BLOOM_RADIUS = 0.12;
const BLOOM_THRESHOLD = 0.95;

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

interface EmitterEdit {
  readonly mesh: Mesh;
  readonly original: Material | Material[];
  readonly emissive: ReadonlyArray<MeshPhongMaterial | MeshStandardMaterial>;
  readonly castShadow: boolean;
  readonly receiveShadow: boolean;
}

const attempt = (operation: () => void): void => {
  try {
    operation();
  } catch {
    // Cleanup must not replace the failure that initiated rollback.
  }
};

function restoreEmitterEdits(edits: readonly EmitterEdit[]): void {
  for (const edit of edits) {
    attempt(() => { edit.mesh.material = edit.original; });
    attempt(() => { edit.mesh.castShadow = edit.castShadow; });
    attempt(() => { edit.mesh.receiveShadow = edit.receiveShadow; });
    for (const material of edit.emissive) attempt(() => material.dispose());
  }
}

const findFrame = (robot: SemanticRobot, name: string): Object3D | undefined =>
  robot.links?.[name] ?? robot.getObjectByName(name);

function findEmitterEdits(emitter: Object3D): EmitterEdit[] {
  const edits: EmitterEdit[] = [];

  try {
    emitter.traverse((node) => {
      if (!(node instanceof Mesh)) return;
      const original = node.material;
      const materials = Array.isArray(original) ? original : [original];
      const emissive: Array<MeshPhongMaterial | MeshStandardMaterial> = [];
      let replacements: Material[];
      try {
        replacements = materials.map((material) => {
          if (
            material.name !== "fixture_light" ||
            !(material instanceof MeshPhongMaterial || material instanceof MeshStandardMaterial)
          ) return material;
          const replacement = material.clone();
          replacement.name = `${material.name}-emissive`;
          emissive.push(replacement);
          return replacement;
        });
      } catch (error) {
        for (const material of emissive) attempt(() => material.dispose());
        throw error;
      }
      if (emissive.length === 0) return;
      edits.push({
        mesh: node,
        original,
        emissive,
        castShadow: node.castShadow,
        receiveShadow: node.receiveShadow,
      });
      node.material = Array.isArray(original) ? replacements : replacements[0]!;
      // The bulb encloses the PointLight. A luminous bulb must not shadow or
      // receive shadows from its own source, or the point light is trapped.
      node.castShadow = false;
      node.receiveShadow = false;
    });

    if (edits.length === 0) {
      throw new Error("Luxo lighting cannot find the fixture_light emitter bulb");
    }
    return edits;
  } catch (error) {
    restoreEmitterEdits(edits);
    throw error;
  }
}

export class LightingRig {
  readonly source: PointLight;
  readonly composer: EffectComposer;
  readonly renderPass: RenderPass;
  readonly bloomPass: UnrealBloomPass;
  private readonly emitterEdits: EmitterEdit[];
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
    const shadowEnabled = options.renderer.shadowMap.enabled;
    const shadowType = options.renderer.shadowMap.type;
    const emitterEdits = findEmitterEdits(emitter);
    let source: PointLight | undefined;
    let composer: EffectComposer | undefined;
    let renderPass: RenderPass | undefined;
    let bloomPass: UnrealBloomPass | undefined;
    try {
      source = new PointLight(0xffffff, 0, 2.4, 2);
      source.name = "luxo-light-source";
      // Put the physical source at the front of the semantic bulb. Leaving it
      // at the link origin places it almost inside the shade's cover plate,
      // making that plate appear to be the emitter instead of the bulb.
      source.position.x = POINT_LIGHT_FORWARD_OFFSET_M;
      source.castShadow = true;
      source.shadow.mapSize.set(512, 512);
      source.shadow.bias = -0.0001;
      emitter.add(source);
      options.renderer.shadowMap.enabled = true;
      options.renderer.shadowMap.type = PCFSoftShadowMap;
      composer = new EffectComposer(options.renderer);
      renderPass = new RenderPass(options.scene, options.camera);
      composer.addPass(renderPass);
      bloomPass = new UnrealBloomPass(
        new Vector2(1, 1),
        0,
        BLOOM_RADIUS,
        BLOOM_THRESHOLD,
      );
      composer.addPass(bloomPass);
    } catch (error) {
      if (bloomPass) attempt(() => bloomPass?.dispose());
      if (renderPass) attempt(() => renderPass?.dispose());
      if (composer) attempt(() => composer?.dispose());
      if (source) {
        attempt(() => source?.removeFromParent());
        attempt(() => source?.dispose());
      }
      restoreEmitterEdits(emitterEdits);
      attempt(() => { options.renderer.shadowMap.enabled = shadowEnabled; });
      attempt(() => { options.renderer.shadowMap.type = shadowType; });
      throw error;
    }
    this.emitterEdits = emitterEdits;
    this.source = source;
    this.composer = composer;
    this.renderPass = renderPass;
    this.bloomPass = bloomPass;
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
    for (const edit of this.emitterEdits) {
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
    let firstError: unknown;
    const release = (operation: () => void) => {
      try { operation(); } catch (error) { firstError ??= error; }
    };
    release(() => this.source.removeFromParent());
    release(() => this.source.dispose());
    for (const edit of this.emitterEdits) {
      release(() => { edit.mesh.material = edit.original; });
      release(() => { edit.mesh.castShadow = edit.castShadow; });
      release(() => { edit.mesh.receiveShadow = edit.receiveShadow; });
      for (const material of edit.emissive) release(() => material.dispose());
    }
    release(() => this.renderPass.dispose());
    release(() => this.bloomPass.dispose());
    release(() => this.composer.dispose());
    if (firstError !== undefined) throw firstError;
  }
}
