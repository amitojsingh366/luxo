import "./style.css";

import {
  ACESFilmicToneMapping,
  AmbientLight,
  Color,
  CylinderGeometry,
  DirectionalLight,
  Fog,
  Mesh,
  MeshStandardMaterial,
  PCFSoftShadowMap,
  PlaneGeometry,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import type { URDFRobot } from "urdf-loader";

import type { LightState } from "./protocol/types";
import { createStudioCamera } from "./scene/camera";
import { LightingRig } from "./scene/lighting";
import {
  anchorLampBase,
  applyJointPositions,
  loadLampRobot,
  REST_JOINTS,
  type JointPositions,
} from "./scene/urdf";

export interface BodyStateLike {
  readonly joints: JointPositions;
  readonly light: LightState;
}

export interface RendererHandle {
  applyBodyState(state: BodyStateLike): void;
  destroy(): void;
}

export const STATIC_REST_BODY_STATE: BodyStateLike = Object.freeze({
  joints: REST_JOINTS,
  light: Object.freeze({
    intensity: 0.55,
    color_k: 2700,
    pattern: "steady",
    bloom: 0.6,
  }),
});

function makeElement<K extends keyof HTMLElementTagNameMap>(
  name: K,
  className: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(name);
  element.className = className;
  return element;
}

function buildStage(root: HTMLElement): {
  stage: HTMLDivElement;
  canvas: HTMLCanvasElement;
  status: HTMLDivElement;
} {
  const stage = makeElement("div", "luxo-stage");
  const canvas = makeElement("canvas", "luxo-canvas");
  const vignette = makeElement("div", "luxo-vignette");
  const brand = makeElement("div", "luxo-brand");
  const name = makeElement("span", "luxo-brand__name");
  const state = makeElement("span", "luxo-brand__state");
  const status = makeElement("div", "luxo-status");

  name.textContent = "Luxo";
  state.textContent = "Browser body · 5 DOF";
  status.textContent = "Loading articulated body";
  status.dataset.tone = "loading";
  canvas.setAttribute("aria-label", "Interactive three-dimensional lamp scene");
  canvas.tabIndex = 0;
  vignette.setAttribute("aria-hidden", "true");

  brand.append(name, state);
  stage.append(canvas, vignette, brand, status);
  root.replaceChildren(stage);
  return { stage, canvas, status };
}

function addStudio(scene: Scene): void {
  const ground = new Mesh(
    new PlaneGeometry(8, 8),
    new MeshStandardMaterial({
      color: 0x10131a,
      metalness: 0.05,
      roughness: 0.92,
    }),
  );
  ground.position.z = -0.002;
  ground.receiveShadow = true;
  scene.add(ground);

  const plinth = new Mesh(
    new CylinderGeometry(0.24, 0.27, 0.025, 72),
    new MeshStandardMaterial({
      color: 0x161a22,
      metalness: 0.18,
      roughness: 0.68,
    }),
  );
  plinth.rotation.x = Math.PI / 2;
  plinth.position.z = -0.015;
  plinth.receiveShadow = true;
  scene.add(plinth);

  const ambient = new AmbientLight(0x8190aa, 0.62);
  scene.add(ambient);

  const key = new DirectionalLight(0xffe0b8, 3.6);
  key.position.set(-1.25, -1.4, 2.2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.1;
  key.shadow.camera.far = 5;
  key.shadow.camera.left = -1.2;
  key.shadow.camera.right = 1.2;
  key.shadow.camera.top = 1.2;
  key.shadow.camera.bottom = -1.2;
  key.shadow.bias = -0.0002;
  scene.add(key);

  const rim = new DirectionalLight(0x8ba9d6, 1.75);
  rim.position.set(1.25, 0.8, 1.35);
  scene.add(rim);
}

/**
 * Mount the browser-owned body. The returned method accepts any structurally
 * compatible protocol body state; this module never makes behavior choices.
 */
export async function mountRenderer(root: HTMLElement): Promise<RendererHandle> {
  const { stage, canvas, status } = buildStage(root);
  const scene = new Scene();
  scene.background = new Color(0x070a10);
  scene.fog = new Fog(0x070a10, 1.65, 4.2);
  addStudio(scene);

  const renderer = new WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;

  const cameraRig = createStudioCamera(canvas);
  let lightingRig: LightingRig | null = null;
  let destroyed = false;
  const resize = () => {
    const { width, height } = stage.getBoundingClientRect();
    const safeWidth = Math.max(width, 1);
    const safeHeight = Math.max(height, 1);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(safeWidth, safeHeight, false);
    cameraRig.resize(safeWidth, safeHeight);
    lightingRig?.resize(safeWidth, safeHeight, pixelRatio);
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(stage);
  resize();

  const teardown = (clearRoot: boolean) => {
    if (destroyed) return;
    destroyed = true;
    let firstError: unknown;
    const release = (operation: () => void) => {
      try {
        operation();
      } catch (error) {
        firstError ??= error;
      }
    };
    release(() => renderer.setAnimationLoop(null));
    release(() => resizeObserver.disconnect());
    const rig = lightingRig;
    lightingRig = null;
    release(() => rig?.dispose());
    release(() => cameraRig.dispose());
    release(() => renderer.dispose());
    if (clearRoot) release(() => root.replaceChildren());
    if (firstError !== undefined) throw firstError;
  };

  let robot: URDFRobot;
  let failureStatus = "Body model failed to load";
  try {
    robot = await loadLampRobot();
    scene.add(anchorLampBase(robot));
    failureStatus = "Body scene failed to initialise";
    lightingRig = new LightingRig({
      robot,
      renderer,
      scene,
      camera: cameraRig.camera,
    });
    resize();
    applyJointPositions(robot, STATIC_REST_BODY_STATE.joints);
    lightingRig.applyState(STATIC_REST_BODY_STATE.light);
  } catch (error) {
    status.textContent = failureStatus;
    status.dataset.tone = "error";
    try {
      teardown(false);
    } catch {
      // Setup failure remains the actionable error; cleanup is best effort.
    }
    throw error;
  }

  const applyBodyState = (bodyState: BodyStateLike) => {
    applyJointPositions(robot, bodyState.joints);
    lightingRig?.applyState(bodyState.light);
  };

  renderer.setAnimationLoop((nowMs) => {
    cameraRig.controls.update();
    lightingRig?.render(nowMs);
  });
  status.textContent = "Body ready · rest pose";
  status.dataset.tone = "ready";

  return {
    applyBodyState,
    destroy() {
      teardown(true);
    },
  };
}
