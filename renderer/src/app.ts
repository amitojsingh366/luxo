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
  PointLight,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import type { URDFRobot } from "urdf-loader";

import { createStudioCamera } from "./scene/camera";
import {
  anchorLampBase,
  applyJointPositions,
  loadLampRobot,
  REST_JOINTS,
  type JointPositions,
} from "./scene/urdf";

export interface BodyStateLike {
  readonly joints: JointPositions;
  readonly light?: {
    readonly intensity: number;
    readonly color_k: number;
  };
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
  }),
});

const WARM_LIGHT = new Color("#ffd0a0");
const COOL_LIGHT = new Color("#d7e7ff");

function colorFromKelvin(kelvin: number, target: Color): Color {
  const mix = Math.min(1, Math.max(0, (kelvin - 2700) / (4500 - 2700)));
  return target.lerpColors(WARM_LIGHT, COOL_LIGHT, mix);
}

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
  const stage = makeElement("div", "lumen-stage");
  const canvas = makeElement("canvas", "lumen-canvas");
  const vignette = makeElement("div", "lumen-vignette");
  const brand = makeElement("div", "lumen-brand");
  const name = makeElement("span", "lumen-brand__name");
  const state = makeElement("span", "lumen-brand__state");
  const status = makeElement("div", "lumen-status");

  name.textContent = "Lumen";
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

function attachEmitterLight(robot: URDFRobot): PointLight {
  const emitter = robot.links.light_emitter_link;
  if (!emitter) {
    throw new Error("Lamp light_emitter_link is unavailable");
  }

  const light = new PointLight(WARM_LIGHT, 12, 2.4, 2);
  light.name = "lumen-emitter-light";
  light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  light.shadow.bias = -0.0001;
  emitter.add(light);
  return light;
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
  const resize = () => {
    const { width, height } = stage.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
    cameraRig.resize(width, height);
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(stage);
  resize();

  renderer.setAnimationLoop(() => {
    cameraRig.controls.update();
    renderer.render(scene, cameraRig.camera);
  });

  let robot: URDFRobot;
  try {
    robot = await loadLampRobot();
  } catch (error) {
    status.textContent = "Body model failed to load";
    status.dataset.tone = "error";
    throw error;
  }

  scene.add(anchorLampBase(robot));
  const emitterLight = attachEmitterLight(robot);
  const lightColor = new Color();

  const applyBodyState = (bodyState: BodyStateLike) => {
    applyJointPositions(robot, bodyState.joints);

    if (bodyState.light) {
      emitterLight.intensity = Math.max(0, bodyState.light.intensity) * 22;
      emitterLight.color.copy(colorFromKelvin(bodyState.light.color_k, lightColor));
    }
  };

  applyBodyState(STATIC_REST_BODY_STATE);
  status.textContent = "Body ready · rest pose";
  status.dataset.tone = "ready";

  return {
    applyBodyState,
    destroy() {
      renderer.setAnimationLoop(null);
      resizeObserver.disconnect();
      cameraRig.dispose();
      renderer.dispose();
      root.replaceChildren();
    },
  };
}
