import { PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CAMERA_POSITION = new Vector3(-1.28, -1.18, 0.96);
const CAMERA_TARGET = new Vector3(0, 0, 0.39);

export interface StudioCameraRig {
  readonly camera: PerspectiveCamera;
  readonly controls: OrbitControls;
  resize(width: number, height: number): void;
  dispose(): void;
}

/**
 * Build a Z-up camera rig for the URDF's ROS coordinate system.
 * Orbit controls are intentionally present for pose and mesh debugging.
 */
export function createStudioCamera(canvas: HTMLCanvasElement): StudioCameraRig {
  const camera = new PerspectiveCamera(33, 1, 0.02, 20);
  camera.up.set(0, 0, 1);
  camera.position.copy(CAMERA_POSITION);

  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(CAMERA_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.enablePan = false;
  controls.minDistance = 0.62;
  controls.maxDistance = 3.2;
  controls.minPolarAngle = 0.26;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.rotateSpeed = 0.62;
  controls.zoomSpeed = 0.75;
  controls.update();

  return {
    camera,
    controls,
    resize(width: number, height: number) {
      camera.aspect = Math.max(width, 1) / Math.max(height, 1);
      camera.updateProjectionMatrix();
    },
    dispose() {
      controls.dispose();
    },
  };
}
