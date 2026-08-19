import {
  Group,
  LoadingManager,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
} from "three";
import URDFLoader, { type URDFRobot } from "urdf-loader";

import lampShadeUrl from "../../../robot/assets/lamp_shade.stl?url";
import lampUrdfUrl from "../../../robot/dummy_lamp_5dof.urdf?url";

export const BODY_TO_URDF_JOINT = {
  base_yaw: "base_yaw_joint",
  shoulder_pitch: "shoulder_pitch_joint",
  elbow_pitch: "elbow_pitch_joint",
  neck_yaw: "neck_yaw_joint",
  head_pitch: "head_pitch_joint",
} as const;

export type BodyJointName = keyof typeof BODY_TO_URDF_JOINT;
export type UrdfJointName = (typeof BODY_TO_URDF_JOINT)[BodyJointName];
export type JointPositions = Readonly<Record<BodyJointName, number>>;

export const REST_JOINTS: JointPositions = Object.freeze({
  base_yaw: 0,
  shoulder_pitch: 0.35,
  elbow_pitch: -0.75,
  neck_yaw: 0,
  head_pitch: 0.25,
});

export const LAMP_URDF_URL = lampUrdfUrl;
export const LAMP_SHADE_URL = lampShadeUrl;

const REQUIRED_SEMANTIC_FRAMES = [
  "speaker_link",
  "light_emitter_link",
  "camera_link",
] as const;

/** Resolve the URDF's bare-relative STL path to Vite's imported asset URL. */
export function resolveRobotAssetUrl(requestedUrl: string): string {
  const pathWithoutQuery = requestedUrl.split(/[?#]/, 1)[0] ?? requestedUrl;
  const normalizedPath = decodeURIComponent(pathWithoutQuery).replaceAll("\\", "/");

  if (normalizedPath.endsWith("assets/lamp_shade.stl")) {
    return LAMP_SHADE_URL;
  }

  return requestedUrl;
}

function assertLampFacts(robot: URDFRobot): void {
  for (const urdfJointName of Object.values(BODY_TO_URDF_JOINT)) {
    if (!robot.joints[urdfJointName]) {
      throw new Error(`Lamp URDF is missing required joint: ${urdfJointName}`);
    }
  }

  if (!robot.links.base_link) {
    throw new Error("Lamp URDF is missing base_link");
  }

  // These deliberately include an empty speaker and transform-only camera/light
  // links. Presence is required; inertia and renderable geometry are not.
  for (const frameName of REQUIRED_SEMANTIC_FRAMES) {
    const frame = robot.links[frameName];
    if (!frame) {
      throw new Error(`Lamp URDF is missing semantic frame: ${frameName}`);
    }
    frame.userData.semanticFrame = true;
  }

  const speakerFrame = robot.links.speaker_link;
  if (speakerFrame) {
    speakerFrame.visible = false;
  }
}

function enableLampShadows(root: Object3D): void {
  root.traverse((node) => {
    if (!(node instanceof Mesh)) return;

    node.castShadow = true;
    node.receiveShadow = true;

    const sourceMaterials = Array.isArray(node.material)
      ? node.material
      : [node.material];
    node.material = sourceMaterials.map((material: Material) => {
      if (material instanceof MeshStandardMaterial) {
        material.roughness = Math.max(material.roughness, 0.34);
      }
      return material;
    });

    if (node.material.length === 1) {
      node.material = node.material[0]!;
    }
  });
}

export async function loadLampRobot(): Promise<URDFRobot> {
  const manager = new LoadingManager();
  manager.setURLModifier(resolveRobotAssetUrl);
  let assetError: Error | undefined;
  const assetsReady = new Promise<void>((resolve) => {
    manager.onLoad = resolve;
    manager.onError = (url) => {
      assetError = new Error(`Failed to load lamp asset: ${url}`);
    };
  });

  const loader = new URDFLoader(manager);
  loader.parseCollision = false;

  const robot = await loader.loadAsync(LAMP_URDF_URL);
  await assetsReady;
  // STLLoader closes its manager item immediately before its onLoad callback.
  // Yield once so the mesh is attached before the body is announced as ready.
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  if (assetError) throw assetError;

  assertLampFacts(robot);
  enableLampShadows(robot);

  robot.name = "lumen-lamp";
  robot.position.set(0, 0, 0);
  robot.rotation.set(0, 0, 0);
  return robot;
}

/**
 * The URDF has no world joint. This immutable identity transform is the
 * renderer-owned world anchor for its root `base_link`.
 */
export function anchorLampBase(robot: URDFRobot): Group {
  const anchor = new Group();
  anchor.name = "lumen-base-anchor";
  anchor.matrixAutoUpdate = false;
  anchor.updateMatrix();
  anchor.add(robot);
  return anchor;
}

export function applyJointPositions(
  robot: URDFRobot,
  joints: JointPositions,
): void {
  const urdfValues: Record<UrdfJointName, number> = {
    base_yaw_joint: joints.base_yaw,
    shoulder_pitch_joint: joints.shoulder_pitch,
    elbow_pitch_joint: joints.elbow_pitch,
    neck_yaw_joint: joints.neck_yaw,
    head_pitch_joint: joints.head_pitch,
  };

  robot.setJointValues(urdfValues);
}
