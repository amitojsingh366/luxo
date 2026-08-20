import type {
  FaceLandmarkerOptions,
  FaceLandmarkerResult,
  HandLandmarkerOptions,
  HandLandmarkerResult,
  Matrix,
  NormalizedLandmark,
} from '@mediapipe/tasks-vision';

import type { GazeMessage } from '../protocol/types';
import type { CameraSensor } from './camera';

export const GAZE_HZ = 10;
export const GAZE_INTERVAL_MS = 1_000 / GAZE_HZ;
export const GAZE_EMA_ALPHA = 0.4;
export const GAZE_INPUT_SIZE = Object.freeze({ width: 320, height: 240 });
export const DEFAULT_WASM_PATH = '/mediapipe/wasm';
export const DEFAULT_MODEL_PATH = '/models/face_landmarker.task';
export const DEFAULT_HAND_MODEL_PATH = '/models/hand_landmarker.task';

export type GazeLandmarkerResult = Pick<
  FaceLandmarkerResult,
  'faceLandmarks' | 'facialTransformationMatrixes'
>;

export interface GazeLandmarker {
  detectForVideo(
    frame: HTMLCanvasElement,
    timestampMs: number,
  ): GazeLandmarkerResult | Promise<GazeLandmarkerResult>;
  close(): void;
}

export type HandTrackingResult = Pick<HandLandmarkerResult, 'landmarks' | 'handedness'>;

export interface HandTrackingLandmarker {
  detectForVideo(
    frame: HTMLCanvasElement,
    timestampMs: number,
  ): HandTrackingResult | Promise<HandTrackingResult>;
  close(): void;
}

export interface FaceLandmarkerConfiguration {
  readonly wasmBasePath: string;
  readonly options: FaceLandmarkerOptions;
}

export interface HandLandmarkerConfiguration {
  readonly wasmBasePath: string;
  readonly options: HandLandmarkerOptions;
}

export interface GazeClock {
  nowMs(): number;
  nowSeconds(): number;
}

export interface GazeScheduler {
  setInterval(callback: () => void, intervalMs: number): unknown;
  clearInterval(handle: unknown): void;
}

type GazeCamera = Pick<
  CameraSensor,
  'cameraSpec' | 'drawAnalysisFrame' | 'live'
>;

export interface GazeSensorOptions {
  readonly camera: GazeCamera;
  readonly publish: (message: GazeMessage) => void;
  readonly createLandmarker?: (
    configuration: FaceLandmarkerConfiguration,
  ) => Promise<GazeLandmarker>;
  readonly enableHandTracking?: boolean;
  readonly createHandLandmarker?: (
    configuration: HandLandmarkerConfiguration,
  ) => Promise<HandTrackingLandmarker>;
  readonly clock?: GazeClock;
  readonly scheduler?: GazeScheduler;
  readonly wasmBasePath?: string;
  readonly modelAssetPath?: string;
  readonly handModelAssetPath?: string;
  readonly onDebugFrame?: (frame: LocalVisionDebugFrame) => void;
  readonly onError?: (error: Error) => void;
}

export interface FaceOrientation {
  readonly yawDeg: number;
  readonly pitchDeg: number;
}

export interface FaceCentroid {
  readonly x: number;
  readonly y: number;
}

export interface TargetAngles {
  readonly az: number;
  readonly el: number;
  readonly vfovRad: number;
}

export interface NormalizedRegion {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface LocalVisionDetection {
  readonly kind: 'face' | 'hand';
  readonly index: number;
  readonly bounds: NormalizedRegion;
  readonly centroid: FaceCentroid;
  readonly confidence: number;
}

export interface LocalVisionDebugFrame {
  readonly t: number;
  readonly face: LocalVisionDetection | null;
  readonly hands: readonly LocalVisionDetection[];
  readonly handCentroid: FaceCentroid | null;
}

export function createFaceLandmarkerConfiguration(
  modelAssetPath = DEFAULT_MODEL_PATH,
  wasmBasePath = DEFAULT_WASM_PATH,
): FaceLandmarkerConfiguration {
  requireLocalAssetPath(modelAssetPath, 'modelAssetPath');
  requireLocalAssetPath(wasmBasePath, 'wasmBasePath');
  return {
    wasmBasePath,
    options: {
      baseOptions: { modelAssetPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    },
  };
}

export function createHandLandmarkerConfiguration(
  modelAssetPath = DEFAULT_HAND_MODEL_PATH,
  wasmBasePath = DEFAULT_WASM_PATH,
): HandLandmarkerConfiguration {
  requireLocalAssetPath(modelAssetPath, 'handModelAssetPath');
  requireLocalAssetPath(wasmBasePath, 'wasmBasePath');
  return {
    wasmBasePath,
    options: {
      baseOptions: { modelAssetPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    },
  };
}

export const DEFAULT_FACE_LANDMARKER_CONFIGURATION = Object.freeze(
  createFaceLandmarkerConfiguration(),
);
export const DEFAULT_HAND_LANDMARKER_CONFIGURATION = Object.freeze(
  createHandLandmarkerConfiguration(),
);

/**
 * Extract face orientation from MediaPipe's row-major canonical-face-to-camera
 * 4x4 transform. Camera +x is image-right, +y is up, and it looks along -z.
 * We decompose Rz(roll) * Ry(yaw) * Rx(pitch), report positive yaw toward
 * image-right, and negate camera pitch so positive pitch means looking down.
 */
export function extractFaceOrientation(matrix: Matrix): FaceOrientation {
  if (
    !Number.isInteger(matrix.rows) ||
    !Number.isInteger(matrix.columns) ||
    matrix.rows < 3 ||
    matrix.columns < 3 ||
    matrix.data.length < matrix.rows * matrix.columns
  ) {
    throw new RangeError('Face transform must be a complete matrix of at least 3x3');
  }
  const columns = matrix.columns;
  const r00 = finite(matrix.data[0], 'matrix[0,0]');
  const r10 = finite(matrix.data[columns], 'matrix[1,0]');
  const r20 = finite(matrix.data[2 * columns], 'matrix[2,0]');
  const r21 = finite(matrix.data[2 * columns + 1], 'matrix[2,1]');
  const r22 = finite(matrix.data[2 * columns + 2], 'matrix[2,2]');
  const yaw = Math.atan2(-r20, Math.hypot(r00, r10));
  const cameraPitch = Math.atan2(r21, r22);
  return {
    yawDeg: clamp(radiansToDegrees(yaw), -180, 180),
    pitchDeg: clamp(-radiansToDegrees(cameraPitch), -180, 180),
  };
}

export function faceCentroid(
  landmarks: readonly Pick<NormalizedLandmark, 'x' | 'y'>[],
): FaceCentroid {
  if (landmarks.length === 0) throw new RangeError('Face has no landmarks');
  let x = 0;
  let y = 0;
  for (const landmark of landmarks) {
    x += finite(landmark.x, 'landmark.x');
    y += finite(landmark.y, 'landmark.y');
  }
  return {
    x: clamp(x / landmarks.length, 0, 1),
    y: clamp(y / landmarks.length, 0, 1),
  };
}

export function verticalFovRadians(
  horizontalFovDegrees: number,
  width: number,
  height: number,
): number {
  const hfov = degreesToRadians(validHorizontalFov(horizontalFovDegrees));
  const aspect = positiveFinite(width, 'camera width') / positiveFinite(height, 'camera height');
  return 2 * Math.atan(Math.tan(hfov / 2) / aspect);
}

/** The selfie view mirrors horizontal motion; image-down remains positive elevation. */
export function targetAnglesFromCentroid(
  centroid: FaceCentroid,
  horizontalFovDegrees: number,
  width: number,
  height: number,
): TargetAngles {
  const x = clamp(finite(centroid.x, 'centroid.x'), 0, 1);
  const y = clamp(finite(centroid.y, 'centroid.y'), 0, 1);
  const hfov = degreesToRadians(validHorizontalFov(horizontalFovDegrees));
  const vfov = verticalFovRadians(horizontalFovDegrees, width, height);
  return {
    // Luxo faces the person across the desk, so their image-right is Luxo's
    // left. This is the opposite sign from a camera mounted on Luxo itself.
    az: -Math.atan(2 * (x - 0.5) * Math.tan(hfov / 2)),
    // Luxo's positive head pitch looks down, so image-down must stay positive.
    el: Math.atan(2 * (y - 0.5) * Math.tan(vfov / 2)),
    vfovRad: vfov,
  };
}

export function handsCentroid(
  hands: readonly (readonly Pick<NormalizedLandmark, 'x' | 'y'>[])[],
): FaceCentroid | null {
  const landmarks = hands.flat();
  if (landmarks.length === 0) return null;
  return faceCentroid(landmarks);
}

export function normalizedLandmarkBounds(
  landmarks: readonly Pick<NormalizedLandmark, 'x' | 'y'>[],
): NormalizedRegion {
  if (landmarks.length === 0) throw new RangeError('Detection has no landmarks');
  let left = 1;
  let top = 1;
  let right = 0;
  let bottom = 0;
  for (const landmark of landmarks) {
    const x = clamp(finite(landmark.x, 'landmark.x'), 0, 1);
    const y = clamp(finite(landmark.y, 'landmark.y'), 0, 1);
    left = Math.min(left, x);
    top = Math.min(top, y);
    right = Math.max(right, x);
    bottom = Math.max(bottom, y);
  }
  return Object.freeze({
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  });
}

export function acceptedHandConfidence(result: HandTrackingResult | null): number {
  const values = (result?.handedness ?? [])
    .flat()
    .map((category) => category.score)
    .filter((value): value is number => Number.isFinite(value));
  if (values.length === 0) return result?.landmarks.length ? 0.55 : 0;
  return clamp(Math.max(...values), 0.55, 1);
}

export function acceptedFaceConfidence(
  landmarks: readonly Pick<NormalizedLandmark, 'visibility'>[],
): number {
  const values = landmarks
    .map((landmark) => landmark.visibility)
    .filter((value): value is number => Number.isFinite(value));
  if (values.length === 0) return 0.5;
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  // MediaPipe has already accepted the face through three 0.5 thresholds.
  return clamp(mean, 0.5, 1);
}

export class GazeEma {
  private value: FaceOrientation | null = null;

  update(sample: FaceOrientation): FaceOrientation {
    const yawDeg = finite(sample.yawDeg, 'yawDeg');
    const pitchDeg = finite(sample.pitchDeg, 'pitchDeg');
    if (!this.value) {
      this.value = { yawDeg, pitchDeg };
    } else {
      this.value = {
        yawDeg: this.value.yawDeg + GAZE_EMA_ALPHA * (yawDeg - this.value.yawDeg),
        pitchDeg:
          this.value.pitchDeg + GAZE_EMA_ALPHA * (pitchDeg - this.value.pitchDeg),
      };
    }
    return this.value;
  }

  reset(): void {
    this.value = null;
  }
}

export function absentGazeMessage(t: number): GazeMessage {
  return {
    type: 'gaze',
    t: safeProtocolSeconds(t),
    present: false,
    yaw_deg: 0,
    pitch_deg: 0,
    az: 0,
    el: 0,
    conf: 0,
  };
}

export class GazeSensor {
  private readonly camera: GazeCamera;
  private readonly publish: (message: GazeMessage) => void;
  private readonly createLandmarker: NonNullable<GazeSensorOptions['createLandmarker']>;
  private readonly createHandLandmarker: NonNullable<GazeSensorOptions['createHandLandmarker']>;
  private readonly clock: GazeClock;
  private readonly scheduler: GazeScheduler;
  private readonly configuration: FaceLandmarkerConfiguration;
  private readonly handConfiguration: HandLandmarkerConfiguration;
  private readonly handTrackingEnabled: boolean;
  private readonly onDebugFrame?: (frame: LocalVisionDebugFrame) => void;
  private readonly onError?: (error: Error) => void;
  private readonly ema = new GazeEma();
  private landmarker: GazeLandmarker | null = null;
  private handLandmarker: HandTrackingLandmarker | null = null;
  private timer: unknown = null;
  private startPromise: Promise<void> | null = null;
  private inFlight = false;
  private completed: GazeLandmarkerResult | null = null;
  private completedHands: HandTrackingResult | null = null;
  private completedReady = false;
  private generation = 0;
  private disposed = false;
  private lastMediaTimestampMs = -Infinity;

  constructor(options: GazeSensorOptions) {
    this.camera = options.camera;
    this.publish = options.publish;
    this.createLandmarker = options.createLandmarker ?? createDefaultLandmarker;
    this.createHandLandmarker = options.createHandLandmarker ?? createDefaultHandLandmarker;
    this.handTrackingEnabled = options.enableHandTracking === true;
    this.clock = options.clock ?? {
      nowMs: () => performance.now(),
      nowSeconds: () => Date.now() / 1_000,
    };
    this.scheduler = options.scheduler ?? {
      setInterval: (callback, intervalMs) => globalThis.setInterval(callback, intervalMs),
      clearInterval: (handle) => globalThis.clearInterval(handle as number),
    };
    this.configuration = createFaceLandmarkerConfiguration(
      options.modelAssetPath,
      options.wasmBasePath,
    );
    this.handConfiguration = createHandLandmarkerConfiguration(
      options.handModelAssetPath,
      options.wasmBasePath,
    );
    this.onDebugFrame = options.onDebugFrame;
    this.onError = options.onError;
  }

  get running(): boolean {
    return this.landmarker !== null && this.timer !== null;
  }

  start(): Promise<void> {
    if (this.disposed) return Promise.reject(new Error('Gaze sensor is disposed'));
    if (!this.camera.live) return Promise.reject(new Error('Camera permission is required before gaze starts'));
    if (this.running) return Promise.resolve();
    if (this.startPromise) return this.startPromise;
    const generation = ++this.generation;
    const pending = this.createTrackingModels()
      .then(({ face, hands }) => {
        if (generation !== this.generation || this.disposed || !this.camera.live) {
          face.close();
          hands?.close();
          throw new Error('Gaze start was cancelled');
        }
        this.landmarker = face;
        this.handLandmarker = hands;
        this.timer = this.scheduler.setInterval(
          () => this.tick(generation),
          GAZE_INTERVAL_MS,
        );
      })
      .catch((value: unknown) => {
        const error = toError(value);
        this.reportError(error);
        throw error;
      });
    const tracked = pending.finally(() => {
      if (this.startPromise === tracked) this.startPromise = null;
    });
    this.startPromise = tracked;
    return tracked;
  }

  stop(): void {
    this.generation += 1;
    this.startPromise = null;
    if (this.timer !== null) this.scheduler.clearInterval(this.timer);
    this.timer = null;
    const landmarker = this.landmarker;
    this.landmarker = null;
    if (landmarker) landmarker.close();
    const handLandmarker = this.handLandmarker;
    this.handLandmarker = null;
    if (handLandmarker) handLandmarker.close();
    this.inFlight = false;
    this.completed = null;
    this.completedHands = null;
    this.completedReady = false;
    this.lastMediaTimestampMs = -Infinity;
    this.ema.reset();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stop();
  }

  private tick(generation: number): void {
    if (generation !== this.generation || !this.camera.live || !this.landmarker) return;
    const t = safeProtocolSeconds(this.clock.nowSeconds());
    if (this.completedReady) {
      const result = this.completed;
      const hands = this.completedHands;
      this.completed = null;
      this.completedHands = null;
      this.completedReady = false;
      this.publishResult(result, hands, t);
    } else {
      this.ema.reset();
      this.publish(this.withHandFact(absentGazeMessage(t), null));
    }
    if (this.inFlight) return;
    this.beginDetection(generation);
  }

  private beginDetection(generation: number): void {
    const landmarker = this.landmarker;
    if (!landmarker) return;
    this.inFlight = true;
    try {
      const frame = this.camera.drawAnalysisFrame();
      if (
        frame.width !== GAZE_INPUT_SIZE.width ||
        frame.height !== GAZE_INPUT_SIZE.height
      ) {
        throw new Error('Gaze input must be the camera 320x240 analysis canvas');
      }
      const timestampMs = this.nextMediaTimestamp();
      const result = landmarker.detectForVideo(frame, timestampMs);
      const hands = this.handLandmarker?.detectForVideo(frame, timestampMs) ?? null;
      if (isPromiseLike(result) || isPromiseLike(hands)) {
        void Promise.all([result, hands]).then(
          ([faceValue, handValue]) => this.completeDetection(faceValue, handValue, generation, landmarker),
          (value: unknown) => this.failDetection(value, generation, landmarker),
        );
      } else {
        this.completeDetection(result, hands, generation, landmarker);
      }
    } catch (value) {
      this.failDetection(value, generation, landmarker);
    }
  }

  private completeDetection(
    result: GazeLandmarkerResult,
    hands: HandTrackingResult | null,
    generation: number,
    landmarker: GazeLandmarker,
  ): void {
    if (generation !== this.generation || landmarker !== this.landmarker) return;
    this.completed = result;
    this.completedHands = hands;
    this.completedReady = true;
    this.inFlight = false;
  }

  private failDetection(
    value: unknown,
    generation: number,
    landmarker: GazeLandmarker,
  ): void {
    if (generation !== this.generation || landmarker !== this.landmarker) return;
    this.completed = null;
    this.completedHands = null;
    this.completedReady = true;
    this.inFlight = false;
    this.reportError(toError(value));
  }

  private publishResult(
    result: GazeLandmarkerResult | null,
    hands: HandTrackingResult | null,
    t: number,
  ): void {
    const landmarks = result?.faceLandmarks[0];
    const matrix = result?.facialTransformationMatrixes[0];
    if (!landmarks || landmarks.length === 0 || !matrix) {
      this.ema.reset();
      this.publishDebugFrame(t, null, hands);
      this.publish(this.withHandFact(absentGazeMessage(t), hands));
      return;
    }
    try {
      const orientation = this.ema.update(extractFaceOrientation(matrix));
      const centroid = faceCentroid(landmarks);
      this.publishDebugFrame(t, {
        kind: 'face',
        index: 0,
        bounds: normalizedLandmarkBounds(landmarks),
        centroid,
        confidence: acceptedFaceConfidence(landmarks),
      }, hands);
      const spec = this.camera.cameraSpec;
      const target = targetAnglesFromCentroid(centroid, spec.hfov_deg, spec.w, spec.h);
      this.publish(this.withHandFact({
        type: 'gaze',
        t,
        present: true,
        yaw_deg: clamp(orientation.yawDeg, -180, 180),
        pitch_deg: clamp(orientation.pitchDeg, -180, 180),
        az: clamp(target.az, -Math.PI, Math.PI),
        el: clamp(target.el, -Math.PI / 2, Math.PI / 2),
        conf: acceptedFaceConfidence(landmarks),
      }, hands));
    } catch (value) {
      this.ema.reset();
      this.publishDebugFrame(t, null, hands);
      this.publish(this.withHandFact(absentGazeMessage(t), hands));
      this.reportError(toError(value));
    }
  }

  private async createTrackingModels(): Promise<{
    face: GazeLandmarker;
    hands: HandTrackingLandmarker | null;
  }> {
    const face = await this.createLandmarker(this.configuration);
    if (!this.handTrackingEnabled) return { face, hands: null };
    try {
      const hands = await this.createHandLandmarker(this.handConfiguration);
      return { face, hands };
    } catch (error) {
      face.close();
      throw error;
    }
  }

  private withHandFact(message: GazeMessage, result: HandTrackingResult | null): GazeMessage {
    if (!this.handTrackingEnabled) return message;
    const centroid = handsCentroid(result?.landmarks ?? []);
    if (!centroid) {
      return { ...message, hands_present: false, hand_az: 0, hand_el: 0, hand_conf: 0 };
    }
    const spec = this.camera.cameraSpec;
    const target = targetAnglesFromCentroid(centroid, spec.hfov_deg, spec.w, spec.h);
    return {
      ...message,
      hands_present: true,
      hand_az: clamp(target.az, -Math.PI, Math.PI),
      hand_el: clamp(target.el, -Math.PI / 2, Math.PI / 2),
      hand_conf: acceptedHandConfidence(result),
    };
  }

  private publishDebugFrame(
    t: number,
    face: LocalVisionDetection | null,
    result: HandTrackingResult | null,
  ): void {
    if (!this.onDebugFrame) return;
    try {
      const handLandmarks = result?.landmarks ?? [];
      const hands = handLandmarks.flatMap((landmarks, index): LocalVisionDetection[] =>
        landmarks.length === 0 ? [] : [{
          kind: 'hand',
          index,
          bounds: normalizedLandmarkBounds(landmarks),
          centroid: faceCentroid(landmarks),
          confidence: acceptedHandConfidence({
            landmarks: [landmarks],
            handedness: result?.handedness[index] ? [result.handedness[index]] : [],
          }),
        }]);
      this.onDebugFrame(Object.freeze({
        t,
        face,
        hands: Object.freeze(hands),
        handCentroid: handsCentroid(handLandmarks.filter((landmarks) => landmarks.length > 0)),
      }));
    } catch (value) {
      this.reportError(toError(value));
    }
  }

  private nextMediaTimestamp(): number {
    const now = finite(this.clock.nowMs(), 'clock.nowMs');
    this.lastMediaTimestampMs = Math.max(now, this.lastMediaTimestampMs + 0.001);
    return this.lastMediaTimestampMs;
  }

  private reportError(error: Error): void {
    this.onError?.(error);
  }
}

async function createDefaultLandmarker(
  configuration: FaceLandmarkerConfiguration,
): Promise<GazeLandmarker> {
  const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
  const fileset = await FilesetResolver.forVisionTasks(configuration.wasmBasePath);
  const landmarker = await FaceLandmarker.createFromOptions(
    fileset,
    configuration.options,
  );
  return {
    detectForVideo: (frame, timestampMs) =>
      landmarker.detectForVideo(frame, timestampMs),
    close: () => landmarker.close(),
  };
}

async function createDefaultHandLandmarker(
  configuration: HandLandmarkerConfiguration,
): Promise<HandTrackingLandmarker> {
  const { HandLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
  const fileset = await FilesetResolver.forVisionTasks(configuration.wasmBasePath);
  const landmarker = await HandLandmarker.createFromOptions(fileset, configuration.options);
  return {
    detectForVideo: (frame, timestampMs) => landmarker.detectForVideo(frame, timestampMs),
    close: () => landmarker.close(),
  };
}

function requireLocalAssetPath(value: string, label: string): void {
  if (!value.startsWith('/') || value.startsWith('//') || /^(?:https?:)?\/\//i.test(value)) {
    throw new Error(`${label} must be a local root-relative path`);
  }
}

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return typeof (value as Promise<T>)?.then === 'function';
}

function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function validHorizontalFov(value: number): number {
  const fov = finite(value, 'horizontalFovDegrees');
  if (fov <= 0 || fov >= 180) throw new RangeError('Horizontal FOV must be between 0 and 180 degrees');
  return fov;
}

function safeProtocolSeconds(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

function positiveFinite(value: number, label: string): number {
  const number = finite(value, label);
  if (number <= 0) throw new RangeError(`${label} must be greater than zero`);
  return number;
}

function finite(value: number | undefined, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.min(Math.max(value, lower), upper);
}

function degreesToRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function radiansToDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

// Manual integration: on localhost, validate GPU model loading and direction
// signs against a real camera. Static validation cannot claim that hardware pass.
