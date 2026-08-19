import type {
  FaceLandmarkerOptions,
  FaceLandmarkerResult,
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

export interface FaceLandmarkerConfiguration {
  readonly wasmBasePath: string;
  readonly options: FaceLandmarkerOptions;
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
  readonly clock?: GazeClock;
  readonly scheduler?: GazeScheduler;
  readonly wasmBasePath?: string;
  readonly modelAssetPath?: string;
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

export const DEFAULT_FACE_LANDMARKER_CONFIGURATION = Object.freeze(
  createFaceLandmarkerConfiguration(),
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

/** Image-right maps to positive azimuth; image-down maps to positive elevation. */
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
    az: Math.atan(2 * (x - 0.5) * Math.tan(hfov / 2)),
    // Luxo's positive head pitch looks down, so image-down must stay positive.
    el: Math.atan(2 * (y - 0.5) * Math.tan(vfov / 2)),
    vfovRad: vfov,
  };
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
  private readonly clock: GazeClock;
  private readonly scheduler: GazeScheduler;
  private readonly configuration: FaceLandmarkerConfiguration;
  private readonly onError?: (error: Error) => void;
  private readonly ema = new GazeEma();
  private landmarker: GazeLandmarker | null = null;
  private timer: unknown = null;
  private startPromise: Promise<void> | null = null;
  private inFlight = false;
  private completed: GazeLandmarkerResult | null = null;
  private completedReady = false;
  private generation = 0;
  private disposed = false;
  private lastMediaTimestampMs = -Infinity;

  constructor(options: GazeSensorOptions) {
    this.camera = options.camera;
    this.publish = options.publish;
    this.createLandmarker = options.createLandmarker ?? createDefaultLandmarker;
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
    const pending = this.createLandmarker(this.configuration)
      .then((landmarker) => {
        if (generation !== this.generation || this.disposed || !this.camera.live) {
          landmarker.close();
          throw new Error('Gaze start was cancelled');
        }
        this.landmarker = landmarker;
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
    this.inFlight = false;
    this.completed = null;
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
      this.completed = null;
      this.completedReady = false;
      this.publishResult(result, t);
    } else {
      this.ema.reset();
      this.publish(absentGazeMessage(t));
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
      if (isPromiseLike(result)) {
        void result.then(
          (value) => this.completeDetection(value, generation, landmarker),
          (value: unknown) => this.failDetection(value, generation, landmarker),
        );
      } else {
        this.completeDetection(result, generation, landmarker);
      }
    } catch (value) {
      this.failDetection(value, generation, landmarker);
    }
  }

  private completeDetection(
    result: GazeLandmarkerResult,
    generation: number,
    landmarker: GazeLandmarker,
  ): void {
    if (generation !== this.generation || landmarker !== this.landmarker) return;
    this.completed = result;
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
    this.completedReady = true;
    this.inFlight = false;
    this.reportError(toError(value));
  }

  private publishResult(result: GazeLandmarkerResult | null, t: number): void {
    const landmarks = result?.faceLandmarks[0];
    const matrix = result?.facialTransformationMatrixes[0];
    if (!landmarks || landmarks.length === 0 || !matrix) {
      this.ema.reset();
      this.publish(absentGazeMessage(t));
      return;
    }
    try {
      const orientation = this.ema.update(extractFaceOrientation(matrix));
      const centroid = faceCentroid(landmarks);
      const spec = this.camera.cameraSpec;
      const target = targetAnglesFromCentroid(centroid, spec.hfov_deg, spec.w, spec.h);
      this.publish({
        type: 'gaze',
        t,
        present: true,
        yaw_deg: clamp(orientation.yawDeg, -180, 180),
        pitch_deg: clamp(orientation.pitchDeg, -180, 180),
        az: clamp(target.az, -Math.PI, Math.PI),
        el: clamp(target.el, -Math.PI / 2, Math.PI / 2),
        conf: acceptedFaceConfidence(landmarks),
      });
    } catch (value) {
      this.ema.reset();
      this.publish(absentGazeMessage(t));
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
