import type { CameraSpec } from '../protocol/types';

export const ANALYSIS_SIZE = Object.freeze({ width: 320, height: 240 });
export const CAPTURE_LONGEST_EDGE = 512;
export const JPEG_QUALITY = 0.8;
export const JPEG_MAX_BYTES = 60 * 1024;

// Keep the specified 512px output. Only quality is reduced, in bounded steps,
// when a noisy q80 frame would exceed the transport budget.
export const JPEG_QUALITY_ATTEMPTS = Object.freeze([0.8, 0.72, 0.64, 0.56]);

export const CAMERA_CONSTRAINTS = Object.freeze({
  audio: false,
  video: Object.freeze({
    width: Object.freeze({ ideal: 640 }),
    height: Object.freeze({ ideal: 480 }),
    facingMode: 'user',
  }),
}) satisfies MediaStreamConstraints;

export interface FrameSize {
  readonly width: number;
  readonly height: number;
}

export interface CameraSensorOptions {
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createVideo?: () => HTMLVideoElement;
  createCanvas?: () => HTMLCanvasElement;
  onError?: (error: Error) => void;
}

export function fitLongestEdge(
  sourceWidth: number,
  sourceHeight: number,
  longestEdge = CAPTURE_LONGEST_EDGE,
): FrameSize {
  if (
    !Number.isFinite(sourceWidth) ||
    !Number.isFinite(sourceHeight) ||
    sourceWidth <= 0 ||
    sourceHeight <= 0 ||
    !Number.isFinite(longestEdge) ||
    longestEdge <= 0
  ) {
    throw new RangeError('Frame dimensions must be positive finite numbers');
  }
  const scale = Math.min(1, longestEdge / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

export function isWithinJpegBudget(
  byteLength: number,
  maximum = JPEG_MAX_BYTES,
): boolean {
  return Number.isInteger(byteLength) && byteLength >= 0 && byteLength <= maximum;
}

export function bestHorizontalFov(...sources: readonly unknown[]): number {
  const keys = ['horizontalFieldOfView', 'fieldOfView', 'hfov', 'hfov_deg'];
  for (const source of sources) {
    if (!source || typeof source !== 'object') continue;
    const record = source as Record<string, unknown>;
    for (const key of keys) {
      const value = record[key];
      if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
        return value;
      }
    }
  }
  return 60;
}

export class CameraCaptureBudgetError extends Error {
  constructor(readonly lastByteLength: number) {
    super(`JPEG capture exceeds ${JPEG_MAX_BYTES} bytes after bounded fallback`);
    this.name = 'CameraCaptureBudgetError';
  }
}

class CameraStartCancelledError extends Error {}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop();
}

function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

async function encodeJpeg(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Browser failed to encode camera frame as JPEG'));
    }, 'image/jpeg', quality);
  });
}

export class CameraSensor {
  private readonly getUserMedia: CameraSensorOptions['getUserMedia'];
  private readonly createVideo: () => HTMLVideoElement;
  private readonly createCanvas: () => HTMLCanvasElement;
  private readonly onError?: (error: Error) => void;
  private readonly videoElement: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private analysisCanvas: HTMLCanvasElement | null = null;
  private startPromise: Promise<CameraSpec> | null = null;
  private generation = 0;
  private spec: CameraSpec | null = null;

  constructor(options: CameraSensorOptions = {}) {
    this.getUserMedia =
      options.getUserMedia ??
      ((constraints) => {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera capture is unavailable in this browser');
        }
        return navigator.mediaDevices.getUserMedia(constraints);
      });
    this.createVideo = options.createVideo ?? (() => document.createElement('video'));
    this.createCanvas = options.createCanvas ?? (() => document.createElement('canvas'));
    this.onError = options.onError;
    this.videoElement = this.createVideo();
    this.videoElement.autoplay = true;
    this.videoElement.muted = true;
    this.videoElement.playsInline = true;
    this.videoElement.hidden = true;
  }

  get video(): HTMLVideoElement {
    return this.videoElement;
  }

  get cameraSpec(): CameraSpec {
    if (!this.spec) throw new Error('Camera is not live');
    return this.spec;
  }

  get live(): boolean {
    return this.stream !== null;
  }

  start(): Promise<CameraSpec> {
    if (this.spec && this.stream) return Promise.resolve(this.spec);
    if (this.startPromise) return this.startPromise;
    const generation = ++this.generation;
    const pending = this.acquire(generation).catch((value: unknown) => {
      const error = toError(value);
      if (!(error instanceof CameraStartCancelledError)) this.onError?.(error);
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
    const stream = this.stream;
    this.stream = null;
    this.spec = null;
    this.videoElement.srcObject = null;
    if (stream) stopTracks(stream);
    if (this.analysisCanvas) {
      this.analysisCanvas.width = 0;
      this.analysisCanvas.height = 0;
      this.analysisCanvas = null;
    }
  }

  drawAnalysisFrame(): HTMLCanvasElement {
    this.assertDrawable();
    const canvas = this.analysisCanvas ?? this.createCanvas();
    canvas.width = ANALYSIS_SIZE.width;
    canvas.height = ANALYSIS_SIZE.height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('2D canvas is unavailable for camera analysis');
    context.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
    this.analysisCanvas = canvas;
    return canvas;
  }

  async captureJpeg(): Promise<Blob> {
    this.assertDrawable();
    const size = fitLongestEdge(
      this.videoElement.videoWidth,
      this.videoElement.videoHeight,
    );
    const canvas = this.createCanvas();
    canvas.width = size.width;
    canvas.height = size.height;
    try {
      const context = canvas.getContext('2d');
      if (!context) throw new Error('2D canvas is unavailable for camera capture');
      context.drawImage(this.videoElement, 0, 0, size.width, size.height);
      let lastByteLength = 0;
      for (const quality of JPEG_QUALITY_ATTEMPTS) {
        const blob = await encodeJpeg(canvas, quality);
        lastByteLength = blob.size;
        if (isWithinJpegBudget(blob.size)) return blob;
      }
      throw new CameraCaptureBudgetError(lastByteLength);
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  private async acquire(generation: number): Promise<CameraSpec> {
    let stream: MediaStream | null = null;
    try {
      stream = await this.getUserMedia!(CAMERA_CONSTRAINTS);
      if (generation !== this.generation) {
        throw new CameraStartCancelledError('Camera start was cancelled');
      }
      const [track] = stream.getVideoTracks();
      if (!track) throw new Error('Camera stream contains no video track');
      this.videoElement.srcObject = stream;
      await this.waitForMetadata();
      await this.videoElement.play();
      const settings = track.getSettings();
      const width = this.videoElement.videoWidth || settings.width || 0;
      const height = this.videoElement.videoHeight || settings.height || 0;
      if (width <= 0 || height <= 0) throw new Error('Camera reported zero dimensions');
      if (generation !== this.generation) throw new CameraStartCancelledError();
      this.stream = stream;
      this.spec = {
        w: width,
        h: height,
        hfov_deg: bestHorizontalFov(settings, track.getCapabilities?.()),
      };
      track.addEventListener('ended', this.handleTrackEnded, { once: true });
      return this.spec;
    } catch (error) {
      if (stream && this.stream !== stream) stopTracks(stream);
      if (this.videoElement.srcObject === stream) this.videoElement.srcObject = null;
      throw error;
    }
  }

  private waitForMetadata(): Promise<void> {
    if (this.videoElement.readyState >= 1) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      this.videoElement.addEventListener('loadedmetadata', () => resolve(), { once: true });
      this.videoElement.addEventListener(
        'error',
        () => reject(new Error('Camera video element failed to load')),
        { once: true },
      );
    });
  }

  private readonly handleTrackEnded = (): void => {
    if (!this.stream) return;
    const error = new Error('Camera device ended');
    this.stop();
    this.onError?.(error);
  };

  private assertDrawable(): void {
    if (!this.stream || !this.spec) throw new Error('Camera is not live');
    if (this.videoElement.videoWidth <= 0 || this.videoElement.videoHeight <= 0) {
      throw new Error('Camera has no drawable frame');
    }
  }
}

// Manual verification: grant/deny camera permission on localhost, unplug the
// active device, and confirm the browser permission indicator clears on stop.
