import { mountRenderer, type RendererHandle } from "./app";
import { BrowserAudioMixer, type BrowserAudioMixerOptions } from "./audio/mixer";
import { CORE_DISCONNECT_DRIFT_SECONDS, RendererDisconnectFallback } from "./degraded";
import { ProtocolClient, type ProtocolClientOptions } from "./protocol/client";
import type { BodyStateMessage, CameraSpec, GazeMessage, VadMessage } from "./protocol/types";
import { CameraSensor, type CameraSensorOptions } from "./sensors/camera";
import { GazeSensor, type GazeSensorOptions } from "./sensors/gaze";
import { MicrophoneCapture, type MicrophoneOptions, type VadAudioSink } from "./sensors/mic";
import { SileroVadScorer, VadProcessor, type VadOptions, type VadScorer } from "./sensors/vad";
import { mountTelemetryOverlay, type TelemetryOverlayHandle } from "./ui/overlay";

type MaybePromise = void | Promise<void>;

interface RuntimeCamera {
  readonly cameraSpec: CameraSpec;
  readonly live: boolean;
  start(): Promise<CameraSpec>;
  stop(): void;
  drawAnalysisFrame(): HTMLCanvasElement;
  captureJpeg(): Promise<Blob>;
}

interface RuntimeMixer {
  unlock(): Promise<void>;
  applyBodyState(state: Pick<BodyStateMessage, "audio">): void;
  playCue(cue: { readonly sfx: string }): void;
  speakBegin(message: { readonly type: "speak_begin"; readonly envelope_hz: number }): void;
  enqueueTtsPcm(chunk: Uint8Array): void;
  speakEnd(message: { readonly type: "speak_end" }): void;
  dispose(): void;
}

interface RuntimeProtocol {
  connect(): void;
  disconnect(): void;
  sendGaze(fact: Omit<GazeMessage, "type">): boolean;
  sendVad(event: VadMessage["event"], t?: number): boolean;
  sendTtsDone(t?: number): boolean;
  sendError(where: string, detail: string): boolean;
  sendUtterancePcm(payload: Uint8Array): boolean;
  sendCapturedJpeg(payload: ArrayBuffer | Uint8Array): boolean;
}

interface RuntimeGaze { start(): Promise<void>; dispose(): void; }
interface RuntimeMicrophone {
  start(): Promise<void>;
  setTtsPlaying(active: boolean): void;
  dispose(): Promise<void>;
}
interface RuntimeVad extends VadAudioSink { dispose(): Promise<void>; }
interface RuntimeFallback {
  readonly status: "connected" | "disconnected";
  readonly lastValidState: BodyStateMessage | null;
  acceptBodyState(state: BodyStateMessage): boolean;
  connect(): void;
  disconnect(atSeconds: number): boolean;
  sample(atSeconds: number): BodyStateMessage | null;
  dispose(): void;
}

export interface LuxoRuntimeDependencies {
  mountRenderer(root: HTMLElement): Promise<RendererHandle>;
  createCamera(options: CameraSensorOptions): RuntimeCamera;
  createMixer(options: BrowserAudioMixerOptions): RuntimeMixer;
  createScorer(): Promise<VadScorer>;
  createVad(options: VadOptions): RuntimeVad;
  createMicrophone(options: MicrophoneOptions): RuntimeMicrophone;
  createGaze(options: GazeSensorOptions): RuntimeGaze;
  createProtocol(options: ProtocolClientOptions): RuntimeProtocol;
  createOverlay(root: HTMLElement): TelemetryOverlayHandle;
  createFallback(): RuntimeFallback;
  nowSeconds(): number;
  requestFrame(callback: () => void): number;
  cancelFrame(handle: number): void;
  onError(error: Error): void;
}

const DEFAULT_DEPENDENCIES: LuxoRuntimeDependencies = {
  mountRenderer,
  createCamera: (options) => new CameraSensor(options),
  createMixer: (options) => new BrowserAudioMixer(options),
  createScorer: () => SileroVadScorer.create(),
  createVad: (options) => new VadProcessor(options),
  createMicrophone: (options) => new MicrophoneCapture(options),
  createGaze: (options) => new GazeSensor(options),
  createProtocol: (options) => new ProtocolClient(options),
  createOverlay: mountTelemetryOverlay,
  createFallback: () => new RendererDisconnectFallback(),
  nowSeconds: () => performance.now() / 1_000,
  requestFrame: (callback) => requestAnimationFrame(callback),
  cancelFrame: (handle) => cancelAnimationFrame(handle),
  onError: (error) => console.error(error),
};

function errorOf(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

export class LuxoBrowserRuntime {
  private readonly camera: RuntimeCamera;
  private readonly mixer: RuntimeMixer;
  private readonly fallback: RuntimeFallback;
  private readonly overlay: TelemetryOverlayHandle;
  private readonly prompt: HTMLElement;
  private readonly startButton: HTMLButtonElement;
  private readonly startStatus: HTMLElement;
  private protocol: RuntimeProtocol | null = null;
  private gaze: RuntimeGaze | null = null;
  private microphone: RuntimeMicrophone | null = null;
  private vad: RuntimeVad | null = null;
  private startPromise: Promise<void> | null = null;
  private frameHandle: number | null = null;
  private fallbackEndSeconds: number | null = null;
  private captureBusy = false;
  private generation = 0;
  private started = false;
  private destroyed = false;
  private promptCleared = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly renderer: RendererHandle,
    private readonly dependencies: LuxoRuntimeDependencies,
  ) {
    this.camera = dependencies.createCamera({ onError: (error) => this.report("camera", error) });
    this.mixer = dependencies.createMixer({
      setVadSuppressed: (active) => this.guard("tts", () => this.microphone?.setTtsPlaying(active)),
      onTtsDone: () => this.guard("tts", () => { this.protocol?.sendTtsDone(); }),
    });
    this.fallback = dependencies.createFallback();
    this.overlay = dependencies.createOverlay(root);
    this.overlay.setConnectionStatus("connecting");
    const documentRef = root.ownerDocument;
    this.prompt = documentRef.createElement("section");
    this.prompt.className = "lumen-start";
    this.prompt.setAttribute("aria-label", "Start Luxo sensors and audio");
    this.startButton = documentRef.createElement("button");
    this.startButton.type = "button";
    this.startButton.className = "lumen-start__button";
    this.startButton.textContent = "Start Luxo";
    this.startStatus = documentRef.createElement("p");
    this.startStatus.className = "lumen-start__status";
    this.startStatus.setAttribute("role", "status");
    this.startStatus.setAttribute("aria-live", "polite");
    this.startStatus.textContent = "Camera, microphone, and audio wait for your gesture.";
    this.startButton.addEventListener("click", this.handleStart);
    this.prompt.append(this.startButton, this.startStatus);
    root.append(this.prompt);
  }

  startFromGesture(): Promise<void> {
    if (this.destroyed) return Promise.reject(new Error("Luxo runtime has been destroyed"));
    if (this.started) return Promise.resolve();
    if (this.startPromise) return this.startPromise;
    this.startButton.disabled = true;
    this.startStatus.textContent = "Starting local sensors…";
    const generation = ++this.generation;
    let unlock: Promise<void>;
    let camera: Promise<CameraSpec>;
    let scorer: Promise<VadScorer>;
    let cameraStarted = false;
    try {
      unlock = this.mixer.unlock();
      cameraStarted = true;
      camera = this.camera.start();
      scorer = this.dependencies.createScorer();
    } catch (value) {
      if (cameraStarted) {
        try { this.camera.stop(); } catch (error) { this.report("startup_cleanup", errorOf(error)); }
      }
      return this.finishFailedStart(errorOf(value));
    }
    const pending = this.finishStart(generation, unlock, camera, scorer)
      .catch((value: unknown) => this.handleStartFailure(errorOf(value)))
      .finally(() => { if (this.startPromise === pending) this.startPromise = null; });
    this.startPromise = pending;
    return pending;
  }

  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.generation += 1;
    const starting = this.startPromise;
    let firstError: unknown;
    const release = async (operation: () => MaybePromise) => {
      try { await operation(); } catch (error) { firstError ??= error; }
    };
    this.cancelFallbackFrame();
    this.fallbackEndSeconds = null;
    await release(() => this.cleanupOperational());
    await release(() => this.camera.stop());
    await release(() => this.mixer.dispose());
    await release(() => this.fallback.dispose());
    await release(() => this.overlay.dispose());
    await release(() => this.renderer.destroy());
    await release(() => this.clearPrompt());
    await starting?.catch(() => undefined);
    if (firstError !== undefined) throw firstError;
  }

  async destroySafely(): Promise<void> {
    try { await this.destroy(); } catch (error) { this.dependencies.onError(errorOf(error)); }
  }

  private readonly handleStart = (): void => { void this.startFromGesture(); };

  private async finishStart(
    generation: number,
    unlockPromise: Promise<void>,
    cameraPromise: Promise<CameraSpec>,
    scorerPromise: Promise<VadScorer>,
  ): Promise<void> {
    const [audioResult, cameraResult, scorerResult] = await Promise.allSettled([
      unlockPromise, cameraPromise, scorerPromise,
    ]);
    if (
      audioResult.status === "rejected" || cameraResult.status === "rejected" ||
      scorerResult.status === "rejected" || generation !== this.generation || this.destroyed
    ) {
      const failure = audioResult.status === "rejected" ? audioResult.reason
        : cameraResult.status === "rejected" ? cameraResult.reason
          : scorerResult.status === "rejected" ? scorerResult.reason
            : new Error("Luxo startup was cancelled");
      if (scorerResult.status === "fulfilled") {
        try { await scorerResult.value.dispose?.(); }
        catch (error) { this.report("startup_cleanup", errorOf(error)); }
      }
      if (!this.destroyed) {
        try { this.camera.stop(); }
        catch (error) { this.report("startup_cleanup", errorOf(error)); }
      }
      throw failure;
    }
    const cameraSpec = cameraResult.value;
    const scorer = scorerResult.value;
    let scorerTransferred = false;
    try {
      this.protocol = this.dependencies.createProtocol(this.protocolOptions(cameraSpec, generation));
      const protocol = this.protocol;
      protocol.connect();
      this.vad = this.dependencies.createVad({
        scorer,
        publish: (message) => this.route(generation, "vad", () => { protocol.sendVad(message.event, message.t); }),
        onUtterance: (pcm) => this.route(generation, "vad", () => { protocol.sendUtterancePcm(pcm); }),
        onError: (error) => this.report("vad", error),
      });
      scorerTransferred = true;
      this.microphone = this.dependencies.createMicrophone({
        sink: this.vad,
        onError: (error) => this.report("microphone", error),
      });
      this.gaze = this.dependencies.createGaze({
        camera: this.camera as CameraSensor,
        publish: ({ type: _type, ...fact }) => this.route(generation, "gaze", () => { protocol.sendGaze(fact); }),
        onError: (error) => this.report("gaze", error),
      });
      const sensorResults = await Promise.allSettled([this.gaze.start(), this.microphone.start()]);
      const sensorFailure = sensorResults.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (sensorFailure) throw sensorFailure.reason;
      if (generation !== this.generation || this.destroyed) {
        throw new Error("Luxo startup was cancelled");
      }
      this.started = true;
      this.clearPrompt();
    } catch (error) {
      try { await this.cleanupOperational(); }
      catch (cleanupError) { this.report("startup_cleanup", errorOf(cleanupError)); }
      if (!scorerTransferred) {
        try { await scorer.dispose?.(); }
        catch (cleanupError) { this.report("startup_cleanup", errorOf(cleanupError)); }
      }
      if (!this.destroyed) {
        try { this.camera.stop(); }
        catch (cleanupError) { this.report("startup_cleanup", errorOf(cleanupError)); }
      }
      throw error;
    }
  }

  private protocolOptions(camera: CameraSpec, generation: number): ProtocolClientOptions {
    return {
      url: "ws://127.0.0.1:8765",
      hello: { type: "hello", fps: 60, camera },
      onConnected: () => this.route(generation, "protocol", () => this.handleConnected()),
      onDisconnected: () => this.route(generation, "protocol", () => this.handleDisconnected()),
      onBodyState: (state) => this.route(generation, "body_state", () => this.handleBodyState(state)),
      onCue: (cue) => this.route(generation, "audio", () => this.mixer.playCue(cue)),
      onCaptureFrame: () => this.route(generation, "camera", () => this.captureFrame()),
      onSpeakBegin: (message) => this.route(generation, "tts", () => this.mixer.speakBegin(message)),
      onTtsPcm: (pcm) => this.route(generation, "tts", () => this.mixer.enqueueTtsPcm(pcm)),
      onSpeakEnd: () => this.route(generation, "tts", () => this.mixer.speakEnd({ type: "speak_end" })),
      onError: (error) => this.route(generation, "protocol", () => this.report("protocol", error)),
    };
  }

  private handleBodyState(state: BodyStateMessage): void {
    if (!this.fallback.acceptBodyState(state) || !this.fallback.lastValidState) {
      throw new Error("Core sent an invalid or stale body_state");
    }
    this.applyBodyState(this.fallback.lastValidState);
  }

  private applyBodyState(state: BodyStateMessage): void {
    this.renderer.applyBodyState(state);
    this.mixer.applyBodyState(state);
    this.overlay.updateBodyState(state);
  }

  private handleConnected(): void {
    this.cancelFallbackFrame();
    this.fallbackEndSeconds = null;
    this.fallback.connect();
    this.overlay.setConnectionStatus("connected");
  }

  private handleDisconnected(): void {
    this.overlay.setConnectionStatus("disconnected");
    if (this.fallback.status === "disconnected") return;
    const now = this.dependencies.nowSeconds();
    if (this.fallback.disconnect(now)) {
      this.fallbackEndSeconds = now + CORE_DISCONNECT_DRIFT_SECONDS;
      const state = this.fallback.sample(now);
      if (state) this.applyBodyState(state);
      if (state && now < this.fallbackEndSeconds) this.scheduleFallbackFrame();
    }
  }

  private scheduleFallbackFrame(): void {
    if (this.frameHandle !== null || this.fallback.status !== "disconnected") return;
    this.frameHandle = this.dependencies.requestFrame(() => {
      this.frameHandle = null;
      if (this.destroyed || this.fallback.status !== "disconnected") return;
      const now = this.dependencies.nowSeconds();
      const state = this.fallback.sample(now);
      if (state) this.applyBodyState(state);
      if (this.fallbackEndSeconds !== null && now < this.fallbackEndSeconds) {
        this.scheduleFallbackFrame();
      }
    });
  }

  private cancelFallbackFrame(): void {
    if (this.frameHandle === null) return;
    this.dependencies.cancelFrame(this.frameHandle);
    this.frameHandle = null;
  }

  private captureFrame(): void {
    if (this.captureBusy) { this.report("camera", new Error("Camera capture already in progress")); return; }
    this.captureBusy = true;
    const generation = this.generation;
    void this.camera.captureJpeg()
      .then((blob) => blob.arrayBuffer())
      .then((jpeg) => { if (!this.destroyed && generation === this.generation) this.protocol?.sendCapturedJpeg(jpeg); })
      .catch((error: unknown) => this.report("camera", errorOf(error)))
      .finally(() => { this.captureBusy = false; });
  }

  private async cleanupOperational(): Promise<void> {
    const protocol = this.protocol;
    const gaze = this.gaze;
    const microphone = this.microphone;
    const vad = this.vad;
    this.protocol = null; this.gaze = null; this.microphone = null; this.vad = null;
    let first: unknown;
    const release = async (operation: () => MaybePromise) => {
      try { await operation(); } catch (error) { first ??= error; }
    };
    await release(() => protocol?.disconnect());
    await release(() => gaze?.dispose());
    await release(() => microphone?.dispose());
    await release(() => vad?.dispose());
    if (first !== undefined) throw first;
  }

  private finishFailedStart(error: Error): Promise<void> {
    const pending = this.handleStartFailure(error);
    this.startPromise = pending.finally(() => { this.startPromise = null; });
    return this.startPromise;
  }

  private async handleStartFailure(error: Error): Promise<void> {
    this.started = false;
    if (this.destroyed) return;
    this.startButton.disabled = false;
    this.startButton.textContent = "Retry Luxo";
    this.startStatus.textContent = error.message;
    this.report("startup", error);
  }

  private clearPrompt(): void {
    if (this.promptCleared) return;
    this.promptCleared = true;
    this.startButton.removeEventListener("click", this.handleStart);
    this.prompt.remove();
  }

  private guard(where: string, operation: () => void): void {
    try { operation(); } catch (error) { this.report(where, errorOf(error)); }
  }

  private route(generation: number, where: string, operation: () => void): void {
    if (!this.destroyed && generation === this.generation) this.guard(where, operation);
  }

  private report(where: string, error: Error): void {
    try { this.protocol?.sendError(where, error.message); } catch { /* local reporting still runs */ }
    try { this.dependencies.onError(error); } catch { /* error reporting is non-fatal */ }
  }
}

export async function mountLuxoBrowserRuntime(
  root: HTMLElement,
  overrides: Partial<LuxoRuntimeDependencies> = {},
): Promise<LuxoBrowserRuntime> {
  const dependencies = { ...DEFAULT_DEPENDENCIES, ...overrides };
  const renderer = await dependencies.mountRenderer(root);
  try {
    return new LuxoBrowserRuntime(root, renderer, dependencies);
  } catch (error) {
    try { renderer.destroy(); } catch { /* preserve construction error */ }
    throw error;
  }
}
