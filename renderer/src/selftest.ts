export const CHECK_DEFINITIONS = Object.freeze([
  { id: 'secure', title: 'Secure localhost', detail: 'Requires a loopback secure context.' },
  { id: 'camera', title: 'Camera + frame', detail: 'Requests camera access and captures one local JPEG.' },
  { id: 'microphone', title: 'Microphone + level', detail: 'Requests microphone access and measures local RMS level.' },
  { id: 'webgl', title: 'WebGL', detail: 'Creates and releases a browser graphics context.' },
  { id: 'mediapipe', title: 'Face Landmarker', detail: 'Loads the local MediaPipe WASM and model assets.' },
  { id: 'silero', title: 'Silero VAD', detail: 'Loads local ONNX/WASM and runs one silent frame.' },
  { id: 'audio', title: 'Audio output', detail: 'Use Play tone for a user-gesture-gated local sample.', gesture: true },
  { id: 'websocket', title: 'Core socket', detail: 'Connects to ws://127.0.0.1:8765 and sends only hello.' },
] as const);

export type CheckId = (typeof CHECK_DEFINITIONS)[number]['id'];
export type CheckStatus = 'pending' | 'pass' | 'fail';

export interface CheckState {
  readonly id: CheckId;
  readonly title: string;
  readonly status: CheckStatus;
  readonly detail: string;
  readonly level?: number;
}

export interface CheckContext {
  readonly signal: AbortSignal;
  report(detail: string, level?: number): void;
}

export type CheckRunner = (context: CheckContext) => Promise<string>;
export type CheckRunners = Readonly<Record<CheckId, CheckRunner>>;

function errorDetail(value: unknown): string {
  if (value instanceof DOMException) return `${value.name}: ${value.message}`;
  return value instanceof Error ? value.message : String(value);
}

function stoppedError(): DOMException {
  return new DOMException('Check stopped', 'AbortError');
}

export function isSecureLoopback(secure: boolean, hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase();
  return secure && (
    normalized === 'localhost' ||
    normalized === '::1' ||
    normalized === '[::1]' ||
    /^127(?:\.\d{1,3}){3}$/.test(normalized)
  );
}

export function rmsLevel(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const sample of samples) {
    const finite = Number.isFinite(sample) ? Math.max(-1, Math.min(1, sample)) : 0;
    sum += finite * finite;
  }
  return Math.max(0, Math.min(1, Math.sqrt(sum / samples.length)));
}

export function withDeadline<T>(
  operation: Promise<T>,
  milliseconds: number,
  signal: AbortSignal,
): Promise<T> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeout: ReturnType<typeof globalThis.setTimeout>;
    const cleanup = () => {
      globalThis.clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
    };
    const resolveOnce = (value: T) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };
    const rejectOnce = (error: unknown) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const abort = () => rejectOnce(stoppedError());
    timeout = globalThis.setTimeout(
      () => rejectOnce(new Error(`Check timed out after ${milliseconds} ms`)),
      milliseconds,
    );
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    operation.then(resolveOnce, rejectOnce);
  });
}

export class SelfTestController {
  private readonly states = new Map<CheckId, CheckState>();
  private readonly active = new Map<CheckId, {
    token: number;
    abort: AbortController;
    promise: Promise<CheckState>;
  }>();
  private readonly versions = new Map<CheckId, number>();
  private readonly listeners = new Set<(states: readonly CheckState[]) => void>();

  constructor(private readonly runners: CheckRunners) {
    for (const check of CHECK_DEFINITIONS) {
      this.states.set(check.id, {
        id: check.id,
        title: check.title,
        status: 'pending',
        detail: check.detail,
      });
    }
  }

  snapshot(): readonly CheckState[] {
    return CHECK_DEFINITIONS.map((check) => ({ ...this.states.get(check.id)! }));
  }

  subscribe(listener: (states: readonly CheckState[]) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot());
    return () => this.listeners.delete(listener);
  }

  run(id: CheckId): Promise<CheckState> {
    const current = this.active.get(id);
    if (current) return current.promise;
    const token = (this.versions.get(id) ?? 0) + 1;
    this.versions.set(id, token);
    const abort = new AbortController();
    this.update(id, 'pending', 'Checking…');
    const context: CheckContext = {
      signal: abort.signal,
      report: (detail, level) => {
        if (this.versions.get(id) === token) this.update(id, 'pending', detail, level);
      },
    };
    const promise = Promise.resolve()
      .then(() => this.runners[id](context))
      .then(
        (detail) => this.finish(id, token, 'pass', detail),
        (error: unknown) => {
          if (abort.signal.aborted) return this.finish(id, token, 'pending', 'Stopped; resources released.');
          return this.finish(id, token, 'fail', errorDetail(error));
        },
      )
      .finally(() => {
        if (this.active.get(id)?.token === token) this.active.delete(id);
      });
    this.active.set(id, { token, abort, promise });
    return promise;
  }

  async runAll(): Promise<readonly CheckState[]> {
    const automatic = CHECK_DEFINITIONS.filter((check) => !('gesture' in check));
    await Promise.all(automatic.map((check) => this.run(check.id)));
    return this.snapshot();
  }

  stop(): void {
    for (const [id, running] of this.active) {
      this.versions.set(id, running.token + 1);
      running.abort.abort();
      this.active.delete(id);
      this.update(id, 'pending', 'Stopped; resources released.');
    }
  }

  private finish(
    id: CheckId,
    token: number,
    status: CheckStatus,
    detail: string,
  ): CheckState {
    if (this.versions.get(id) === token) this.update(id, status, detail);
    return { ...this.states.get(id)! };
  }

  private update(
    id: CheckId,
    status: CheckStatus,
    detail: string,
    level?: number,
  ): void {
    const previous = this.states.get(id)!;
    this.states.set(id, { id, title: previous.title, status, detail, level });
    const snapshot = this.snapshot();
    for (const listener of this.listeners) listener(snapshot);
  }
}

async function secureCheck(): Promise<string> {
  if (!isSecureLoopback(globalThis.isSecureContext, globalThis.location.hostname)) {
    throw new Error('Open this page from localhost or 127.0.0.1 in a secure context');
  }
  return `${globalThis.location.origin} is an accepted loopback secure context.`;
}

async function cameraCheck({ signal }: CheckContext): Promise<string> {
  const { CameraSensor } = await import('./sensors/camera');
  const camera = new CameraSensor();
  const stop = () => camera.stop();
  signal.addEventListener('abort', stop, { once: true });
  try {
    const spec = await camera.start();
    if (signal.aborted) throw stoppedError();
    const jpeg = await camera.captureJpeg();
    return `${spec.w}×${spec.h}; one ${jpeg.size}-byte JPEG captured locally and discarded.`;
  } finally {
    signal.removeEventListener('abort', stop);
    camera.stop();
  }
}

async function microphoneCheck(context: CheckContext): Promise<string> {
  const { openBrowserMicrophone } = await import('./sensors/mic');
  let resolveSamples!: () => void;
  let rejectSamples!: (error: Error) => void;
  let samplesSeen = 0;
  let peak = 0;
  const heard = new Promise<void>((resolve, reject) => {
    resolveSamples = resolve;
    rejectSamples = reject;
  });
  const session = await openBrowserMicrophone(
    (samples) => {
      peak = Math.max(peak, rmsLevel(samples));
      samplesSeen += samples.length;
      context.report(`Listening locally · level ${Math.round(peak * 100)}%`, peak);
      if (samplesSeen >= 8_192) resolveSamples();
    },
    (error) => rejectSamples(error),
  );
  const stop = () => { void session.close(); };
  context.signal.addEventListener('abort', stop, { once: true });
  try {
    if (context.signal.aborted) throw stoppedError();
    await withDeadline(heard, 3_000, context.signal);
    return `${session.sampleRate} Hz device path; live RMS peaked at ${Math.round(peak * 100)}%.`;
  } finally {
    context.signal.removeEventListener('abort', stop);
    await session.close();
  }
}

async function webglCheck(): Promise<string> {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
  if (!context) throw new Error('WebGL context creation failed');
  const version = context.getParameter(context.VERSION) as string;
  context.getExtension('WEBGL_lose_context')?.loseContext();
  canvas.width = 0;
  canvas.height = 0;
  return `${version}; temporary context released.`;
}

async function mediaPipeCheck({ signal }: CheckContext): Promise<string> {
  const { createFaceLandmarkerConfiguration } = await import('./sensors/gaze');
  const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
  const configuration = createFaceLandmarkerConfiguration();
  const fileset = await FilesetResolver.forVisionTasks(configuration.wasmBasePath);
  if (signal.aborted) throw stoppedError();
  const landmarker = await FaceLandmarker.createFromOptions(fileset, configuration.options);
  landmarker.close();
  return '/models/face_landmarker.task loaded through /mediapipe/wasm.';
}

async function sileroCheck({ signal }: CheckContext): Promise<string> {
  const { SileroVadScorer, SILERO_FRAME_SAMPLES } = await import('./sensors/vad');
  const scorer = await SileroVadScorer.create();
  try {
    if (signal.aborted) throw stoppedError();
    await scorer.score(new Float32Array(SILERO_FRAME_SAMPLES));
    return '/models/silero_vad.onnx loaded through /onnxruntime/wasm/.';
  } finally {
    await scorer.dispose();
  }
}

async function audioCheck({ signal }: CheckContext): Promise<string> {
  const context = new AudioContext();
  try {
    await context.resume();
    if (signal.aborted) throw stoppedError();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 440;
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.12, context.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.22);
    oscillator.connect(gain).connect(context.destination);
    const ended = new Promise<void>((resolve) => { oscillator.onended = () => resolve(); });
    oscillator.start();
    oscillator.stop(context.currentTime + 0.23);
    await withDeadline(ended, 1_000, signal);
    oscillator.disconnect();
    gain.disconnect();
    return 'Local 440 Hz sample tone played through the browser output.';
  } finally {
    await context.close().catch(() => undefined);
  }
}

async function websocketCheck({ signal }: CheckContext): Promise<string> {
  const { ProtocolClient } = await import('./protocol/client');
  let resolveConnected!: () => void;
  let rejectConnected!: (error: Error) => void;
  const connected = new Promise<void>((resolve, reject) => {
    resolveConnected = resolve;
    rejectConnected = reject;
  });
  const client = new ProtocolClient({
    url: 'ws://127.0.0.1:8765',
    onConnected: resolveConnected,
    onError: rejectConnected,
  });
  const stop = () => client.disconnect();
  signal.addEventListener('abort', stop, { once: true });
  try {
    client.connect();
    await withDeadline(connected, 2_000, signal);
    return 'Connected to ws://127.0.0.1:8765; normal hello sent.';
  } finally {
    signal.removeEventListener('abort', stop);
    client.disconnect();
  }
}

export function createBrowserChecks(): CheckRunners {
  return {
    secure: () => secureCheck(),
    camera: cameraCheck,
    microphone: microphoneCheck,
    webgl: () => webglCheck(),
    mediapipe: mediaPipeCheck,
    silero: sileroCheck,
    audio: audioCheck,
    websocket: websocketCheck,
  };
}

export function mountSelfTest(root: HTMLElement): SelfTestController {
  const controller = new SelfTestController(createBrowserChecks());
  const cards = new Map<CheckId, { card: HTMLElement; status: HTMLElement; detail: HTMLElement; meter?: HTMLElement }>();
  for (const check of CHECK_DEFINITIONS) {
    const card = document.createElement('article');
    card.dataset.status = 'pending';
    card.innerHTML = `<div class="card-top"><h2></h2><span class="status">pending</span></div><p class="detail"></p>`;
    card.querySelector('h2')!.textContent = check.title;
    const status = card.querySelector<HTMLElement>('.status')!;
    const detail = card.querySelector<HTMLElement>('.detail')!;
    let meter: HTMLElement | undefined;
    if (check.id === 'microphone') {
      card.insertAdjacentHTML('beforeend', '<div class="meter" aria-label="Microphone level"><span></span></div>');
      meter = card.querySelector<HTMLElement>('.meter > span')!;
    }
    if ('gesture' in check) {
      const button = document.createElement('button');
      button.className = 'card-action';
      button.type = 'button';
      button.textContent = 'Play tone';
      button.addEventListener('click', () => { void controller.run('audio'); });
      card.append(button);
    }
    root.append(card);
    cards.set(check.id, { card, status, detail, meter });
  }
  controller.subscribe((states) => {
    for (const state of states) {
      const elements = cards.get(state.id)!;
      elements.card.dataset.status = state.status;
      elements.status.textContent = state.status;
      elements.detail.textContent = state.detail;
      if (elements.meter) elements.meter.style.width = `${Math.round((state.level ?? 0) * 100)}%`;
    }
    const passed = states.filter((state) => state.status === 'pass').length;
    const failed = states.filter((state) => state.status === 'fail').length;
    document.querySelector('#summary')!.textContent = `${passed} passed · ${failed} failed · ${states.length - passed - failed} pending`;
  });
  return controller;
}

if (typeof document !== 'undefined') {
  const root = document.querySelector<HTMLElement>('#checks');
  const run = document.querySelector<HTMLButtonElement>('#run');
  const stop = document.querySelector<HTMLButtonElement>('#stop');
  if (!root || !run || !stop) throw new Error('Preflight page is missing required elements');
  const controller = mountSelfTest(root);
  run.addEventListener('click', () => { void controller.runAll(); });
  stop.addEventListener('click', () => controller.stop());
  globalThis.addEventListener('beforeunload', () => controller.stop(), { once: true });
}

// Manual validation: run on localhost, exercise allow/deny permissions, hear
// the tone, stop mid-check, and confirm browser camera/mic indicators clear.
