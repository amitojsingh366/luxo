import type { InferenceSession, Tensor } from 'onnxruntime-web/wasm';

import type { VadMessage } from '../protocol/types';
import { encodePcm16Le, MIC_SAMPLE_RATE } from './mic';

export const SILERO_FRAME_SAMPLES = 512;
export const SILERO_STATE_SAMPLES = 2 * 1 * 128;
export const SPEECH_THRESHOLD = 0.5;
export const END_SILENCE_SAMPLES = 4_800; // 300 ms at 16 kHz
export const MIN_UTTERANCE_SAMPLES = 6_400; // 400 ms at 16 kHz
export const MAX_UTTERANCE_SAMPLES = 30 * MIC_SAMPLE_RATE;
export const MAX_QUEUED_FRAMES = 64;
export const DEFAULT_SILERO_MODEL_PATH = '/models/silero_vad.onnx';
export const DEFAULT_ORT_WASM_PATH = '/onnxruntime/wasm/';
export const SILERO_SESSION_OPTIONS = Object.freeze({
  executionProviders: ['wasm'] as const,
  executionMode: 'sequential' as const,
  graphOptimizationLevel: 'all' as const,
  intraOpNumThreads: 1,
  interOpNumThreads: 1,
});

export interface VadScorer {
  score(frame: Float32Array): Promise<number>;
  reset(): void | Promise<void>;
  dispose?(): void | Promise<void>;
}

export interface VadOptions {
  readonly scorer: VadScorer;
  readonly publish: (message: VadMessage) => void;
  readonly onUtterance: (pcm: Uint8Array) => void;
  readonly nowSeconds?: () => number;
  readonly onError?: (error: Error) => void;
}

type OrtModule = typeof import('onnxruntime-web/wasm');

function requireLocal(path: string, name: string): string {
  if (!path.startsWith('/') || path.startsWith('//')) {
    throw new Error(`${name} must be a local root-relative path`);
  }
  return path;
}

/**
 * onnxruntime-web reaches its emscripten glue through `await import(url)` with a *variable*
 * specifier. Vite's import analysis rewrites every such dynamic import to
 * `__vite__injectQuery(url, 'import')`; the `@vite-ignore` comment ORT ships only suppresses the
 * warning, never the rewrite. A root-relative prefix therefore arrives at the dev server as
 * `/onnxruntime/wasm/ort-wasm-simd-threaded.mjs?import`, and that query is exactly what makes
 * Vite's public-file middleware stand aside so the transform pipeline can reject the file for
 * living in `/public`.
 *
 * `__vite__injectQuery` returns any specifier starting with neither `.` nor `/` untouched, so
 * resolving the staged prefix against the document origin hands ORT a URL Vite leaves alone. The
 * assets stay exactly where `setup.sh` stages them and `doctor.py` verifies them; only the
 * spelling of the prefix changes, and it stays same-origin so ORT still imports the glue directly
 * instead of falling back to its cross-origin blob preload. Without a document base (Node) there
 * is nothing to resolve against, so the path passes through unchanged.
 */
function resolveAgainstDocument(path: string): string {
  const base = globalThis.location?.href;
  return base ? new URL(path, base).href : path;
}

/** Thin stateful adapter for the 16 kHz Silero ONNX input contract. */
export class SileroVadScorer implements VadScorer {
  private readonly state = new Float32Array(SILERO_STATE_SAMPLES);

  private constructor(
    private readonly ort: OrtModule,
    private readonly session: InferenceSession,
  ) {}

  static async create(options: {
    readonly modelPath?: string;
    readonly wasmPath?: string;
    readonly loadRuntime?: () => Promise<OrtModule>;
  } = {}): Promise<SileroVadScorer> {
    const modelPath = requireLocal(options.modelPath ?? DEFAULT_SILERO_MODEL_PATH, 'modelPath');
    const wasmPath = requireLocal(options.wasmPath ?? DEFAULT_ORT_WASM_PATH, 'wasmPath');
    const ort = await (options.loadRuntime?.() ?? import('onnxruntime-web/wasm'));
    ort.env.wasm.wasmPaths = resolveAgainstDocument(wasmPath);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    // ORT hands the session options to `appendDefaultOptions`, which writes `extra.session` onto
    // them, and it only wraps the object in a `get` proxy, so the write lands on our frozen
    // constant and throws. Give it a mutable shallow copy and keep the exported contract frozen.
    const session = await ort.InferenceSession.create(modelPath, { ...SILERO_SESSION_OPTIONS });
    return new SileroVadScorer(ort, session);
  }

  async score(frame: Float32Array): Promise<number> {
    if (frame.length !== SILERO_FRAME_SAMPLES) {
      throw new RangeError(`Silero frame must contain ${SILERO_FRAME_SAMPLES} samples`);
    }
    const feeds = {
      input: new this.ort.Tensor('float32', frame, [1, SILERO_FRAME_SAMPLES]),
      state: new this.ort.Tensor('float32', this.state.slice(), [2, 1, 128]),
      sr: new this.ort.Tensor('int64', BigInt64Array.of(BigInt(MIC_SAMPLE_RATE)), [1]),
    };
    const outputs = await this.session.run(feeds);
    const probability = outputs.output as Tensor | undefined;
    const nextState = outputs.stateN as Tensor | undefined;
    const value = Number(probability?.data[0]);
    if (!Number.isFinite(value) || !(nextState?.data instanceof Float32Array) ||
      nextState.data.length !== SILERO_STATE_SAMPLES) {
      throw new Error('Silero returned invalid output or recurrent state');
    }
    this.state.set(nextState.data);
    return Math.max(0, Math.min(1, value));
  }

  reset(): void { this.state.fill(0); }
  async dispose(): Promise<void> { await this.session.release(); }
}

function join(chunks: readonly Float32Array[], length: number): Float32Array {
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    const count = Math.min(chunk.length, length - offset);
    result.set(chunk.subarray(0, count), offset);
    offset += count;
    if (offset === length) break;
  }
  return result;
}

export class VadProcessor {
  private pending = new Float32Array(0);
  private readonly queue: Float32Array[] = [];
  private speech: Float32Array[] = [];
  private utteranceSamples = 0;
  private lastVoiceSamples = 0;
  private silenceSamples = 0;
  private speaking = false;
  private ttsPlaying = false;
  private generation = 0;
  private resetModel = false;
  private draining = false;
  private idlePromise: Promise<void> = Promise.resolve();

  constructor(private readonly options: VadOptions) {}

  push16k(samples: Float32Array): void {
    if (this.ttsPlaying || samples.length === 0) return;
    const combined = new Float32Array(this.pending.length + samples.length);
    combined.set(this.pending);
    combined.set(samples, this.pending.length);
    let offset = 0;
    while (combined.length - offset >= SILERO_FRAME_SAMPLES) {
      if (this.queue.length >= MAX_QUEUED_FRAMES) {
        this.fail(new Error('VAD inference queue overflow'));
        return;
      }
      this.queue.push(combined.slice(offset, offset + SILERO_FRAME_SAMPLES));
      offset += SILERO_FRAME_SAMPLES;
    }
    this.pending = combined.slice(offset);
    this.ensureDrain();
  }

  setTtsPlaying(active: boolean): void {
    if (this.ttsPlaying === active) return;
    this.ttsPlaying = active;
    if (active) this.reset();
  }

  reset(): void {
    this.generation += 1;
    this.pending = new Float32Array(0);
    this.queue.length = 0;
    this.clearSpeech();
    this.resetModel = true;
    this.ensureDrain();
  }

  whenIdle(): Promise<void> { return this.idlePromise; }

  async dispose(): Promise<void> {
    this.reset();
    await this.whenIdle();
    await this.options.scorer.dispose?.();
  }

  private ensureDrain(): void {
    if (this.draining || (!this.resetModel && this.queue.length === 0)) return;
    this.draining = true;
    this.idlePromise = this.drain().finally(() => {
      this.draining = false;
      if (this.resetModel || this.queue.length) this.ensureDrain();
    });
  }

  private async drain(): Promise<void> {
    while (this.resetModel || this.queue.length) {
      if (this.resetModel) {
        this.resetModel = false;
        try { await this.options.scorer.reset(); } catch (error) { this.report(error); }
        continue;
      }
      const frame = this.queue.shift();
      if (!frame) continue;
      const generation = this.generation;
      try {
        const probability = await this.options.scorer.score(frame);
        if (generation === this.generation && !this.ttsPlaying) this.accept(frame, probability);
      } catch (error) {
        this.fail(error);
      }
    }
  }

  private accept(frame: Float32Array, probability: number): void {
    const voiced = probability >= SPEECH_THRESHOLD;
    if (!this.speaking) {
      if (!voiced) return;
      this.speaking = true;
      this.options.publish({ type: 'vad', event: 'start', t: this.safeTime() });
    }
    this.speech.push(frame);
    this.utteranceSamples += frame.length;
    if (voiced) {
      this.lastVoiceSamples = this.utteranceSamples;
      this.silenceSamples = 0;
    } else {
      this.silenceSamples += frame.length;
    }
    if (this.utteranceSamples > MAX_UTTERANCE_SAMPLES) {
      this.fail(new Error('VAD utterance exceeded 30 seconds'));
    } else if (this.silenceSamples >= END_SILENCE_SAMPLES) {
      if (this.lastVoiceSamples >= MIN_UTTERANCE_SAMPLES) {
        this.options.onUtterance(encodePcm16Le(join(this.speech, this.lastVoiceSamples)));
      }
      this.clearSpeech();
      this.resetModel = true;
    }
  }

  private clearSpeech(): void {
    this.speech = [];
    this.utteranceSamples = 0;
    this.lastVoiceSamples = 0;
    this.silenceSamples = 0;
    this.speaking = false;
  }

  private fail(value: unknown): void {
    this.generation += 1;
    this.pending = new Float32Array(0);
    this.queue.length = 0;
    this.clearSpeech();
    this.resetModel = true;
    this.report(value);
  }

  private report(value: unknown): void {
    this.options.onError?.(value instanceof Error ? value : new Error(String(value)));
  }

  private safeTime(): number {
    const value = this.options.nowSeconds?.() ?? Date.now() / 1_000;
    return Number.isFinite(value) && value >= 0 ? value : 0;
  }
}

// PCM remains local and prefix-free here; protocol integration adds 0x01 later.
