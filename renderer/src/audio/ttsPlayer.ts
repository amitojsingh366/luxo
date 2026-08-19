export const TTS_SAMPLE_RATE = 22_050;
export const TTS_CHANNELS = 1;
export const TTS_MAX_CHUNK_BYTES = 8_192;

export type TtsPlayerErrorCode =
  | "audio-context"
  | "disposed"
  | "invalid-chunk"
  | "invalid-order"
  | "underrun";

export class TtsPlayerError extends Error {
  constructor(
    readonly code: TtsPlayerErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "TtsPlayerError";
  }
}

export type TtsAudioContext = Pick<
  AudioContext,
  | "close"
  | "createBuffer"
  | "createBufferSource"
  | "currentTime"
  | "destination"
  | "resume"
  | "state"
>;

export interface TtsPlayerOptions {
  context?: TtsAudioContext;
  destination?: AudioNode;
  contextFactory?: () => TtsAudioContext;
  onTtsDone: () => void;
  setVadSuppressed: (suppressed: boolean) => void;
  schedulingLeadSeconds?: number;
}

export interface TtsPlaybackFacts {
  speaking: boolean;
  inputOpen: boolean;
  chunkCount: number;
  utteranceDurationSeconds: number;
  queuedDurationSeconds: number;
  scheduledUntil: number | null;
}

interface Utterance {
  generation: number;
  inputOpen: boolean;
  chunkCount: number;
  sampleCount: number;
  scheduledUntil: number | null;
  sources: Set<AudioBufferSourceNode>;
}

const DEFAULT_LEAD_SECONDS = 0.05;
const TIME_EPSILON_SECONDS = 1e-6;

function asBytes(chunk: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (chunk instanceof ArrayBuffer) return new Uint8Array(chunk);
  return new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
}

export function validateTtsPcmChunk(
  chunk: ArrayBuffer | ArrayBufferView,
): Uint8Array {
  const bytes = asBytes(chunk);
  if (bytes.byteLength === 0) {
    throw new TtsPlayerError("invalid-chunk", "TTS PCM chunk must not be empty");
  }
  if (bytes.byteLength > TTS_MAX_CHUNK_BYTES) {
    throw new TtsPlayerError(
      "invalid-chunk",
      `TTS PCM chunk exceeds ${TTS_MAX_CHUNK_BYTES} bytes`,
    );
  }
  if (bytes.byteLength % 2 !== 0) {
    throw new TtsPlayerError(
      "invalid-chunk",
      "TTS PCM chunk must contain aligned int16 samples",
    );
  }
  return bytes;
}

export function pcm16LeToFloat32(
  chunk: ArrayBuffer | ArrayBufferView,
): Float32Array {
  const bytes = validateTtsPcmChunk(chunk);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = view.getInt16(index * 2, true);
    samples[index] = sample < 0 ? sample / 32_768 : sample / 32_767;
  }
  return samples;
}

function defaultContextFactory(): TtsAudioContext {
  const constructors = globalThis as typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };
  const Context = constructors.AudioContext ?? constructors.webkitAudioContext;
  if (Context === undefined) {
    throw new TtsPlayerError(
      "audio-context",
      "Web Audio is unavailable in this browser",
    );
  }
  return new Context();
}

export class TtsPlayer {
  private context: TtsAudioContext | null;
  private destination: AudioNode | null;
  private readonly contextFactory: () => TtsAudioContext;
  private readonly onTtsDone: () => void;
  private readonly setVadSuppressed: (suppressed: boolean) => void;
  private readonly schedulingLeadSeconds: number;
  private utterance: Utterance | null = null;
  private generation = 0;
  private disposed = false;
  private ownsContext = false;

  constructor(options: TtsPlayerOptions) {
    const lead = options.schedulingLeadSeconds ?? DEFAULT_LEAD_SECONDS;
    if (!Number.isFinite(lead) || lead < 0) {
      throw new RangeError("schedulingLeadSeconds must be finite and non-negative");
    }
    this.context = options.context ?? null;
    this.destination = options.destination ?? options.context?.destination ?? null;
    this.contextFactory = options.contextFactory ?? defaultContextFactory;
    this.onTtsDone = options.onTtsDone;
    this.setVadSuppressed = options.setVadSuppressed;
    this.schedulingLeadSeconds = lead;
  }

  get facts(): TtsPlaybackFacts {
    const utterance = this.utterance;
    const now = this.context?.currentTime ?? 0;
    const duration = (utterance?.sampleCount ?? 0) / TTS_SAMPLE_RATE;
    const scheduledUntil = utterance?.scheduledUntil ?? null;
    return {
      speaking: utterance !== null,
      inputOpen: utterance?.inputOpen ?? false,
      chunkCount: utterance?.chunkCount ?? 0,
      utteranceDurationSeconds: duration,
      queuedDurationSeconds:
        scheduledUntil === null ? 0 : Math.max(0, scheduledUntil - now),
      scheduledUntil,
    };
  }

  async unlock(): Promise<void> {
    this.assertUsable();
    const context = this.ensureContext();
    try {
      if (context.state === "closed") {
        throw new TtsPlayerError("audio-context", "TTS audio context is closed");
      }
      if (context.state !== "running") await context.resume();
    } catch (error) {
      this.cancelUtterance();
      throw error;
    }
  }

  begin(): void {
    this.assertUsable();
    if (this.utterance?.inputOpen === true) return;
    if (this.utterance !== null) this.cancelUtterance();
    this.ensureContext();
    const utterance: Utterance = {
      generation: ++this.generation,
      inputOpen: true,
      chunkCount: 0,
      sampleCount: 0,
      scheduledUntil: null,
      sources: new Set(),
    };
    this.utterance = utterance;
    try {
      this.setVadSuppressed(true);
    } catch (error) {
      this.utterance = null;
      throw error;
    }
  }

  enqueue(chunk: ArrayBuffer | ArrayBufferView): void {
    this.assertUsable();
    const utterance = this.utterance;
    if (utterance === null || !utterance.inputOpen) {
      throw new TtsPlayerError(
        "invalid-order",
        "TTS PCM chunk received outside an open utterance",
      );
    }

    let samples: Float32Array;
    try {
      samples = pcm16LeToFloat32(chunk);
    } catch (error) {
      this.cancelUtterance();
      throw error;
    }

    const context = this.ensureContext();
    const nextStart = utterance.scheduledUntil;
    if (
      nextStart !== null &&
      context.currentTime > nextStart + TIME_EPSILON_SECONDS
    ) {
      this.cancelUtterance();
      throw new TtsPlayerError(
        "underrun",
        "TTS PCM arrived after the scheduled audio queue ran dry",
      );
    }
    const startAt =
      nextStart ?? context.currentTime + this.schedulingLeadSeconds;

    let source: AudioBufferSourceNode | null = null;
    try {
      const buffer = context.createBuffer(
        TTS_CHANNELS,
        samples.length,
        TTS_SAMPLE_RATE,
      );
      buffer.getChannelData(0).set(samples);
      source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.destination ?? context.destination);
      const generation = utterance.generation;
      source.onended = () => this.handleEnded(source as AudioBufferSourceNode, generation);
      source.start(startAt);
    } catch (error) {
      if (source !== null) this.cleanupSource(source, true);
      this.cancelUtterance();
      throw error;
    }

    utterance.sources.add(source);
    utterance.chunkCount += 1;
    utterance.sampleCount += samples.length;
    utterance.scheduledUntil = startAt + samples.length / TTS_SAMPLE_RATE;
  }

  end(): void {
    this.assertUsable();
    const utterance = this.utterance;
    if (utterance === null) {
      throw new TtsPlayerError(
        "invalid-order",
        "speak_end received without speak_begin",
      );
    }
    if (!utterance.inputOpen) return;
    utterance.inputOpen = false;
    this.completeIfFinished(utterance);
  }

  stop(): void {
    if (this.disposed) return;
    this.cancelUtterance();
  }

  dispose(): void {
    if (this.disposed) return;
    this.cancelUtterance();
    this.disposed = true;
    if (this.ownsContext && this.context?.state !== "closed") {
      void this.context?.close();
    }
    this.context = null;
    this.destination = null;
  }

  private ensureContext(): TtsAudioContext {
    if (this.context === null) {
      this.context = this.contextFactory();
      this.ownsContext = true;
    }
    if (this.destination === null) this.destination = this.context.destination;
    return this.context;
  }

  private handleEnded(source: AudioBufferSourceNode, generation: number): void {
    const utterance = this.utterance;
    if (utterance === null || utterance.generation !== generation) return;
    this.cleanupSource(source, false);
    utterance.sources.delete(source);
    this.completeIfFinished(utterance);
  }

  private completeIfFinished(utterance: Utterance): void {
    if (utterance.inputOpen || utterance.sources.size > 0) return;
    if (this.utterance?.generation !== utterance.generation) return;
    this.utterance = null;
    this.setVadSuppressed(false);
    this.onTtsDone();
  }

  private cancelUtterance(): void {
    const utterance = this.utterance;
    if (utterance === null) return;
    this.utterance = null;
    this.generation += 1;
    for (const source of utterance.sources) this.cleanupSource(source, true);
    utterance.sources.clear();
    this.setVadSuppressed(false);
  }

  private cleanupSource(source: AudioBufferSourceNode, stop: boolean): void {
    source.onended = null;
    if (stop) {
      try {
        source.stop();
      } catch {
        // A source which has already ended needs only disconnection.
      }
    }
    try {
      source.disconnect();
    } catch {
      // Disconnect is idempotent across browser implementations.
    }
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new TtsPlayerError("disposed", "TTS player has been disposed");
    }
  }
}
