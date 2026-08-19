export const MIC_SAMPLE_RATE = 16_000;
export const MIC_CONSTRAINTS = Object.freeze({
  video: false,
  audio: Object.freeze({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
    sampleRate: Object.freeze({ ideal: MIC_SAMPLE_RATE }),
  }),
}) satisfies MediaStreamConstraints;

export interface VadAudioSink {
  push16k(samples: Float32Array): void;
  reset(): void;
  setTtsPlaying(active: boolean): void;
}

export interface MicrophoneSession {
  readonly sampleRate: number;
  close(): Promise<void>;
}

export interface MicrophoneOptions {
  readonly sink: VadAudioSink;
  readonly openAudio?: (
    onSamples: (samples: Float32Array) => void,
    onEnded: (error: Error) => void,
  ) => Promise<MicrophoneSession>;
  readonly onError?: (error: Error) => void;
}

/** Stateful linear resampling; fractional phase and its anchor survive chunks. */
export class StreamingResampler {
  private samples = new Float32Array(0);
  private bufferStart = 0;
  private received = 0;
  private nextPosition = 0;

  constructor(
    readonly sourceRate: number,
    readonly targetRate = MIC_SAMPLE_RATE,
  ) {
    if (!(sourceRate > 0) || !(targetRate > 0)) {
      throw new RangeError('Sample rates must be positive');
    }
  }

  push(input: Float32Array): Float32Array {
    if (input.length === 0) return new Float32Array(0);
    const joined = new Float32Array(this.samples.length + input.length);
    joined.set(this.samples);
    joined.set(input, this.samples.length);
    this.samples = joined;
    this.received += input.length;
    const output: number[] = [];
    const step = this.sourceRate / this.targetRate;
    const last = this.received - 1;
    while (this.nextPosition <= last) {
      const left = Math.floor(this.nextPosition);
      const fraction = this.nextPosition - left;
      const right = fraction === 0 ? left : left + 1;
      if (right > last) break;
      const a = this.samples[left - this.bufferStart] ?? 0;
      const b = this.samples[right - this.bufferStart] ?? a;
      output.push(a + (b - a) * fraction);
      this.nextPosition += step;
    }
    const keepFrom = Math.floor(this.nextPosition);
    const discard = Math.min(this.samples.length, Math.max(0, keepFrom - this.bufferStart));
    this.samples = this.samples.slice(discard);
    this.bufferStart += discard;
    return Float32Array.from(output);
  }

  reset(): void {
    this.samples = new Float32Array(0);
    this.bufferStart = 0;
    this.received = 0;
    this.nextPosition = 0;
  }
}

export function encodePcm16Le(samples: Float32Array): Uint8Array {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index] ?? 0));
    const value = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    view.setInt16(index * 2, value, true);
  }
  return bytes;
}

const WORKLET_SOURCE = `
class LuxoMicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) { const copy = channel.slice(); this.port.postMessage(copy, [copy.buffer]); }
    return true;
  }
}
registerProcessor('luxo-mic', LuxoMicProcessor);`;

export interface BrowserAudioDependencies {
  readonly getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  readonly createContext?: () => AudioContext;
  readonly createWorkletNode?: (context: AudioContext, name: string) => AudioWorkletNode;
}

/** Browser-owned capture. ScriptProcessor is an isolated compatibility fallback. */
export async function openBrowserMicrophone(
  onSamples: (samples: Float32Array) => void,
  onEnded: (error: Error) => void,
  dependencies: BrowserAudioDependencies = {},
): Promise<MicrophoneSession> {
  const getUserMedia = dependencies.getUserMedia ?? ((constraints) => {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone is unavailable');
    return navigator.mediaDevices.getUserMedia(constraints);
  });
  const stream = await getUserMedia(MIC_CONSTRAINTS);
  const track = stream.getAudioTracks()[0];
  if (!track) {
    stream.getTracks().forEach((item) => item.stop());
    throw new Error('Microphone stream contains no audio track');
  }
  const createContext = dependencies.createContext ?? (() => new AudioContext());
  let context: AudioContext;
  let source: MediaStreamAudioSourceNode;
  try {
    context = createContext();
    source = context.createMediaStreamSource(stream);
  } catch (error) {
    stream.getTracks().forEach((item) => item.stop());
    throw error;
  }
  let processor: AudioWorkletNode | ScriptProcessorNode;
  let silentSink: GainNode | null = null;
  try {
    if (context.audioWorklet) {
      const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
      try {
        await context.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      processor = dependencies.createWorkletNode?.(context, 'luxo-mic') ??
        new AudioWorkletNode(context, 'luxo-mic', { numberOfInputs: 1, numberOfOutputs: 0 });
      processor.port.onmessage = (event: MessageEvent<unknown>) => {
        if (event.data instanceof Float32Array) onSamples(event.data);
      };
    } else {
      const fallback = context.createScriptProcessor(4_096, 1, 1);
      fallback.onaudioprocess = (event) => onSamples(event.inputBuffer.getChannelData(0).slice());
      silentSink = context.createGain();
      silentSink.gain.value = 0;
      fallback.connect(silentSink);
      silentSink.connect(context.destination);
      processor = fallback;
    }
    source.connect(processor);
    await context.resume();
  } catch (error) {
    source.disconnect();
    stream.getTracks().forEach((item) => item.stop());
    await context.close().catch(() => undefined);
    throw error;
  }
  let closed = false;
  const handleEnded = () => onEnded(new Error('Microphone device ended'));
  track.addEventListener('ended', handleEnded, { once: true });
  return {
    sampleRate: context.sampleRate,
    async close() {
      if (closed) return;
      closed = true;
      track.removeEventListener('ended', handleEnded);
      source.disconnect();
      processor.disconnect();
      if (processor instanceof AudioWorkletNode) processor.port.close();
      else processor.onaudioprocess = null;
      silentSink?.disconnect();
      stream.getTracks().forEach((item) => item.stop());
      await context.close().catch(() => undefined);
    },
  };
}

export class MicrophoneCapture {
  private readonly openAudio: NonNullable<MicrophoneOptions['openAudio']>;
  private session: MicrophoneSession | null = null;
  private resampler: StreamingResampler | null = null;
  private startPromise: Promise<void> | null = null;
  private generation = 0;
  private disposed = false;

  constructor(private readonly options: MicrophoneOptions) {
    this.openAudio = options.openAudio ?? ((samples, ended) => openBrowserMicrophone(samples, ended));
  }

  get live(): boolean { return this.session !== null; }

  start(): Promise<void> {
    if (this.disposed) return Promise.reject(new Error('Microphone is disposed'));
    if (this.session) return Promise.resolve();
    if (this.startPromise) return this.startPromise;
    const generation = ++this.generation;
    const pending = this.openAudio(
      (samples) => {
        const converted = this.resampler?.push(samples);
        if (converted?.length) this.options.sink.push16k(converted);
      },
      (error) => { this.options.onError?.(error); void this.stop(); },
    ).then(async (session) => {
      if (generation !== this.generation || this.disposed) {
        await session.close();
        throw new Error('Microphone start was cancelled');
      }
      this.session = session;
      this.resampler = new StreamingResampler(session.sampleRate);
    }).catch((value: unknown) => {
      const error = value instanceof Error ? value : new Error(String(value));
      if (!/cancelled/.test(error.message)) this.options.onError?.(error);
      throw error;
    });
    const tracked = pending.finally(() => {
      if (this.startPromise === tracked) this.startPromise = null;
    });
    this.startPromise = tracked;
    return tracked;
  }

  async stop(): Promise<void> {
    this.generation += 1;
    this.startPromise = null;
    const session = this.session;
    this.session = null;
    this.resampler?.reset();
    this.resampler = null;
    this.options.sink.reset();
    if (session) await session.close();
  }

  setTtsPlaying(active: boolean): void {
    if (active) this.resampler?.reset();
    this.options.sink.setTtsPlaying(active);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.stop();
  }
}

// Manual integration: on localhost grant/deny permission, unplug the device,
// verify the worklet path, then confirm the browser recording indicator clears.
