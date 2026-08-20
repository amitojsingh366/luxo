import {
  BinaryPrefix,
  type BodyStateMessage,
  type BrowserToCoreMessage,
  type CaptureFrameMessage,
  type ClearMemoryMessage,
  type CoreToBrowserMessage,
  type CueMessage,
  type ErrorMessage,
  type GazeMessage,
  type HelloMessage,
  type SpeakBeginMessage,
  type TtsDoneMessage,
  type VadMessage,
} from './types';

// The core is not trusted. Every text frame is checked against the frozen
// schema in schema/messages.schema.json before any behaviour callback runs,
// so a malformed capture_frame can never reach the camera (PRD 5.3).

type Validator = (value: unknown, path: string) => string | null;

interface NumberRules {
  readonly integer?: boolean;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly exclusiveMinimum?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasOwn(target: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(target, key);
}

function numberField(rules: NumberRules = {}): Validator {
  return (value, path) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return `${path} must be a finite number`;
    }
    if (rules.integer === true && !Number.isInteger(value)) return `${path} must be an integer`;
    if (rules.minimum !== undefined && value < rules.minimum) {
      return `${path} must be at least ${rules.minimum}`;
    }
    if (rules.maximum !== undefined && value > rules.maximum) {
      return `${path} must be at most ${rules.maximum}`;
    }
    if (rules.exclusiveMinimum !== undefined && value <= rules.exclusiveMinimum) {
      return `${path} must be greater than ${rules.exclusiveMinimum}`;
    }
    return null;
  };
}

function booleanField(): Validator {
  return (value, path) => (typeof value === 'boolean' ? null : `${path} must be a boolean`);
}

function enumField(allowed: readonly string[]): Validator {
  return (value, path) =>
    typeof value === 'string' && allowed.includes(value)
      ? null
      : `${path} must be one of ${allowed.join(', ')}`;
}

function nonEmptyStringField(): Validator {
  return (value, path) =>
    typeof value === 'string' && value.length > 0 ? null : `${path} must be a non-empty string`;
}

// Every core-to-browser object in the schema lists all of its properties as
// required and sets additionalProperties: false, so an exact key match is the
// faithful reading of the schema.
function objectField(fields: Readonly<Record<string, Validator>>): Validator {
  return (value, path) => {
    if (!isRecord(value)) return `${path} must be an object`;
    for (const key of Object.keys(value)) {
      if (!hasOwn(fields, key)) return `${path}.${key} is not allowed by the schema`;
    }
    for (const key of Object.keys(fields)) {
      if (!hasOwn(value, key)) return `${path}.${key} is required`;
      const failure = fields[key]!(value[key], `${path}.${key}`);
      if (failure !== null) return failure;
    }
    return null;
  };
}

const jointsField = objectField({
  base_yaw: numberField(),
  shoulder_pitch: numberField(),
  elbow_pitch: numberField(),
  neck_yaw: numberField(),
  head_pitch: numberField(),
});

const lightField = objectField({
  intensity: numberField({ minimum: 0 }),
  color_k: numberField({ integer: true, minimum: 1 }),
  pattern: enumField(['steady', 'pulse', 'flicker', 'blink']),
  bloom: numberField({ minimum: 0 }),
});

const audioField = objectField({
  speaking: booleanField(),
  arousal: numberField({ minimum: 0, maximum: 1 }),
});

const telemetryField = objectField({
  state: enumField([
    'BOOT',
    'DORMANT',
    'NOTICING',
    'ENGAGED',
    'LISTENING',
    'THINKING',
    'SPEAKING',
    'INSPECTING',
    'ACTING',
    'DISENGAGING',
  ]),
  plan_depth: numberField({ integer: true, minimum: 0 }),
  memory_count: numberField({ integer: true, minimum: 0 }),
  last_latency_ms: numberField({ minimum: 0 }),
  clamps: objectField({
    vel: numberField({ integer: true, minimum: 0 }),
    limit: numberField({ integer: true, minimum: 0 }),
  }),
  gaze: objectField({
    present: booleanField(),
    yaw_deg: numberField(),
    pitch_deg: numberField(),
  }),
});

const CORE_TO_BROWSER_FIELDS: Readonly<Record<CoreToBrowserMessage['type'], Validator>> = {
  body_state: objectField({
    type: enumField(['body_state']),
    t: numberField(),
    seq: numberField({ integer: true, minimum: 0 }),
    joints: jointsField,
    light: lightField,
    audio: audioField,
    telemetry: telemetryField,
  }),
  cue: objectField({
    type: enumField(['cue']),
    sfx: enumField([
      'chirp_up',
      'chirp_found',
      'boing',
      'whirr_short',
      'hmm',
      'blip_sad',
      'fanfare_small',
      'click',
    ]),
  }),
  capture_frame: objectField({
    type: enumField(['capture_frame']),
    req_id: nonEmptyStringField(),
  }),
  speak_begin: objectField({
    type: enumField(['speak_begin']),
    envelope_hz: numberField({ exclusiveMinimum: 0 }),
  }),
  speak_end: objectField({ type: enumField(['speak_end']) }),
};

export type CoreMessageCheck =
  | { readonly ok: true; readonly message: CoreToBrowserMessage }
  | { readonly ok: false; readonly reason: string };

export function validateCoreToBrowserMessage(value: unknown): CoreMessageCheck {
  if (!isRecord(value)) return { ok: false, reason: 'Core sent a message that is not an object' };
  const type = value['type'];
  if (typeof type !== 'string') {
    return { ok: false, reason: 'Core sent a message without a string type' };
  }
  if (!hasOwn(CORE_TO_BROWSER_FIELDS, type)) {
    return { ok: false, reason: `Core sent unknown message type: ${type}` };
  }
  const validate = CORE_TO_BROWSER_FIELDS[type as CoreToBrowserMessage['type']];
  const failure = validate(value, type);
  if (failure !== null) {
    return { ok: false, reason: `Core sent an invalid ${type} message: ${failure}` };
  }
  return { ok: true, message: value as unknown as CoreToBrowserMessage };
}

// PRD 8.4: TTS is 22.05 kHz mono int16 streamed in chunks of at most 8 KiB.
export const TTS_PCM_MAX_PAYLOAD_BYTES = 8_192;

export function validateTtsPcmFrame(frame: Uint8Array): string | null {
  if (frame.length === 0) return 'Core sent an empty binary frame';
  // Exactly one prefix byte, then payload: byte 0 must be the TTS prefix and
  // every remaining byte belongs to the PCM chunk.
  if (frame[0] !== BinaryPrefix.tts_pcm) {
    return `Core sent unknown binary prefix: 0x${frame[0]!.toString(16)}`;
  }
  const payloadBytes = frame.length - 1;
  if (payloadBytes === 0) return 'Core sent a TTS frame with no PCM payload';
  if (payloadBytes % 2 !== 0) {
    return `Core sent a TTS payload of ${payloadBytes} bytes, not whole 16-bit samples`;
  }
  if (payloadBytes > TTS_PCM_MAX_PAYLOAD_BYTES) {
    return `Core sent a TTS payload of ${payloadBytes} bytes, over the ${TTS_PCM_MAX_PAYLOAD_BYTES}-byte limit`;
  }
  return null;
}

export interface ProtocolClientCallbacks {
  onBodyState?: (state: BodyStateMessage) => void;
  onCue?: (event: CueMessage) => void;
  onCaptureFrame?: (event: CaptureFrameMessage) => void;
  onSpeakBegin?: (event: SpeakBeginMessage) => void;
  onSpeakEnd?: () => void;
  onTtsPcm?: (pcm: Uint8Array) => void;
  onConnected?: () => void;
  onDisconnected?: (event: CloseEvent) => void;
  onError?: (error: Error) => void;
}

export interface ProtocolClientOptions extends ProtocolClientCallbacks {
  url?: string;
  hello?: HelloMessage;
  reconnectInitialMs?: number;
  reconnectMaxMs?: number;
  reconnectFactor?: number;
  reconnectJitter?: number;
  webSocketFactory?: (url: string) => WebSocket;
}

const DEFAULT_HELLO: HelloMessage = {
  type: 'hello',
  fps: 60,
  camera: { w: 640, h: 480, hfov_deg: 60 },
};

export class ProtocolClient {
  private readonly url: string;
  private readonly hello: HelloMessage;
  private readonly callbacks: ProtocolClientCallbacks;
  private readonly reconnectInitialMs: number;
  private readonly reconnectMaxMs: number;
  private readonly reconnectFactor: number;
  private readonly reconnectJitter: number;
  private readonly webSocketFactory: (url: string) => WebSocket;
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelayMs: number;
  private stopped = true;

  constructor(options: ProtocolClientOptions = {}) {
    this.url = options.url ?? 'ws://127.0.0.1:8765';
    this.hello = options.hello ?? DEFAULT_HELLO;
    this.reconnectInitialMs = Math.max(50, options.reconnectInitialMs ?? 250);
    this.reconnectMaxMs = Math.max(
      this.reconnectInitialMs,
      options.reconnectMaxMs ?? 5_000,
    );
    this.reconnectFactor = Math.max(1, options.reconnectFactor ?? 2);
    this.reconnectJitter = Math.min(1, Math.max(0, options.reconnectJitter ?? 0.2));
    this.reconnectDelayMs = this.reconnectInitialMs;
    this.webSocketFactory = options.webSocketFactory ?? ((url) => new WebSocket(url));
    this.callbacks = options;
  }

  connect(): void {
    this.stopped = false;
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }
    this.clearReconnectTimer();
    const socket = this.webSocketFactory(this.url);
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.addEventListener('open', () => {
      if (this.socket !== socket) return;
      this.reconnectDelayMs = this.reconnectInitialMs;
      this.sendJson(this.hello);
      this.callbacks.onConnected?.();
    });
    socket.addEventListener('message', (event) => {
      if (this.socket !== socket) return;
      void this.handleIncoming(event.data);
    });
    socket.addEventListener('error', () => {
      if (this.socket === socket) {
        this.callbacks.onError?.(new Error('WebSocket transport error'));
      }
    });
    socket.addEventListener('close', (event) => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.callbacks.onDisconnected?.(event);
      if (!this.stopped) this.scheduleReconnect();
    });
  }

  disconnect(): void {
    this.stopped = true;
    this.clearReconnectTimer();
    const socket = this.socket;
    this.socket = null;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close(1000, 'client shutdown');
    }
  }

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  sendGaze(fact: Omit<GazeMessage, 'type'>): boolean {
    return this.sendJson({ type: 'gaze', ...fact });
  }

  sendVad(event: VadMessage['event'], t = Date.now() / 1_000): boolean {
    return this.sendJson({ type: 'vad', t, event });
  }

  sendTtsDone(t = Date.now() / 1_000): boolean {
    const message: TtsDoneMessage = { type: 'tts_done', t };
    return this.sendJson(message);
  }

  sendError(where: string, detail: string): boolean {
    const message: ErrorMessage = { type: 'error', where, detail };
    return this.sendJson(message);
  }

  sendClearMemory(): boolean {
    const message: ClearMemoryMessage = { type: 'clear_memory' };
    return this.sendJson(message);
  }

  sendUtterancePcm(payload: ArrayBuffer | Uint8Array): boolean {
    return this.sendBinary(BinaryPrefix.utterance_pcm, payload);
  }

  sendCapturedJpeg(payload: ArrayBuffer | Uint8Array): boolean {
    return this.sendBinary(BinaryPrefix.capture_jpeg, payload);
  }

  private sendJson(message: BrowserToCoreMessage): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  private sendBinary(prefix: number, payload: ArrayBuffer | Uint8Array): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    const source = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
    const frame = new Uint8Array(source.byteLength + 1);
    frame[0] = prefix;
    frame.set(source, 1);
    this.socket.send(frame);
    return true;
  }

  private async handleIncoming(data: unknown): Promise<void> {
    if (typeof data === 'string') {
      this.handleText(data);
      return;
    }
    if (data instanceof ArrayBuffer) {
      this.handleBinary(data);
      return;
    }
    if (data instanceof Blob) {
      this.handleBinary(await data.arrayBuffer());
      return;
    }
    this.callbacks.onError?.(new Error('Unsupported WebSocket frame type'));
  }

  private handleText(text: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      this.callbacks.onError?.(new Error('Core sent invalid JSON'));
      return;
    }
    const checked = validateCoreToBrowserMessage(parsed);
    if (!checked.ok) {
      this.callbacks.onError?.(new Error(checked.reason));
      return;
    }

    const message = checked.message;
    switch (message.type) {
      case 'body_state':
        this.callbacks.onBodyState?.(message);
        break;
      case 'cue':
        this.callbacks.onCue?.(message);
        break;
      case 'capture_frame':
        this.callbacks.onCaptureFrame?.(message);
        break;
      case 'speak_begin':
        this.callbacks.onSpeakBegin?.(message);
        break;
      case 'speak_end':
        this.callbacks.onSpeakEnd?.();
        break;
    }
  }

  private handleBinary(buffer: ArrayBuffer): void {
    const frame = new Uint8Array(buffer);
    const failure = validateTtsPcmFrame(frame);
    if (failure !== null) {
      this.callbacks.onError?.(new Error(failure));
      return;
    }
    this.callbacks.onTtsPcm?.(frame.subarray(1));
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer();
    const jitter = 1 + (Math.random() * 2 - 1) * this.reconnectJitter;
    const delay = Math.round(this.reconnectDelayMs * jitter);
    this.reconnectDelayMs = Math.min(
      this.reconnectMaxMs,
      this.reconnectDelayMs * this.reconnectFactor,
    );
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.stopped) this.connect();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
