import {
  BinaryPrefix,
  type BodyStateMessage,
  type BrowserToCoreMessage,
  type CaptureFrameMessage,
  type CueMessage,
  type ErrorMessage,
  type GazeMessage,
  type HelloMessage,
  type SpeakBeginMessage,
  type TtsDoneMessage,
  type VadMessage,
} from './types';

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
    let message: unknown;
    try {
      message = JSON.parse(text);
    } catch {
      this.callbacks.onError?.(new Error('Core sent invalid JSON'));
      return;
    }
    if (!message || typeof message !== 'object' || !('type' in message)) {
      this.callbacks.onError?.(new Error('Core sent an invalid message shape'));
      return;
    }

    switch ((message as { type: string }).type) {
      case 'body_state':
        this.callbacks.onBodyState?.(message as BodyStateMessage);
        break;
      case 'cue':
        this.callbacks.onCue?.(message as CueMessage);
        break;
      case 'capture_frame':
        this.callbacks.onCaptureFrame?.(message as CaptureFrameMessage);
        break;
      case 'speak_begin':
        this.callbacks.onSpeakBegin?.(message as SpeakBeginMessage);
        break;
      case 'speak_end':
        this.callbacks.onSpeakEnd?.();
        break;
      default:
        this.callbacks.onError?.(
          new Error(`Core sent unknown message type: ${(message as { type: string }).type}`),
        );
    }
  }

  private handleBinary(buffer: ArrayBuffer): void {
    const frame = new Uint8Array(buffer);
    if (frame.length === 0) {
      this.callbacks.onError?.(new Error('Core sent an empty binary frame'));
      return;
    }
    if (frame[0] !== BinaryPrefix.tts_pcm) {
      this.callbacks.onError?.(
        new Error(`Core sent unknown binary prefix: 0x${frame[0]!.toString(16)}`),
      );
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
