import * as Tone from "tone";

import type {
  BodyStateMessage,
  CueMessage,
  SpeakBeginMessage,
  SpeakEndMessage,
} from "../protocol/types";
import {
  createMusicGraph,
  normalizeBodyAudio,
  type BodyAudioState,
  type RoutedMusicGraph,
} from "./music";
import { createSfxGraph, type RoutedSfxGraph } from "./sfx";
import {
  TtsPlayer,
  type TtsAudioContext,
  type TtsPlaybackFacts,
  type TtsPlayerOptions,
} from "./ttsPlayer";

export interface BrowserAudioGraph {
  readonly context: TtsAudioContext;
  readonly destination: AudioNode;
  readonly music: RoutedMusicGraph;
  readonly sfx: RoutedSfxGraph;
  unlock(): Promise<void>;
  dispose(): void;
}

export interface MixerTtsPlayer {
  readonly facts: TtsPlaybackFacts;
  unlock(): Promise<void>;
  begin(): void;
  enqueue(chunk: ArrayBuffer | ArrayBufferView): void;
  end(): void;
  stop(): void;
  dispose(): void;
}

export interface BrowserAudioMixerOptions {
  readonly setVadSuppressed: (suppressed: boolean) => void;
  readonly onTtsDone: () => void;
  readonly graphFactory?: (initial: BodyAudioState) => BrowserAudioGraph;
  readonly ttsFactory?: (options: TtsPlayerOptions) => MixerTtsPlayer;
}

const SILENT_AUDIO = Object.freeze({ arousal: 0, speaking: false });
const SILENT_TTS_FACTS: TtsPlaybackFacts = Object.freeze({
  speaking: false,
  inputOpen: false,
  chunkCount: 0,
  utteranceDurationSeconds: 0,
  queuedDurationSeconds: 0,
  scheduledUntil: null,
});

class ToneBrowserAudioGraph implements BrowserAudioGraph {
  readonly context: TtsAudioContext;
  readonly destination: AudioNode;
  readonly music: RoutedMusicGraph;
  readonly sfx: RoutedSfxGraph;
  private readonly master: Tone.Gain;
  private disposed = false;

  constructor(initial: BodyAudioState) {
    const rawContext = Tone.getContext().rawContext;
    if (!("resume" in rawContext) || !("close" in rawContext)) {
      throw new Error("Luxo audio requires a realtime browser AudioContext");
    }
    this.context = rawContext as TtsAudioContext;
    const master = new Tone.Gain(0.9).toDestination();
    let sfx: RoutedSfxGraph | null = null;
    try {
      sfx = createSfxGraph(master);
      this.music = createMusicGraph(master, initial);
    } catch (error) {
      try {
        sfx?.dispose();
      } finally {
        master.dispose();
      }
      throw error;
    }
    this.master = master;
    this.destination = master.input;
    this.sfx = sfx;
  }

  async unlock(): Promise<void> {
    if (this.disposed) throw new Error("Browser audio graph has been disposed");
    await Tone.start();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    let failure: unknown;
    for (const release of [
      () => this.music.dispose(),
      () => this.sfx.dispose(),
      () => this.master.dispose(),
    ]) {
      try {
        release();
      } catch (error) {
        failure ??= error;
      }
    }
    if (failure !== undefined) throw failure;
  }
}

export function createBrowserAudioGraph(
  initial: BodyAudioState = SILENT_AUDIO,
): BrowserAudioGraph {
  return new ToneBrowserAudioGraph(normalizeBodyAudio(initial));
}

export class BrowserAudioMixer {
  private readonly graphFactory: NonNullable<
    BrowserAudioMixerOptions["graphFactory"]
  >;
  private readonly ttsFactory: NonNullable<
    BrowserAudioMixerOptions["ttsFactory"]
  >;
  private requestedAudio: BodyAudioState = SILENT_AUDIO;
  private graph: BrowserAudioGraph | null = null;
  private tts: MixerTtsPlayer | null = null;
  private unlockPromise: Promise<void> | null = null;
  private vadSuppressed = false;
  private generation = 0;
  private disposed = false;

  constructor(private readonly options: BrowserAudioMixerOptions) {
    this.graphFactory = options.graphFactory ?? createBrowserAudioGraph;
    this.ttsFactory = options.ttsFactory ?? ((ttsOptions) => new TtsPlayer(ttsOptions));
  }

  get initialized(): boolean {
    return this.graph !== null;
  }

  get ttsFacts(): TtsPlaybackFacts {
    return this.tts?.facts ?? SILENT_TTS_FACTS;
  }

  applyBodyState(state: Pick<BodyStateMessage, "audio">): void {
    this.setBodyAudio(state.audio);
  }

  setBodyAudio(audio: BodyAudioState): void {
    this.assertUsable();
    this.requestedAudio = normalizeBodyAudio(audio);
    this.graph?.music.setBodyAudio(this.requestedAudio);
  }

  playCue(cue: Pick<CueMessage, "sfx">): void {
    this.assertUsable();
    this.ensureGraph().sfx.play(cue.sfx);
  }

  speakBegin(message: SpeakBeginMessage): void {
    this.assertUsable();
    if (message.type !== "speak_begin") {
      throw new RangeError("Expected a speak_begin message");
    }
    try {
      this.ensureGraph();
      this.tts?.begin();
    } catch (error) {
      this.abortSpeech();
      throw error;
    }
  }

  enqueueTtsPcm(chunk: ArrayBuffer | ArrayBufferView): void {
    this.assertUsable();
    try {
      if (this.tts === null) throw new Error("TTS PCM received before speak_begin");
      this.tts.enqueue(chunk);
    } catch (error) {
      this.abortSpeech();
      throw error;
    }
  }

  speakEnd(message: SpeakEndMessage): void {
    this.assertUsable();
    if (message.type !== "speak_end") {
      throw new RangeError("Expected a speak_end message");
    }
    try {
      if (this.tts === null) throw new Error("speak_end received before speak_begin");
      this.tts.end();
    } catch (error) {
      this.abortSpeech();
      throw error;
    }
  }

  async unlock(): Promise<void> {
    this.assertUsable();
    if (this.unlockPromise !== null) return this.unlockPromise;
    const graph = this.ensureGraph();
    const tts = this.tts;
    const generation = this.generation;
    const pending = graph
      .unlock()
      .then(async () => {
        if (this.disposed || generation !== this.generation) return;
        await tts?.unlock();
      })
      .catch((error: unknown) => {
        if (!this.disposed && generation === this.generation) {
          this.unlockPromise = null;
          this.abortSpeech();
        }
        throw error;
      });
    this.unlockPromise = pending;
    return pending;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.generation += 1;
    let failure: unknown;
    const release = (operation: () => void) => {
      try {
        operation();
      } catch (error) {
        failure ??= error;
      }
    };
    release(() => this.tts?.dispose());
    if (this.vadSuppressed) release(() => this.releaseVadSuppression());
    release(() => this.graph?.dispose());
    this.tts = null;
    this.graph = null;
    this.unlockPromise = null;
    if (failure !== undefined) throw failure;
  }

  private ensureGraph(): BrowserAudioGraph {
    if (this.graph !== null) return this.graph;
    const graph = this.graphFactory(this.requestedAudio);
    try {
      const tts = this.ttsFactory({
        context: graph.context,
        destination: graph.destination,
        setVadSuppressed: (suppressed) => this.setVadSuppression(suppressed),
        onTtsDone: this.options.onTtsDone,
      });
      this.graph = graph;
      this.tts = tts;
      return graph;
    } catch (error) {
      graph.dispose();
      throw error;
    }
  }

  private setVadSuppression(suppressed: boolean): void {
    this.vadSuppressed = suppressed;
    this.options.setVadSuppressed(suppressed);
  }

  private releaseVadSuppression(): void {
    this.vadSuppressed = false;
    this.options.setVadSuppressed(false);
  }

  private abortSpeech(): void {
    try {
      this.tts?.stop();
    } finally {
      if (this.vadSuppressed) this.releaseVadSuppression();
    }
  }

  private assertUsable(): void {
    if (this.disposed) throw new Error("Browser audio mixer has been disposed");
  }
}
