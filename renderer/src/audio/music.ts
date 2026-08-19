import * as Tone from "tone";

export interface BodyAudioState {
  readonly arousal: number;
  readonly speaking: boolean;
}

export const PENTATONIC_NOTES = Object.freeze([
  "C4",
  "D4",
  "E4",
  "G4",
  "A4",
] as const);

export const PLUCK_PATTERN = Object.freeze([
  "C4",
  "E4",
  "G4",
  "D4",
  "A4",
  "G4",
  "E4",
  "D4",
] as const);

export const PAD_OSCILLATORS = Object.freeze([
  Object.freeze({ note: "C3", type: "sine", detune: -6 }),
  Object.freeze({ note: "G3", type: "triangle", detune: 6 }),
] as const);

export const MUSIC_CONFIG = Object.freeze({
  masterHeadroom: 0.3,
  maxPadGain: 0.34,
  maxPluckGain: 0.24,
  maxPlucksPerSecond: 3.2,
  speechDuckGain: 0.24,
  schedulerTickSeconds: 0.125,
  engageSwellSeconds: 0.9,
  disengageThinSeconds: 1.5,
  duckSeconds: 0.14,
  recoverSeconds: 0.5,
});

export function clampArousal(arousal: number): number {
  if (!Number.isFinite(arousal)) return 0;
  return Math.min(1, Math.max(0, arousal));
}
export function arousalToPluckDensity(arousal: number): number {
  const normalized = clampArousal(arousal);
  return MUSIC_CONFIG.maxPlucksPerSecond * normalized * normalized;
}
export function arousalToPadGain(arousal: number): number {
  return MUSIC_CONFIG.maxPadGain * Math.sqrt(clampArousal(arousal));
}
export function arousalToPluckGain(arousal: number): number {
  return MUSIC_CONFIG.maxPluckGain * clampArousal(arousal);
}
export function speakingToDuckGain(speaking: boolean): number {
  return speaking ? MUSIC_CONFIG.speechDuckGain : 1;
}

export function normalizeBodyAudio(audio: BodyAudioState): BodyAudioState {
  return Object.freeze({
    arousal: clampArousal(audio.arousal),
    speaking: audio.speaking === true,
  });
}

export const browserAudioOutput = Object.freeze({
  destination(): ReturnType<typeof Tone.getDestination> {
    return Tone.getDestination();
  },
});

interface Disposable {
  dispose(): unknown;
}

class LuxoMusicGraph {
  private readonly output = new Tone.Gain(MUSIC_CONFIG.masterHeadroom).connect(
    browserAudioOutput.destination(),
  );
  private readonly musicGain = new Tone.Gain(1).connect(this.output);
  private readonly padGain = new Tone.Gain(0).connect(this.musicGain);
  private readonly padFilter = new Tone.Filter({
    type: "lowpass",
    frequency: 1_150,
    rolloff: -12,
    Q: 0.7,
  }).connect(this.padGain);
  private readonly padOscillators = PAD_OSCILLATORS.map((voice) =>
    new Tone.Oscillator({
      frequency: voice.note,
      type: voice.type,
      detune: voice.detune,
      volume: -9,
    }).connect(this.padFilter),
  );
  private readonly pluckGain = new Tone.Gain(0).connect(this.musicGain);
  private readonly pluck = new Tone.PluckSynth({
    attackNoise: 0.7,
    dampening: 3_200,
    resonance: 0.72,
    release: 0.8,
    volume: -12,
  }).connect(this.pluckGain);
  private readonly scheduler = new Tone.Loop(
    (time) => this.schedulePluck(time),
    MUSIC_CONFIG.schedulerTickSeconds,
  );
  private readonly resources: readonly Disposable[];
  private density = 0;
  private accumulator = 0;
  private patternIndex = 0;
  private arousal = 0;
  private speaking = false;
  private disposed = false;

  constructor(initial: BodyAudioState) {
    this.resources = [
      this.scheduler,
      this.pluck,
      this.pluckGain,
      ...this.padOscillators,
      this.padFilter,
      this.padGain,
      this.musicGain,
      this.output,
    ];
    for (const oscillator of this.padOscillators) oscillator.start();
    const transport = Tone.getTransport();
    this.scheduler.start(transport.seconds);
    if (transport.state !== "started") transport.start();
    this.setBodyAudio(initial);
  }

  setBodyAudio(audio: BodyAudioState): void {
    if (this.disposed) return;
    const rampSeconds =
      audio.arousal >= this.arousal
        ? MUSIC_CONFIG.engageSwellSeconds
        : MUSIC_CONFIG.disengageThinSeconds;
    this.padGain.gain.rampTo(arousalToPadGain(audio.arousal), rampSeconds);
    this.pluckGain.gain.rampTo(arousalToPluckGain(audio.arousal), rampSeconds);
    this.density = arousalToPluckDensity(audio.arousal);
    if (this.density === 0) this.accumulator = 0;
    this.arousal = audio.arousal;

    if (audio.speaking !== this.speaking) {
      this.musicGain.gain.rampTo(
        speakingToDuckGain(audio.speaking),
        audio.speaking
          ? MUSIC_CONFIG.duckSeconds
          : MUSIC_CONFIG.recoverSeconds,
      );
      this.speaking = audio.speaking;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.scheduler.stop();
    for (const oscillator of this.padOscillators) oscillator.stop();
    for (const resource of this.resources) resource.dispose();
  }

  private schedulePluck(time: number): void {
    this.accumulator += this.density * MUSIC_CONFIG.schedulerTickSeconds;
    if (this.accumulator < 1) return;
    this.accumulator -= 1;
    const note = PLUCK_PATTERN[this.patternIndex % PLUCK_PATTERN.length];
    this.patternIndex += 1;
    if (note !== undefined) this.pluck.triggerAttack(note, time);
  }
}

let requestedState = normalizeBodyAudio({ arousal: 0, speaking: false });
let graph: LuxoMusicGraph | undefined;
let unlockPromise: Promise<void> | undefined;
let graphGeneration = 0;

export function setBodyAudio(audio: BodyAudioState): void {
  requestedState = normalizeBodyAudio(audio);
  graph?.setBodyAudio(requestedState);
}

export async function unlockMusic(): Promise<void> {
  const expectedGeneration = graphGeneration;
  unlockPromise ??= Tone.start().catch((error: unknown) => {
    unlockPromise = undefined;
    throw error;
  });
  await unlockPromise;
  if (expectedGeneration !== graphGeneration) return;
  graph ??= new LuxoMusicGraph(requestedState);
}

export function disposeMusic(): void {
  graphGeneration += 1;
  graph?.dispose();
  graph = undefined;
}

export const music = Object.freeze({
  unlock: unlockMusic,
  setBodyAudio,
  dispose: disposeMusic,
});
