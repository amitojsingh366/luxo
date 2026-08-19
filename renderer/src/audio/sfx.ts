import * as Tone from "tone";

export const SFX_NAMES = Object.freeze([
  "chirp_up",
  "chirp_found",
  "boing",
  "whirr_short",
  "hmm",
  "blip_sad",
  "fanfare_small",
  "click",
] as const);

export type SfxName = (typeof SFX_NAMES)[number];

export const SFX_RECIPES = Object.freeze({
  chirp_up: Object.freeze({
    durationMs: 120,
    source: "oscillator",
    waveform: "sine",
    sweepHz: Object.freeze([440, 880] as const),
    envelope: "exponential_decay",
  }),
  chirp_found: Object.freeze({
    durationMs: 180,
    source: "pluck",
    scale: "major_pentatonic",
    frequenciesHz: Object.freeze([523.25, 659.25] as const),
    offsetsMs: Object.freeze([0, 90] as const),
  }),
  boing: Object.freeze({
    durationMs: 260,
    source: "oscillator",
    waveform: "sine",
    frequencyHz: 200,
    envelope: "damped",
    pitchEnvelope: Object.freeze({ octaves: 1.25, decayMs: 130 }),
  }),
  whirr_short: Object.freeze({
    durationMs: 200,
    sources: Object.freeze([
      Object.freeze({ source: "filtered_noise", filterHz: 700 }),
      Object.freeze({ source: "oscillator", waveform: "sine", frequencyHz: 90 }),
    ] as const),
    character: "servo",
  }),
  hmm: Object.freeze({
    durationMs: 400,
    source: "oscillator",
    waveform: "sine",
    frequencyHz: 160,
    vibrato: Object.freeze({ rateHz: 4, depth: 0.04 }),
  }),
  blip_sad: Object.freeze({
    durationMs: 200,
    source: "oscillator",
    waveform: "sine",
    sweepHz: Object.freeze([500, 300] as const),
    direction: "descending",
  }),
  fanfare_small: Object.freeze({
    durationMs: 500,
    source: "oscillator",
    waveform: "sine",
    harmony: "major_arpeggio",
    frequenciesHz: Object.freeze([523.25, 659.25, 783.99] as const),
    offsetsMs: Object.freeze([0, 160, 320] as const),
    noteDurationsMs: Object.freeze([150, 150, 180] as const),
  }),
  click: Object.freeze({
    durationMs: 8,
    source: "noise",
    noise: "white",
    envelope: "sharp",
  }),
} as const satisfies Record<SfxName, Readonly<Record<string, unknown>>>);

export function isSfxName(name: string): name is SfxName {
  return Object.prototype.hasOwnProperty.call(SFX_RECIPES, name);
}

export function assertSfxName(name: string): asserts name is SfxName {
  if (!isSfxName(name)) {
    throw new RangeError(`Unknown Luxo SFX: ${name}`);
  }
}

interface Disposable {
  dispose(): unknown;
}

function seconds(milliseconds: number): number {
  return milliseconds / 1_000;
}

interface EnvelopeShape {
  attack: number;
  decay: number;
  sustain: number;
  release: number;
}

const DEFAULT_ENVELOPE = Object.freeze({
  attack: 0.004,
  decay: 0.08,
  sustain: 0.15,
  release: 0.04,
});

function makeSineSynth(
  volume: number,
  envelope: EnvelopeShape = DEFAULT_ENVELOPE,
): Tone.Synth {
  return new Tone.Synth({
    oscillator: { type: "sine" },
    envelope: {
      attack: envelope.attack,
      decay: envelope.decay,
      decayCurve: "exponential",
      sustain: envelope.sustain,
      release: envelope.release,
    },
    volume,
  });
}

class LuxoSfxGraph {
  private readonly master = new Tone.Gain(0.32).toDestination();
  private readonly chirpUp = makeSineSynth(-10, {
    attack: 0.003,
    decay: 0.116,
    sustain: 0,
    release: 0.001,
  }).connect(this.master);
  private readonly chirpFound = [
    new Tone.PluckSynth({
      attackNoise: 0.7,
      dampening: 3_500,
      resonance: 0.76,
      release: 0.08,
      volume: -13,
    }).connect(this.master),
    new Tone.PluckSynth({
      attackNoise: 0.7,
      dampening: 3_500,
      resonance: 0.76,
      release: 0.08,
      volume: -13,
    }).connect(this.master),
  ] as const;
  private readonly boing = new Tone.MembraneSynth({
    pitchDecay: seconds(SFX_RECIPES.boing.pitchEnvelope.decayMs),
    octaves: SFX_RECIPES.boing.pitchEnvelope.octaves,
    oscillator: { type: "sine" },
    envelope: {
      attack: 0.004,
      decay: 0.251,
      decayCurve: "exponential",
      sustain: 0,
      release: 0.005,
    },
    volume: -11,
  }).connect(this.master);
  private readonly whirrFilter = new Tone.Filter({
    frequency: SFX_RECIPES.whirr_short.sources[0].filterHz,
    type: "lowpass",
    rolloff: -24,
    Q: 4,
  }).connect(this.master);
  private readonly whirrNoise = new Tone.NoiseSynth({
    noise: { type: "pink" },
    envelope: { attack: 0.025, decay: 0.14, sustain: 0.12, release: 0.035 },
    volume: -22,
  }).connect(this.whirrFilter);
  private readonly whirrHum = makeSineSynth(-18, {
    attack: 0.025,
    decay: 0.1,
    sustain: 0.12,
    release: 0.035,
  }).connect(this.master);
  private readonly hmmVibrato = new Tone.Vibrato({
    frequency: SFX_RECIPES.hmm.vibrato.rateHz,
    depth: SFX_RECIPES.hmm.vibrato.depth,
    wet: 1,
  }).connect(this.master);
  private readonly hmm = makeSineSynth(-16).connect(this.hmmVibrato);
  private readonly blipSad = makeSineSynth(-13, {
    attack: 0.004,
    decay: 0.195,
    sustain: 0,
    release: 0.001,
  }).connect(this.master);
  private readonly fanfare = [
    makeSineSynth(-15, { ...DEFAULT_ENVELOPE, release: 0.03 }).connect(this.master),
    makeSineSynth(-15, { ...DEFAULT_ENVELOPE, release: 0.03 }).connect(this.master),
    makeSineSynth(-15, { ...DEFAULT_ENVELOPE, release: 0.03 }).connect(this.master),
  ] as const;
  private readonly click = new Tone.NoiseSynth({
    noise: { type: "white" },
    envelope: { attack: 0.001, decay: 0.006, sustain: 0, release: 0.001 },
    volume: -19,
  }).connect(this.master);
  private readonly resources: readonly Disposable[];
  private disposed = false;

  constructor() {
    this.resources = [
      this.chirpUp,
      ...this.chirpFound,
      this.boing,
      this.whirrNoise,
      this.whirrHum,
      this.hmm,
      this.blipSad,
      ...this.fanfare,
      this.click,
      this.whirrFilter,
      this.hmmVibrato,
      this.master,
    ];
  }

  play(name: SfxName): void {
    if (this.disposed) {
      throw new Error("Luxo SFX graph has been disposed");
    }

    const at = Tone.now() + 0.005;
    switch (name) {
      case "chirp_up":
        this.playChirpUp(at);
        break;
      case "chirp_found":
        this.playChirpFound(at);
        break;
      case "boing":
        this.boing.triggerAttackRelease(
          SFX_RECIPES.boing.frequencyHz,
          seconds(SFX_RECIPES.boing.durationMs - 5),
          at,
          0.72,
        );
        break;
      case "whirr_short":
        this.playWhirr(at);
        break;
      case "hmm":
        this.hmm.triggerAttackRelease(
          SFX_RECIPES.hmm.frequencyHz,
          seconds(SFX_RECIPES.hmm.durationMs - 40),
          at,
          0.5,
        );
        break;
      case "blip_sad":
        this.playSadBlip(at);
        break;
      case "fanfare_small":
        this.playFanfare(at);
        break;
      case "click":
        this.click.triggerAttackRelease(
          seconds(SFX_RECIPES.click.durationMs - 1),
          at,
          0.55,
        );
        break;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const resource of this.resources) resource.dispose();
  }

  private playChirpUp(at: number): void {
    const [startHz, endHz] = SFX_RECIPES.chirp_up.sweepHz;
    const duration = seconds(SFX_RECIPES.chirp_up.durationMs);
    this.chirpUp.frequency.cancelAndHoldAtTime(at);
    this.chirpUp.triggerAttack(startHz, at, 0.65);
    this.chirpUp.frequency.exponentialRampTo(endHz, duration, at);
    this.chirpUp.triggerRelease(at + duration - 0.001);
  }

  private playChirpFound(at: number): void {
    for (let index = 0; index < this.chirpFound.length; index += 1) {
      const voice = this.chirpFound[index];
      const frequency = SFX_RECIPES.chirp_found.frequenciesHz[index];
      const offset = SFX_RECIPES.chirp_found.offsetsMs[index];
      if (voice === undefined || frequency === undefined || offset === undefined) {
        continue;
      }
      const noteAt = at + seconds(offset);
      voice.triggerAttack(frequency, noteAt);
      voice.triggerRelease(noteAt + 0.01);
    }
  }

  private playWhirr(at: number): void {
    const duration = seconds(SFX_RECIPES.whirr_short.durationMs);
    this.whirrNoise.triggerAttackRelease(duration - 0.035, at, 0.48);
    this.whirrHum.triggerAttackRelease(
      SFX_RECIPES.whirr_short.sources[1].frequencyHz,
      duration - 0.035,
      at,
      0.38,
    );
  }

  private playSadBlip(at: number): void {
    const [startHz, endHz] = SFX_RECIPES.blip_sad.sweepHz;
    const duration = seconds(SFX_RECIPES.blip_sad.durationMs);
    this.blipSad.frequency.cancelAndHoldAtTime(at);
    this.blipSad.triggerAttack(startHz, at, 0.58);
    this.blipSad.frequency.exponentialRampTo(endHz, duration, at);
    this.blipSad.triggerRelease(at + duration - 0.001);
  }

  private playFanfare(at: number): void {
    for (let index = 0; index < this.fanfare.length; index += 1) {
      const voice = this.fanfare[index];
      const frequency = SFX_RECIPES.fanfare_small.frequenciesHz[index];
      const offsetMs = SFX_RECIPES.fanfare_small.offsetsMs[index];
      const durationMs = SFX_RECIPES.fanfare_small.noteDurationsMs[index];
      if (
        voice === undefined ||
        frequency === undefined ||
        offsetMs === undefined ||
        durationMs === undefined
      ) {
        continue;
      }
      voice.triggerAttackRelease(
        frequency,
        seconds(durationMs - 30),
        at + seconds(offsetMs),
        0.58,
      );
    }
  }
}

export interface SfxPlayer {
  play(name: SfxName): Promise<void>;
  dispose(): void;
}

let graph: LuxoSfxGraph | undefined;
let unlockPromise: Promise<void> | undefined;
let graphGeneration = 0;

async function unlockAudio(): Promise<void> {
  unlockPromise ??= Tone.start().catch((error: unknown) => {
    unlockPromise = undefined;
    throw error;
  });
  await unlockPromise;
}

export const sfx: SfxPlayer = Object.freeze({
  async play(name: SfxName): Promise<void> {
    assertSfxName(name);
    const expectedGeneration = graphGeneration;
    await unlockAudio();
    if (expectedGeneration !== graphGeneration) return;
    graph ??= new LuxoSfxGraph();
    graph.play(name);
  },
  dispose(): void {
    graphGeneration += 1;
    graph?.dispose();
    graph = undefined;
  },
});
