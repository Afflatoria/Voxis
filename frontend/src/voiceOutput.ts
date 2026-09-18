/**
 * Real-time processing between generated PCM and the speakers.
 *
 * Keep this chain separate from StreamPlayer so additional continuously
 * controlled processors (EQ, pitch shift, time stretch) can be inserted here.
 */
export interface VoiceEffects {
  energy: number;
  vocal_size: number;
  formant_strength: number;
  warmth: number;
  brightness: number;
  presence: number;
}

export const NEUTRAL_VOICE_EFFECTS: VoiceEffects = {
  energy: 1,
  vocal_size: 0,
  formant_strength: 0.6,
  warmth: 0,
  brightness: 0,
  presence: 0,
};

export class VoiceOutput {
  readonly input: GainNode;
  private readonly formantCuts: BiquadFilterNode[] = [];
  private readonly formantBoosts: BiquadFilterNode[] = [];
  private readonly warmthFilter: BiquadFilterNode;
  private readonly presenceFilter: BiquadFilterNode;
  private readonly brightnessFilter: BiquadFilterNode;
  private readonly outputGain: GainNode;
  private readonly limiter: DynamicsCompressorNode;
  private readonly context: AudioContext;
  private vocalSize = 0;
  private formantStrength = 0.6;

  constructor(
    context: AudioContext,
    initialEffects: VoiceEffects = NEUTRAL_VOICE_EFFECTS,
  ) {
    this.context = context;
    this.input = context.createGain();
    this.vocalSize = initialEffects.vocal_size;
    this.formantStrength = initialEffects.formant_strength;
    const formantFrequencies = [650, 1200, 2600];
    const formantQs = [1.4, 1.8, 2.2];
    for (let index = 0; index < formantFrequencies.length; index += 1) {
      const cut = context.createBiquadFilter();
      cut.type = "peaking";
      cut.frequency.value = formantFrequencies[index];
      cut.Q.value = formantQs[index];
      const boost = context.createBiquadFilter();
      boost.type = "peaking";
      boost.frequency.value = this.shiftedFormant(
        formantFrequencies[index],
        this.vocalSize,
      );
      boost.Q.value = formantQs[index];
      this.formantCuts.push(cut);
      this.formantBoosts.push(boost);
    }
    this.warmthFilter = context.createBiquadFilter();
    this.warmthFilter.type = "lowshelf";
    this.warmthFilter.frequency.value = 320;
    this.presenceFilter = context.createBiquadFilter();
    this.presenceFilter.type = "peaking";
    this.presenceFilter.frequency.value = 1500;
    this.presenceFilter.Q.value = 0.8;
    this.brightnessFilter = context.createBiquadFilter();
    this.brightnessFilter.type = "highshelf";
    this.brightnessFilter.frequency.value = 3200;
    this.outputGain = context.createGain();
    this.limiter = context.createDynamicsCompressor();
    this.limiter.threshold.value = -3;
    this.limiter.knee.value = 4;
    this.limiter.ratio.value = 8;
    this.limiter.attack.value = 0.004;
    this.limiter.release.value = 0.08;

    this.outputGain.gain.value = this.clampEnergy(initialEffects.energy);
    this.warmthFilter.gain.value = this.clampUnit(initialEffects.warmth) * 9;
    this.brightnessFilter.gain.value =
      this.clampUnit(initialEffects.brightness) * 9;
    this.presenceFilter.gain.value = this.clampUnit(initialEffects.presence) * 7;

    let tail: AudioNode = this.input;
    for (let index = 0; index < this.formantCuts.length; index += 1) {
      tail = tail
        .connect(this.formantCuts[index])
        .connect(this.formantBoosts[index]);
    }
    tail
      .connect(this.warmthFilter)
      .connect(this.presenceFilter)
      .connect(this.brightnessFilter)
      .connect(this.outputGain)
      .connect(this.limiter)
      .connect(context.destination);
    this.updateFormants(0.01);
  }

  setEnergy(energy: number, transitionSeconds = 0.08): void {
    this.ramp(
      this.outputGain.gain,
      this.clampEnergy(energy),
      transitionSeconds,
    );
  }

  setEffects(effects: Partial<VoiceEffects>, transitionSeconds = 0.1): void {
    if (effects.energy !== undefined) {
      this.setEnergy(effects.energy, transitionSeconds);
    }
    if (effects.vocal_size !== undefined) {
      this.vocalSize = this.clampUnit(effects.vocal_size);
    }
    if (effects.formant_strength !== undefined) {
      this.formantStrength = Math.min(
        1,
        Math.max(0, effects.formant_strength),
      );
    }
    if (
      effects.vocal_size !== undefined ||
      effects.formant_strength !== undefined
    ) {
      this.updateFormants(transitionSeconds);
    }
    if (effects.warmth !== undefined) {
      this.ramp(
        this.warmthFilter.gain,
        this.clampUnit(effects.warmth) * 9,
        transitionSeconds,
      );
    }
    if (effects.brightness !== undefined) {
      this.ramp(
        this.brightnessFilter.gain,
        this.clampUnit(effects.brightness) * 9,
        transitionSeconds,
      );
    }
    if (effects.presence !== undefined) {
      this.ramp(
        this.presenceFilter.gain,
        this.clampUnit(effects.presence) * 7,
        transitionSeconds,
      );
    }
  }

  disconnect(): void {
    this.input.disconnect();
    this.formantCuts.forEach((node) => node.disconnect());
    this.formantBoosts.forEach((node) => node.disconnect());
    this.warmthFilter.disconnect();
    this.presenceFilter.disconnect();
    this.brightnessFilter.disconnect();
    this.outputGain.disconnect();
    this.limiter.disconnect();
  }

  private clampEnergy(value: number): number {
    return Math.min(2, Math.max(0, value));
  }

  private clampUnit(value: number): number {
    return Math.min(1, Math.max(-1, value));
  }

  private shiftedFormant(baseFrequency: number, vocalSize: number): number {
    // Positive size models a longer tract and therefore lower resonances.
    return baseFrequency * 2 ** (-this.clampUnit(vocalSize) * 0.24);
  }

  private updateFormants(transitionSeconds: number): void {
    const depth = Math.abs(this.vocalSize) * this.formantStrength;
    const bases = [650, 1200, 2600];
    for (let index = 0; index < bases.length; index += 1) {
      this.ramp(
        this.formantCuts[index].gain,
        -4 * depth,
        transitionSeconds,
      );
      this.ramp(
        this.formantBoosts[index].frequency,
        this.shiftedFormant(bases[index], this.vocalSize),
        transitionSeconds,
      );
      this.ramp(
        this.formantBoosts[index].gain,
        6 * depth,
        transitionSeconds,
      );
    }
  }

  private ramp(parameter: AudioParam, target: number, seconds: number): void {
    const now = this.context.currentTime;
    parameter.cancelAndHoldAtTime(now);
    parameter.setTargetAtTime(target, now, Math.max(seconds / 3, 0.005));
  }
}
