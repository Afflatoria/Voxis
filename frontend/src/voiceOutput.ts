/**
 * Real-time processing between generated PCM and the speakers.
 *
 * Keep this chain separate from StreamPlayer so additional continuously
 * controlled processors (EQ, pitch shift, time stretch) can be inserted here.
 */
export interface VoiceEffects {
  energy: number;
  warmth: number;
  brightness: number;
  presence: number;
}

export const NEUTRAL_VOICE_EFFECTS: VoiceEffects = {
  energy: 1,
  warmth: 0,
  brightness: 0,
  presence: 0,
};

export class VoiceOutput {
  readonly input: GainNode;
  private readonly warmthFilter: BiquadFilterNode;
  private readonly presenceFilter: BiquadFilterNode;
  private readonly brightnessFilter: BiquadFilterNode;
  private readonly outputGain: GainNode;
  private readonly limiter: DynamicsCompressorNode;
  private readonly context: AudioContext;

  constructor(
    context: AudioContext,
    initialEffects: VoiceEffects = NEUTRAL_VOICE_EFFECTS,
  ) {
    this.context = context;
    this.input = context.createGain();
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

    this.input
      .connect(this.warmthFilter)
      .connect(this.presenceFilter)
      .connect(this.brightnessFilter)
      .connect(this.outputGain)
      .connect(this.limiter)
      .connect(context.destination);
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

  private ramp(parameter: AudioParam, target: number, seconds: number): void {
    const now = this.context.currentTime;
    parameter.cancelAndHoldAtTime(now);
    parameter.setTargetAtTime(target, now, Math.max(seconds / 3, 0.005));
  }
}
