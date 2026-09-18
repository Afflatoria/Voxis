/** Web Audio streaming player with jitter buffering. */

import { decodeAudioFrame, int16ToFloat32 } from "./audioCodec";
import {
  NEUTRAL_VOICE_EFFECTS,
  VoiceOutput,
  type VoiceEffects,
} from "./voiceOutput";

export interface PlaybackMetrics {
  requestTime: number | null;
  firstAudioReceived: number | null;
  playbackStarted: number | null;
  ttfaMs: number | null;
  chunkCount: number;
  totalAudioSeconds: number;
}

export class StreamPlayer {
  private audioContext: AudioContext | null = null;
  private voiceOutput: VoiceOutput | null = null;
  private targetEffects: VoiceEffects = { ...NEUTRAL_VOICE_EFFECTS };
  private nextStartTime = 0;
  private readonly bufferSeconds: number;
  private started = false;
  private sampleRate = 24000;
  private metrics: PlaybackMetrics = {
    requestTime: null,
    firstAudioReceived: null,
    playbackStarted: null,
    ttfaMs: null,
    chunkCount: 0,
    totalAudioSeconds: 0,
  };

  constructor(bufferMs = 120) {
    this.bufferSeconds = bufferMs / 1000;
  }

  markRequestSent(): void {
    this.metrics.requestTime = performance.now();
  }

  getMetrics(): PlaybackMetrics {
    return { ...this.metrics };
  }

  async ensureContext(): Promise<AudioContext> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
      this.voiceOutput = new VoiceOutput(this.audioContext, this.targetEffects);
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    return this.audioContext;
  }

  reset(): void {
    this.voiceOutput?.disconnect();
    this.voiceOutput = null;
    if (this.audioContext) {
      void this.audioContext.close();
      this.audioContext = null;
    }
    this.nextStartTime = 0;
    this.started = false;
    this.metrics = {
      requestTime: this.metrics.requestTime,
      firstAudioReceived: null,
      playbackStarted: null,
      ttfaMs: null,
      chunkCount: 0,
      totalAudioSeconds: 0,
    };
  }

  async enqueue(arrayBuffer: ArrayBuffer): Promise<void> {
    const ctx = await this.ensureContext();
    const { sampleRate, pcm } = decodeAudioFrame(arrayBuffer);
    this.sampleRate = sampleRate;

    if (this.metrics.firstAudioReceived === null) {
      this.metrics.firstAudioReceived = performance.now();
    }

    const floats = int16ToFloat32(pcm);
    const audioBuffer = ctx.createBuffer(1, floats.length, sampleRate);
    // Web Audio requires an ArrayBuffer-backed view; decoded data is typed as
    // ArrayBufferLike by newer TypeScript releases.
    audioBuffer.copyToChannel(new Float32Array(floats), 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.voiceOutput!.input);

    const now = ctx.currentTime;
    if (!this.started) {
      this.nextStartTime = now + this.bufferSeconds;
      this.started = true;
      this.metrics.playbackStarted = performance.now();
      if (this.metrics.requestTime !== null) {
        this.metrics.ttfaMs =
          this.metrics.playbackStarted - this.metrics.requestTime;
      }
    }

    const startAt = Math.max(this.nextStartTime, now + 0.01);
    source.start(startAt);
    this.nextStartTime = startAt + audioBuffer.duration;

    this.metrics.chunkCount += 1;
    this.metrics.totalAudioSeconds += audioBuffer.duration;
  }

  stop(): void {
    this.reset();
  }

  setEnergy(energy: number): void {
    this.setVoiceEffects({ energy });
  }

  setVoiceEffects(effects: Partial<VoiceEffects>): void {
    this.targetEffects = { ...this.targetEffects, ...effects };
    this.voiceOutput?.setEffects(effects);
  }
}
