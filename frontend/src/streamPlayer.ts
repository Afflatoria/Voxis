/** Web Audio streaming player with jitter buffering. */

import { decodeAudioFrame, int16ToFloat32 } from "./audioCodec";
import {
  NEUTRAL_VOICE_EFFECTS,
  VoiceOutput,
  type SpectrumFrame,
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
  private streamProcessor: AudioWorkletNode | null = null;
  private initialization: Promise<AudioContext> | null = null;
  private targetEffects: VoiceEffects = { ...NEUTRAL_VOICE_EFFECTS };
  private targetRate = 1;
  private targetPitchSemitones = 0;
  private started = false;
  private metrics: PlaybackMetrics = {
    requestTime: null,
    firstAudioReceived: null,
    playbackStarted: null,
    ttfaMs: null,
    chunkCount: 0,
    totalAudioSeconds: 0,
  };

  constructor(_bufferMs = 120) {}

  markRequestSent(): void {
    this.metrics.requestTime = performance.now();
  }

  getMetrics(): PlaybackMetrics {
    return { ...this.metrics };
  }

  async ensureContext(): Promise<AudioContext> {
    if (!this.initialization) {
      this.initialization = this.initializeAudio();
    }
    return this.initialization;
  }

  private async initializeAudio(): Promise<AudioContext> {
    const context = new AudioContext();
    await context.audioWorklet.addModule("/voice-stream-processor.js");
    this.audioContext = context;
    this.voiceOutput = new VoiceOutput(context, this.targetEffects);
    this.streamProcessor = new AudioWorkletNode(
      context,
      "voice-stream-processor",
      { outputChannelCount: [1] },
    );
    this.streamProcessor.connect(this.voiceOutput.input);
    this.sendPlaybackParameters();
    if (context.state === "suspended") {
      await context.resume();
    }
    return context;
  }

  reset(): void {
    this.streamProcessor?.port.postMessage({ type: "reset" });
    this.streamProcessor?.disconnect();
    this.streamProcessor = null;
    this.voiceOutput?.disconnect();
    this.voiceOutput = null;
    if (this.audioContext) {
      void this.audioContext.close();
      this.audioContext = null;
    }
    this.initialization = null;
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
    await this.ensureContext();
    const { sampleRate, pcm } = decodeAudioFrame(arrayBuffer);

    if (this.metrics.firstAudioReceived === null) {
      this.metrics.firstAudioReceived = performance.now();
    }

    const floats = int16ToFloat32(pcm);
    const transferable = new Float32Array(floats);
    this.streamProcessor!.port.postMessage(
      { type: "chunk", sampleRate, samples: transferable },
      [transferable.buffer],
    );

    if (!this.started) {
      this.started = true;
      this.metrics.playbackStarted = performance.now();
      if (this.metrics.requestTime !== null) {
        this.metrics.ttfaMs =
          this.metrics.playbackStarted - this.metrics.requestTime;
      }
    }

    this.metrics.chunkCount += 1;
    this.metrics.totalAudioSeconds += floats.length / sampleRate;
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

  setPlaybackTransform(rate: number, pitchSemitones: number): void {
    this.targetRate = Math.min(1.5, Math.max(0.65, rate));
    this.targetPitchSemitones = Math.min(8, Math.max(-8, pitchSemitones));
    this.sendPlaybackParameters();
  }

  private sendPlaybackParameters(): void {
    this.streamProcessor?.port.postMessage({
      type: "parameters",
      rate: this.targetRate,
      pitchSemitones: this.targetPitchSemitones,
    });
  }

  readSpectrum(): SpectrumFrame | null {
    return this.voiceOutput?.readSpectrum() ?? null;
  }
}
