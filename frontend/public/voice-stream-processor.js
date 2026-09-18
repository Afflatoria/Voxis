/* global AudioWorkletProcessor, registerProcessor, sampleRate */

/**
 * Streaming mono PCM processor.
 *
 * Stage 1 reads the input ring at a variable rate (time compression/expansion).
 * Stage 2 uses overlapping variable-delay grains to compensate that rate's
 * pitch change or apply an independent pitch offset.
 */
class VoiceStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacity = Math.ceil(sampleRate * 180);
    this.inputRing = new Float32Array(this.capacity);
    this.writeIndex = 0;
    this.readPosition = 0;
    this.inputSampleRate = 24000;
    this.started = false;
    this.initialBufferSeconds = 0.12;

    this.targetRate = 1;
    this.currentRate = 1;
    this.targetPitchSemitones = 0;
    this.currentPitchSemitones = 0;

    this.pitchCapacity = 8192;
    this.pitchRing = new Float32Array(this.pitchCapacity);
    this.pitchWriteIndex = 0;
    this.grainPhase = 0;
    this.grainSize = 1536;
    this.minimumDelay = 96;

    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "chunk") {
        this.inputSampleRate = message.sampleRate;
        this.append(message.samples);
      } else if (message.type === "parameters") {
        if (message.rate !== undefined) {
          this.targetRate = this.clamp(message.rate, 0.65, 1.5);
        }
        if (message.pitchSemitones !== undefined) {
          this.targetPitchSemitones = this.clamp(
            message.pitchSemitones,
            -8,
            8,
          );
        }
      } else if (message.type === "reset") {
        this.reset();
      }
    };
  }

  append(samples) {
    const unread = this.writeIndex - Math.floor(this.readPosition);
    const overflow = unread + samples.length - this.capacity;
    if (overflow > 0) {
      this.readPosition += overflow;
    }
    for (let index = 0; index < samples.length; index += 1) {
      this.inputRing[this.writeIndex % this.capacity] = samples[index];
      this.writeIndex += 1;
    }
  }

  reset() {
    this.writeIndex = 0;
    this.readPosition = 0;
    this.started = false;
    this.pitchWriteIndex = 0;
    this.grainPhase = 0;
    this.inputRing.fill(0);
    this.pitchRing.fill(0);
  }

  readInput(position) {
    const index = Math.floor(position);
    const fraction = position - index;
    const first = this.inputRing[index % this.capacity];
    const second = this.inputRing[(index + 1) % this.capacity];
    return first + (second - first) * fraction;
  }

  readPitchHistory(delay) {
    let position = this.pitchWriteIndex - delay;
    while (position < 0) position += this.pitchCapacity;
    const index = Math.floor(position);
    const fraction = position - index;
    const first = this.pitchRing[index % this.pitchCapacity];
    const second = this.pitchRing[(index + 1) % this.pitchCapacity];
    return first + (second - first) * fraction;
  }

  pitchShift(input, factor) {
    this.pitchRing[this.pitchWriteIndex % this.pitchCapacity] = input;
    this.pitchWriteIndex += 1;

    const difference = Math.abs(1 - factor);
    if (difference < 0.0005) {
      return this.readPitchHistory(this.minimumDelay);
    }

    const phaseA = this.grainPhase;
    const phaseB = (phaseA + 0.5) % 1;
    const direction = factor > 1 ? -1 : 1;
    const delayA =
      this.minimumDelay +
      (direction > 0 ? phaseA : 1 - phaseA) * this.grainSize;
    const delayB =
      this.minimumDelay +
      (direction > 0 ? phaseB : 1 - phaseB) * this.grainSize;
    const windowA = 0.5 - 0.5 * Math.cos(2 * Math.PI * phaseA);
    const windowB = 1 - windowA;
    const output =
      this.readPitchHistory(delayA) * windowA +
      this.readPitchHistory(delayB) * windowB;

    this.grainPhase =
      (this.grainPhase + difference / this.grainSize) % 1;
    return output;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;

    const initialFrames = this.inputSampleRate * this.initialBufferSeconds;
    if (!this.started && this.writeIndex - this.readPosition >= initialFrames) {
      this.started = true;
    }

    for (let index = 0; index < output.length; index += 1) {
      // Around 50 ms smoothing at common hardware sample rates.
      this.currentRate += (this.targetRate - this.currentRate) * 0.0005;
      this.currentPitchSemitones +=
        (this.targetPitchSemitones - this.currentPitchSemitones) * 0.0005;

      const sourceStep =
        (this.inputSampleRate / sampleRate) * this.currentRate;
      if (!this.started || this.readPosition + sourceStep + 1 >= this.writeIndex) {
        output[index] = 0;
        this.started = false;
        continue;
      }

      const resampled = this.readInput(this.readPosition);
      this.readPosition += sourceStep;

      const desiredPitch = 2 ** (this.currentPitchSemitones / 12);
      const compensationFactor = desiredPitch / this.currentRate;
      output[index] = this.pitchShift(resampled, compensationFactor);
    }
    return true;
  }

  clamp(value, lower, upper) {
    return Math.min(upper, Math.max(lower, Number(value)));
  }
}

registerProcessor("voice-stream-processor", VoiceStreamProcessor);
