import type { SpectrumFrame } from "./voiceOutput";

type SpectrumReader = () => SpectrumFrame | null;

export class SpectrogramRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly readSpectrum: SpectrumReader;
  private animationFrame: number | null = null;
  private lastDrawTime = 0;

  constructor(canvas: HTMLCanvasElement, readSpectrum: SpectrumReader) {
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("2D canvas is unavailable");
    this.canvas = canvas;
    this.context = context;
    this.readSpectrum = readSpectrum;
    this.context.fillStyle = "#090b10";
    this.context.fillRect(0, 0, canvas.width, canvas.height);
  }

  start(): void {
    if (this.animationFrame !== null) return;
    this.animationFrame = requestAnimationFrame(this.draw);
  }

  stop(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  private draw = (timestamp: number): void => {
    this.animationFrame = requestAnimationFrame(this.draw);
    if (timestamp - this.lastDrawTime < 33) return;
    this.lastDrawTime = timestamp;

    const spectrum = this.readSpectrum();
    if (!spectrum) return;

    const { width, height } = this.canvas;
    this.context.drawImage(this.canvas, -1, 0);
    const column = this.context.createImageData(1, height);
    const minimumFrequency = 70;
    const maximumFrequency = Math.min(10000, spectrum.sampleRate / 2);
    const binWidth = spectrum.sampleRate / spectrum.fftSize;

    for (let y = 0; y < height; y += 1) {
      const position = 1 - y / Math.max(1, height - 1);
      const frequency =
        minimumFrequency *
        (maximumFrequency / minimumFrequency) ** position;
      const bin = Math.min(
        spectrum.magnitudes.length - 1,
        Math.max(0, Math.round(frequency / binWidth)),
      );
      const intensity = spectrum.magnitudes[bin] / 255;
      const [red, green, blue] = this.color(intensity);
      const offset = y * 4;
      column.data[offset] = red;
      column.data[offset + 1] = green;
      column.data[offset + 2] = blue;
      column.data[offset + 3] = 255;
    }
    this.context.putImageData(column, width - 1, 0);
  };

  private color(intensity: number): [number, number, number] {
    if (intensity < 0.18) {
      const amount = intensity / 0.18;
      return [
        Math.round(9 + 24 * amount),
        Math.round(11 + 8 * amount),
        Math.round(17 + 42 * amount),
      ];
    }
    if (intensity < 0.55) {
      const amount = (intensity - 0.18) / 0.37;
      return [
        Math.round(33 + 106 * amount),
        Math.round(19 + 73 * amount),
        Math.round(59 + 137 * amount),
      ];
    }
    if (intensity < 0.82) {
      const amount = (intensity - 0.55) / 0.27;
      return [
        Math.round(139 - 76 * amount),
        Math.round(92 + 117 * amount),
        Math.round(196 + 7 * amount),
      ];
    }
    const amount = (intensity - 0.82) / 0.18;
    return [
      Math.round(63 + 192 * amount),
      Math.round(209 + 36 * amount),
      Math.round(203 - 111 * amount),
    ];
  }
}
