/** WebSocket client for /v1/stream. */

import { StreamPlayer } from "./streamPlayer";
import type { VoiceEffects } from "./voiceOutput";

export interface VoiceSettings {
  language: string;
  speaker: string | null;
  speaking_rate: number;
  pitch: number;
  energy: number;
  accent?: string | null;
  accent_strength?: number;
}

export type StatusHandler = (status: string) => void;
export type MetricsHandler = (metrics: Record<string, unknown>) => void;
export type VoiceUpdateHandler = (version: number) => void;

export class TTSWebSocketClient {
  private ws: WebSocket | null = null;
  private player = new StreamPlayer(120);
  private onStatus: StatusHandler;
  private onMetrics: MetricsHandler;
  private onVoiceUpdate: VoiceUpdateHandler;
  private url: string;

  constructor(
    url: string,
    onStatus: StatusHandler,
    onMetrics: MetricsHandler,
    onVoiceUpdate: VoiceUpdateHandler = () => undefined,
  ) {
    this.url = url;
    this.onStatus = onStatus;
    this.onMetrics = onMetrics;
    this.onVoiceUpdate = onVoiceUpdate;
  }

  private ensureSocket(): WebSocket {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      return this.ws;
    }
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => this.onStatus("Connected");
    this.ws.onclose = () => this.onStatus("Disconnected");
    this.ws.onerror = () => this.onStatus("Connection error");

    this.ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        if (msg.type === "ready") {
          this.onStatus("Ready");
        } else if (msg.type === "started") {
          this.onStatus("Streaming");
        } else if (msg.type === "complete") {
          this.onStatus("Complete");
          const clientMetrics = this.player.getMetrics();
          this.onMetrics({
            ...msg.metrics,
            client_ttfa_ms: clientMetrics.ttfaMs,
            client_chunk_count: clientMetrics.chunkCount,
            client_total_audio_seconds: clientMetrics.totalAudioSeconds,
          });
        } else if (msg.type === "cancelled") {
          this.onStatus("Cancelled");
        } else if (msg.type === "voice_updated") {
          this.onVoiceUpdate(Number(msg.voice_version));
        } else if (msg.type === "error") {
          this.onStatus(`Error: ${msg.message}`);
        }
      } else if (event.data instanceof ArrayBuffer) {
        await this.player.enqueue(event.data);
      }
    };

    return this.ws;
  }

  async connect(): Promise<void> {
    const ws = this.ensureSocket();
    if (ws.readyState === WebSocket.OPEN) return;
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("connect timeout")), 5000);
      ws.addEventListener("open", () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
      ws.addEventListener("error", () => {
        clearTimeout(timeout);
        reject(new Error("websocket error"));
      }, { once: true });
    });
  }

  async generate(
    text: string,
    voice: VoiceSettings,
    effects: Partial<VoiceEffects> = {},
  ): Promise<void> {
    await this.connect();
    this.player.reset();
    this.player.setVoiceEffects({ ...effects, energy: voice.energy });
    this.player.markRequestSent();
    const backendVoice = { ...voice, energy: 1 };
    this.ws!.send(
      JSON.stringify({
        type: "start",
        text,
        voice: backendVoice,
        request_id: crypto.randomUUID(),
      }),
    );
  }

  stop(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
    this.player.stop();
    this.onStatus("Stopped");
  }

  updateVoice(voice: Partial<VoiceSettings>): void {
    if (voice.energy !== undefined) {
      this.player.setEnergy(voice.energy);
    }

    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const backendVoice = { ...voice };
    delete backendVoice.energy;
    if (Object.keys(backendVoice).length > 0) {
      this.ws.send(
        JSON.stringify({ type: "voice_update", voice: backendVoice }),
      );
    }
  }

  updateOutput(effects: Partial<VoiceEffects>): void {
    this.player.setVoiceEffects(effects);
  }
}
