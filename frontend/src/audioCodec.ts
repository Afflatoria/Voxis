/** Decode binary audio frames from the WebSocket server. */

const AUDIO_FRAME_MAGIC = new Uint8Array([0x4e, 0x54, 0x54, 0x53]); // "NTTS"
const AUDIO_HEADER_SIZE = 16;

export interface DecodedAudioChunk {
  sequence: number;
  sampleRate: number;
  pcm: Int16Array;
}

export function decodeAudioFrame(buffer: ArrayBuffer): DecodedAudioChunk {
  const view = new DataView(buffer);
  for (let i = 0; i < 4; i++) {
    if (view.getUint8(i) !== AUDIO_FRAME_MAGIC[i]) {
      throw new Error("invalid audio frame magic");
    }
  }
  const version = view.getUint8(4);
  if (version !== 1) {
    throw new Error(`unsupported audio frame version: ${version}`);
  }
  const sequence = view.getUint32(8, true);
  const sampleRate = view.getUint32(12, true);
  const pcmBytes = buffer.slice(AUDIO_HEADER_SIZE);
  const pcm = new Int16Array(pcmBytes);
  return { sequence, sampleRate, pcm };
}

export function int16ToFloat32(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    out[i] = pcm[i] / 32768;
  }
  return out;
}
