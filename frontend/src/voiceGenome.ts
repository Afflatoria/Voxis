import type { VoiceEffects } from "./voiceOutput";

export interface VoiceGenome extends VoiceEffects {
  id: string;
  name: string;
  seed: number;
  speaking_rate: number;
}

const API_URL =
  (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL ??
  `http://${window.location.hostname}:8000`;

export const FALLBACK_GENOMES: [VoiceGenome, VoiceGenome] = [
  {
    id: "vg-default-a",
    name: "Voice A001",
    seed: 1001,
    speaking_rate: 0.94,
    energy: 1,
    warmth: 0.45,
    brightness: -0.2,
    presence: 0.15,
  },
  {
    id: "vg-default-b",
    name: "Voice B001",
    seed: 2001,
    speaking_rate: 1.06,
    energy: 0.96,
    warmth: -0.2,
    brightness: 0.42,
    presence: 0.3,
  },
];

export async function generateVoiceCandidates(
  seed: number,
  mutationStrength: number,
  parent?: VoiceGenome,
): Promise<VoiceGenome[]> {
  const response = await fetch(`${API_URL}/v1/voice-genomes/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      seed,
      count: 2,
      mutation_strength: mutationStrength,
      parent,
    }),
  });
  if (!response.ok) {
    throw new Error(`candidate API returned ${response.status}`);
  }
  return (await response.json()) as VoiceGenome[];
}
