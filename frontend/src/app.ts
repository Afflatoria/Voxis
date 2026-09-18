import {
  FALLBACK_GENOMES,
  generateVoiceCandidates,
  type VoiceGenome,
} from "./voiceGenome";
import { TTSWebSocketClient, type VoiceSettings } from "./wsClient";

const WS_URL =
  (import.meta as { env?: { VITE_WS_URL?: string } }).env?.VITE_WS_URL ??
  `ws://${window.location.hostname}:8000/v1/stream`;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function rangeRow(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
) {
  const wrap = el("div", "range-row");
  const labelNode = el("label");
  const output = el("output");
  output.textContent = value.toFixed(2);
  labelNode.textContent = label;
  labelNode.appendChild(output);

  const input = el("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.addEventListener("input", () => {
    output.textContent = Number(input.value).toFixed(2);
  });

  wrap.append(labelNode, input);
  return { wrap, input, output };
}

interface ComparisonSlot {
  name: "A" | "B";
  genome: VoiceGenome;
  settings: VoiceSettings;
  card: HTMLElement;
  heading: HTMLElement;
  genomeId: HTMLElement;
  rate: ReturnType<typeof rangeRow>;
  pitch: ReturnType<typeof rangeRow>;
  energy: ReturnType<typeof rangeRow>;
  warmth: ReturnType<typeof rangeRow>;
  brightness: ReturnType<typeof rangeRow>;
  presence: ReturnType<typeof rangeRow>;
  generateButton: HTMLButtonElement;
  metrics: HTMLElement;
  applyGenome: (genome: VoiceGenome) => void;
}

function setRangeValue(
  row: ReturnType<typeof rangeRow>,
  value: number,
): void {
  row.input.value = String(value);
  row.output.textContent = value.toFixed(2);
}

function buildSlot(
  name: "A" | "B",
  initialGenome: VoiceGenome,
  onGenerate: () => void,
  onSave: () => void,
  onVoiceChange: (slot: ComparisonSlot) => void,
): ComparisonSlot {
  const card = el("section", "voice-card");
  const header = el("div", "voice-card-header");
  const slotBadge = el("span", `slot-badge slot-${name.toLowerCase()}`);
  slotBadge.textContent = name;
  const headingWrap = el("div");
  const heading = el("h2");
  heading.textContent = initialGenome.name;
  const subheading = el("p", "card-subtitle");
  subheading.textContent = initialGenome.id;
  headingWrap.append(heading, subheading);
  header.append(slotBadge, headingWrap);

  const rate = rangeRow(
    "Speaking rate",
    0.75,
    1.3,
    0.01,
    initialGenome.speaking_rate,
  );
  const energy = rangeRow(
    "Energy",
    0.6,
    1.4,
    0.01,
    initialGenome.energy,
  );
  const pitch = rangeRow(
    "Pitch (semitones)",
    -6,
    6,
    0.1,
    initialGenome.pitch_semitones,
  );
  const warmth = rangeRow("Warmth", -1, 1, 0.01, initialGenome.warmth);
  const brightness = rangeRow(
    "Brightness",
    -1,
    1,
    0.01,
    initialGenome.brightness,
  );
  const presence = rangeRow("Presence", -1, 1, 0.01, initialGenome.presence);

  const generateButton = el("button");
  generateButton.textContent = `Play voice ${name}`;
  generateButton.addEventListener("click", onGenerate);
  const saveButton = el("button", "secondary");
  saveButton.textContent = "Save identity";
  saveButton.addEventListener("click", onSave);
  const actions = el("div", "card-actions");
  actions.append(generateButton, saveButton);
  const metrics = el("div", "slot-metrics");
  metrics.textContent = "Not generated yet";

  const slot: ComparisonSlot = {
    name,
    genome: { ...initialGenome },
    settings: {
      language: "English",
      speaker: "default",
      speaking_rate: initialGenome.speaking_rate,
      pitch: 0,
      energy: initialGenome.energy,
    },
    card,
    heading,
    genomeId: subheading,
    rate,
    pitch,
    energy,
    warmth,
    brightness,
    presence,
    generateButton,
    metrics,
    applyGenome: () => undefined,
  };

  slot.applyGenome = (genome) => {
    slot.genome = { ...genome };
    slot.settings.speaking_rate = genome.speaking_rate;
    slot.settings.energy = genome.energy;
    heading.textContent = genome.name;
    subheading.textContent = genome.id;
    setRangeValue(rate, genome.speaking_rate);
    setRangeValue(pitch, genome.pitch_semitones);
    setRangeValue(energy, genome.energy);
    setRangeValue(warmth, genome.warmth);
    setRangeValue(brightness, genome.brightness);
    setRangeValue(presence, genome.presence);
    metrics.textContent = "New candidate · ready to audition";
    onVoiceChange(slot);
  };

  const markCustom = () => {
    slot.genome.name = `Custom ${name}`;
    heading.textContent = slot.genome.name;
  };
  rate.input.addEventListener("input", () => {
    markCustom();
    slot.genome.speaking_rate = Number(rate.input.value);
    slot.settings.speaking_rate = Number(rate.input.value);
    onVoiceChange(slot);
  });
  energy.input.addEventListener("input", () => {
    markCustom();
    slot.genome.energy = Number(energy.input.value);
    slot.settings.energy = Number(energy.input.value);
    onVoiceChange(slot);
  });
  pitch.input.addEventListener("input", () => {
    markCustom();
    slot.genome.pitch_semitones = Number(pitch.input.value);
    onVoiceChange(slot);
  });
  for (const [row, key] of [
    [warmth, "warmth"],
    [brightness, "brightness"],
    [presence, "presence"],
  ] as const) {
    row.input.addEventListener("input", () => {
      markCustom();
      slot.genome[key] = Number(row.input.value);
      onVoiceChange(slot);
    });
  }

  card.append(
    header,
    rate.wrap,
    pitch.wrap,
    energy.wrap,
    warmth.wrap,
    brightness.wrap,
    presence.wrap,
    actions,
    metrics,
  );
  return slot;
}

export function mountApp(root: HTMLElement): void {
  root.innerHTML = "";
  const container = el("main", "container");
  const eyebrow = el("div", "eyebrow");
  eyebrow.textContent = "VOICE GENOME LAB · PHASE 1";
  const title = el("h1");
  title.textContent = "Evolve a voice of your own.";
  const intro = el("p", "intro");
  intro.textContent =
    "Audition two artificial identities, choose a parent, then generate nearby mutations until the voice feels right.";
  container.append(eyebrow, title, intro);

  const textLabel = el("label", "section-label");
  textLabel.textContent = "Script";
  const textarea = el("textarea");
  textarea.placeholder = "Enter text...";
  textarea.value =
    "Hello, this is a streaming text-to-speech test. Speech should begin before the full sentence finishes generating.";
  container.append(textLabel, textarea);

  const comparisonHeader = el("div", "comparison-header");
  const comparisonTitle = el("h2");
  comparisonTitle.textContent = "Candidate generation";
  const capabilityBadge = el("span", "capability-badge");
  capabilityBadge.textContent = "Pitch + rate now waveform processed";
  comparisonHeader.append(comparisonTitle, capabilityBadge);
  container.appendChild(comparisonHeader);

  const mutationBar = el("div", "mutation-bar");
  const mutation = rangeRow("Mutation strength", 0.02, 0.75, 0.01, 0.25);
  const randomButton = el("button", "secondary");
  randomButton.textContent = "New random pair";
  const mutateAButton = el("button", "secondary");
  mutateAButton.textContent = "Mutate from A";
  const mutateBButton = el("button", "secondary");
  mutateBButton.textContent = "Mutate from B";
  const mutationActions = el("div", "mutation-actions");
  mutationActions.append(randomButton, mutateAButton, mutateBButton);
  mutationBar.append(mutation.wrap, mutationActions);
  container.appendChild(mutationBar);

  let activeSlot: ComparisonSlot | null = null;
  let streaming = false;
  let slotA: ComparisonSlot;
  let slotB: ComparisonSlot;

  const connection = el("span", "connection");
  connection.innerHTML =
    'Backend: <strong class="disconnected">connecting…</strong>';
  const statusSpan = connection.querySelector("strong")!;
  const stopButton = el("button", "secondary");
  stopButton.textContent = "Stop playback";
  stopButton.disabled = true;

  const client = new TTSWebSocketClient(
    WS_URL,
    (status) => {
      streaming = status === "Streaming";
      statusSpan.textContent = status;
      statusSpan.className =
        status === "Connected" || status === "Ready" || status === "Streaming"
          ? "connected"
          : status === "Disconnected"
            ? "disconnected"
            : "";
      stopButton.disabled = !streaming;
      slotA.generateButton.disabled = streaming;
      slotB.generateButton.disabled = streaming;
    },
    (metrics) => {
      if (!activeSlot) return;
      const ttfa =
        (metrics.client_ttfa_ms as number | undefined) ??
        (metrics.ttfa_server_ms as number | undefined);
      const rtf = metrics.rtf as number | undefined;
      activeSlot.metrics.textContent = [
        `TTFA ${ttfa != null ? `${Math.round(ttfa)} ms` : "—"}`,
        `RTF ${rtf != null ? rtf.toFixed(3) : "—"}`,
        metrics.total_audio_seconds != null
          ? `${Number(metrics.total_audio_seconds).toFixed(1)}s audio`
          : "",
      ]
        .filter(Boolean)
        .join(" · ");
      slotA.generateButton.disabled = false;
      slotB.generateButton.disabled = false;
      stopButton.disabled = true;
    },
    (version) => {
      if (activeSlot) {
        activeSlot.metrics.textContent =
          `Backend voice update accepted · version ${version}`;
      }
    },
  );

  const queueLiveVoiceUpdate = (slot: ComparisonSlot) => {
    if (activeSlot !== slot) return;
    client.updateOutput({
      energy: slot.genome.energy,
      warmth: slot.genome.warmth,
      brightness: slot.genome.brightness,
      presence: slot.genome.presence,
      speakingRate: slot.genome.speaking_rate,
      pitchSemitones: slot.genome.pitch_semitones,
    });
    slot.metrics.textContent = streaming
      ? "Waveform transformation changing smoothly…"
      : "Identity updated · ready to audition";
  };

  const generateSlot = async (slot: ComparisonSlot) => {
    const text = textarea.value.trim();
    if (!text) {
      textarea.focus();
      return;
    }
    activeSlot = slot;
    slot.metrics.textContent = "Generating…";
    try {
      await client.generate(text, slot.settings, {
        energy: slot.genome.energy,
        warmth: slot.genome.warmth,
        brightness: slot.genome.brightness,
        presence: slot.genome.presence,
        speakingRate: slot.genome.speaking_rate,
        pitchSemitones: slot.genome.pitch_semitones,
      });
    } catch (err) {
      slot.metrics.textContent = `Error: ${(err as Error).message}`;
    }
  };

  const savedGenomeRaw = localStorage.getItem("neural-tts.saved-genome");
  let initialA = FALLBACK_GENOMES[0];
  if (savedGenomeRaw) {
    try {
      initialA = {
        ...FALLBACK_GENOMES[0],
        ...(JSON.parse(savedGenomeRaw) as Partial<VoiceGenome>),
      };
    } catch {
      localStorage.removeItem("neural-tts.saved-genome");
    }
  }

  slotA = buildSlot(
    "A",
    initialA,
    () => void generateSlot(slotA),
    () => {
      localStorage.setItem("neural-tts.saved-genome", JSON.stringify(slotA.genome));
      slotA.metrics.textContent = "Saved as your current voice identity";
    },
    queueLiveVoiceUpdate,
  );
  slotB = buildSlot(
    "B",
    FALLBACK_GENOMES[1],
    () => void generateSlot(slotB),
    () => {
      localStorage.setItem("neural-tts.saved-genome", JSON.stringify(slotB.genome));
      slotB.metrics.textContent = "Saved as your current voice identity";
    },
    queueLiveVoiceUpdate,
  );
  const compareGrid = el("div", "compare-grid");
  compareGrid.append(slotA.card, slotB.card);
  container.appendChild(compareGrid);

  const replaceCandidates = async (parent?: VoiceGenome) => {
    mutationActions.querySelectorAll("button").forEach((button) => {
      button.disabled = true;
    });
    try {
      const candidates = await generateVoiceCandidates(
        Date.now(),
        Number(mutation.input.value),
        parent,
      );
      slotA.applyGenome(candidates[0]);
      slotB.applyGenome(candidates[1]);
    } catch (error) {
      statusSpan.textContent = `candidate error: ${(error as Error).message}`;
      statusSpan.className = "disconnected";
    } finally {
      mutationActions.querySelectorAll("button").forEach((button) => {
        button.disabled = false;
      });
    }
  };
  randomButton.addEventListener("click", () => void replaceCandidates());
  mutateAButton.addEventListener("click", () =>
    void replaceCandidates(slotA.genome),
  );
  mutateBButton.addEventListener("click", () =>
    void replaceCandidates(slotB.genome),
  );

  const transport = el("div", "transport");
  transport.append(connection, stopButton);
  container.appendChild(transport);

  const note = el("div", "note");
  note.textContent =
    "Pitch and rate are processed continuously in the browser using buffered resampling and granular pitch compensation. Extreme values may reveal prototype artifacts.";
  container.appendChild(note);
  root.appendChild(container);

  stopButton.addEventListener("click", () => client.stop());
  void client.connect().catch(() => {
    statusSpan.textContent = "offline — start server on :8000";
    statusSpan.className = "disconnected";
  });
}
