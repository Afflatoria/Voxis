import { TTSWebSocketClient } from "./wsClient";

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

function rangeRow(label: string, min: number, max: number, step: number, value: number) {
  const wrap = el("div");
  const lbl = el("label");
  const valueSpan = el("span", "muted");
  valueSpan.textContent = value.toFixed(2);
  lbl.textContent = label + " ";
  lbl.appendChild(valueSpan);

  const input = document.createElement("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.addEventListener("input", () => {
    valueSpan.textContent = Number(input.value).toFixed(2);
  });

  wrap.appendChild(lbl);
  wrap.appendChild(input);
  return { wrap, input };
}

export function mountApp(root: HTMLElement): void {
  root.innerHTML = "";
  const container = el("div", "container");
  const title = el("h1");
  title.textContent = "Neural TTS";
  container.appendChild(title);

  const textarea = el("textarea");
  textarea.placeholder = "Enter text...";
  textarea.value =
    "Hello, this is a streaming text-to-speech test. Speech should begin before the full sentence finishes generating.";
  container.appendChild(textarea);

  const controls = el("div", "controls");
  const speakerLabel = el("label");
  speakerLabel.textContent = "Voice";
  const speakerSelect = el("select") as HTMLSelectElement;
  for (const opt of ["default"]) {
    const o = el("option") as HTMLOptionElement;
    o.value = opt;
    o.textContent = opt;
    speakerSelect.appendChild(o);
  }
  controls.appendChild(speakerLabel);
  controls.appendChild(speakerSelect);

  const rate = rangeRow("Speaking rate", 0.5, 2.0, 0.05, 1.0);
  const pitch = rangeRow("Pitch (future — not applied in M1)", -1, 1, 0.05, 0);
  const energy = rangeRow("Energy", 0.5, 2.0, 0.05, 1.0);
  controls.append(rate.wrap, pitch.wrap, energy.wrap);
  container.appendChild(controls);

  const row = el("div", "row");
  const generateBtn = el("button");
  generateBtn.textContent = "Generate";
  const stopBtn = el("button", "secondary");
  stopBtn.textContent = "Stop";
  stopBtn.disabled = true;
  row.append(generateBtn, stopBtn);
  container.appendChild(row);

  const statusBox = el("div", "status");
  const statusLine = el("div");
  statusLine.innerHTML = 'Status: <span class="disconnected">Idle</span>';
  const statusSpan = statusLine.querySelector("span")!;
  const metricsBox = el("div", "metrics");
  metricsBox.textContent = "TTFA: —\nRTF: —";
  const note = el("div", "note");
  note.textContent =
    "Pitch/warmth/breathiness sliders are reserved for future models. Speaking rate and energy map to F5-TTS speed/loudness in M1.";
  statusBox.append(statusLine, metricsBox, note);
  container.appendChild(statusBox);

  root.appendChild(container);

  const client = new TTSWebSocketClient(
    WS_URL,
    (status) => {
      statusSpan.textContent = status;
      statusSpan.className =
        status === "Connected" || status === "Ready" || status === "Streaming"
          ? "connected"
          : status === "Disconnected"
            ? "disconnected"
            : "";
      stopBtn.disabled = !["Streaming"].includes(status);
      generateBtn.disabled = status === "Streaming";
    },
    (metrics) => {
      const ttfa =
        (metrics.client_ttfa_ms as number | undefined) ??
        (metrics.ttfa_server_ms as number | undefined);
      const rtf = metrics.rtf as number | undefined;
      metricsBox.textContent = [
        `TTFA: ${ttfa != null ? `${Math.round(ttfa)} ms` : "—"}`,
        `RTF: ${rtf != null ? rtf.toFixed(3) : "—"}`,
        `Chunks: ${metrics.chunk_count ?? "—"}`,
        `Audio: ${metrics.total_audio_seconds != null ? Number(metrics.total_audio_seconds).toFixed(2) + "s" : "—"}`,
      ].join("\n");
      generateBtn.disabled = false;
      stopBtn.disabled = true;
    },
  );

  generateBtn.addEventListener("click", async () => {
    const text = textarea.value.trim();
    if (!text) return;
    metricsBox.textContent = "TTFA: measuring...\nRTF: —";
    try {
      await client.generate(text, {
        language: "English",
        speaker: speakerSelect.value,
        speaking_rate: Number(rate.input.value),
        pitch: Number(pitch.input.value),
        energy: Number(energy.input.value),
      });
    } catch (err) {
      statusSpan.textContent = `Error: ${(err as Error).message}`;
    }
  });

  stopBtn.addEventListener("click", () => client.stop());

  void client.connect().catch(() => {
    statusSpan.textContent = "Backend offline — start server on :8000";
    statusSpan.className = "disconnected";
  });
}
