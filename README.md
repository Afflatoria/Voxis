# Neural TTS

Local, real-time, highly configurable neural text-to-speech — **Milestone 1 baseline**.

This repository establishes a streaming TTS foundation using **Fun-CosyVoice 3** behind a model-independent `StreamingTTS` abstraction. It is intentionally **not** a custom trained TTS system yet; it validates streaming infrastructure, voice configuration APIs, latency measurement, and browser playback before later milestones add disentangled voice control and custom models.

## Architecture

```text
Browser (Web Audio API)
   │
   │  WebSocket  /v1/stream
   │    JSON control frames + binary PCM audio frames
   ▼
FastAPI backend (neural_tts)
   │
   ├── StreamingSession (cancellation, metrics, voice state)
   │
   └── StreamingTTS abstraction
           │
           └── CosyVoiceBackend  (M1 — swappable later)
                   │
                   └── Fun-CosyVoice3-0.5B (local GPU)
```

## Requirements

| Component | Version / notes |
|-----------|-----------------|
| Python | 3.10 – 3.12 (3.10 recommended by CosyVoice upstream) |
| PyTorch | 2.3+ with CUDA 12.1 wheels |
| NVIDIA GPU | Recommended (8 GB+ VRAM for CosyVoice3 0.5B) |
| Node.js | 18+ (frontend dev server) |
| Git | For cloning CosyVoice + submodules |
| OS | Windows / Linux (CosyVoice upstream is Linux-oriented; Windows works with onnxruntime CPU/GPU builds) |

Tested dev GPU: **NVIDIA RTX 2000 Ada (8 GB)**.

## Installation

### 1. Clone and create virtual environment

```powershell
cd voice-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

> **Windows note:** If you have multiple Python installs, always use `python -m pip` from the same interpreter that runs the server.

Optional — CosyVoice inference deps on **Windows** (do not use upstream requirements.txt):

```powershell
python scripts/install_cosyvoice_windows.py
```

On Linux, you can use upstream deps instead:

```powershell
python -m pip install -r third_party/CosyVoice/requirements.txt
```

> **Note:** CosyVoice officially recommends Python 3.10 + conda. If you hit dependency issues on 3.12, create a 3.10 venv for the backend.

### 2. Download model + CosyVoice repo

Install download dependencies first:

```powershell
python -m pip install -e ".[download]"
```

Then download:

```powershell
python scripts/download_models.py
```

This will:
- clone `third_party/CosyVoice` (recursive, includes reference WAV assets)
- download `Fun-CosyVoice3-0.5B-2512` into `pretrained_models/Fun-CosyVoice3-0.5B`

### 3. Configure environment

```powershell
copy .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURAL_TTS_BACKEND` | `cosyvoice` | `cosyvoice` or `mock` |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | Local model path |
| `NEURAL_TTS_PORT` | `8000` | API port |

Use `NEURAL_TTS_BACKEND=mock` to run without GPU/model.

### 4. Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Running

**Terminal 1 — backend:**

```powershell
cd voice-research
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m neural_tts.main
```

Or:

```powershell
neural-tts
```

**Terminal 2 — frontend:**

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173

## Usage

1. Enter text in the textarea.
2. Adjust **Speaking rate** / **Energy** (supported via CosyVoice instruct + speed).
3. Click **Generate** — audio should begin before synthesis completes.
4. Click **Stop** to cancel.
5. Observe **TTFA** and **RTF** metrics.

## WebSocket protocol (`WS /v1/stream`)

### Client → server (JSON text frames)

**Start synthesis**

```json
{
  "type": "start",
  "text": "Hello, this is a streaming TTS test.",
  "voice": {
    "language": "English",
    "speaker": "default",
    "speaking_rate": 1.0,
    "energy": 1.0
  },
  "request_id": "optional-uuid"
}
```

**Stop**

```json
{ "type": "stop" }
```

**Voice update (stored for session; mid-stream rerender in M2)**

```json
{
  "type": "voice_update",
  "voice": { "speaking_rate": 1.2 }
}
```

### Server → client

**JSON events:** `ready`, `started`, `complete`, `cancelled`, `error`, `voice_updated`, `pong`

**Binary audio frames:** header + PCM s16le

```text
magic:   4 bytes  "NTTS"
version: 1 byte   0x01
reserved:3 bytes
sequence:uint32 LE
sample_rate: uint32 LE
payload: PCM int16 LE samples
```

## Benchmarking

Mock backend (no GPU):

```powershell
$env:NEURAL_TTS_BACKEND = "mock"
python scripts/benchmark.py
```

CosyVoice (requires model download):

```powershell
$env:NEURAL_TTS_BACKEND = "cosyvoice"
python scripts/benchmark.py --backend cosyvoice
```

## Testing

```powershell
$env:NEURAL_TTS_BACKEND = "mock"
$env:PYTHONPATH = "src"
pytest tests/ -v
```

Tests use the **mock backend** by default — no multi-GB model download required.

## Voice controls in M1

| Control | M1 status |
|---------|-----------|
| `language` | Mapped to CosyVoice instruct |
| `accent`, `accent_strength` | Mapped to instruct |
| `speaking_rate` | CosyVoice `speed` parameter |
| `energy` | Approximated via instruct |
| `speaker` | Uses bundled reference prompt WAV |
| `pitch`, `warmth`, `breathiness`, `roughness`, `expressiveness` | **Schema only — not applied** |

The UI/API expose future controls without faking backend support.

## Known limitations (M1)

- **Single GPU job at a time** — concurrent sessions serialize via a lock.
- **No seamless mid-stream voice morphing** — `voice_update` stores state for future chunks/sessions.
- **CosyVoice instruct control** is prompt-based, not independent sliders.
- **8 GB GPU** may require closing other GPU apps.
- **First inference** includes model load/warmup latency.
- Windows may need manual CosyVoice dependency troubleshooting.

## What M1 does NOT solve

- Custom TTS training / disentangled embeddings
- Independent acoustic factor control
- Mid-playback seamless voice transitions
- Production deployment, auth, multi-GPU serving

## Roadmap

| Milestone | Focus |
|-----------|-------|
| **M1** (this) | Streaming infra + CosyVoice baseline + metrics |
| **M1.5** | Voice playground — A/B presets, accent instruct matrix, reference library |
| **M2** | Session chunk cache + apply `voice_update` to unplayed audio |
| **M3** | Dataset pipeline + labeled metadata |
| **M4** | Learned voice encoder + adapter fine-tuning |

## Project layout

```text
src/neural_tts/          Python package
  api/                   HTTP + WebSocket
  models/                StreamingTTS + CosyVoiceBackend + MockBackend
  streaming/             Session + scheduler + state
  voice/                 VoiceConfig schema
frontend/                Vite + TypeScript UI
scripts/                 download_models.py, benchmark.py
tests/                   Unit + WebSocket integration tests
configs/                 Model/inference YAML
third_party/CosyVoice/   Upstream TTS (cloned, not modified)
pretrained_models/       Downloaded weights (gitignored)
```

## License

Project scaffolding: MIT. CosyVoice model/code: Apache 2.0 (see upstream).
