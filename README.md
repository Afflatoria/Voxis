# Neural TTS

Local, real-time, highly configurable neural text-to-speech — **Milestone 1 baseline**.

This repository establishes a streaming TTS foundation using **F5-TTS** behind a model-independent `StreamingTTS` abstraction. It is intentionally **not** a custom trained TTS system yet; it validates streaming infrastructure, voice configuration APIs, latency measurement, and browser playback before later milestones add disentangled voice control and custom models.

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
           └── F5TTSBackend  (M1 — swappable later)
                   │
                   └── F5-TTS v1 Base (local GPU, HF cache)
```

## Requirements

| Component | Version / notes |
|-----------|-----------------|
| Python | 3.10 – 3.12 |
| PyTorch | 2.11+ with CUDA wheels (tested with cu126) |
| NVIDIA GPU | Recommended (8 GB+ VRAM) |
| Node.js | 18+ (frontend dev server) |
| OS | Windows / Linux |

Tested dev GPU: **NVIDIA RTX 2000 Ada (8 GB)**.

> **Windows:** F5-TTS depends on `torchcodec`, which needs FFmpeg **shared** DLLs. See [SETUP.md](SETUP.md).

## Installation

### 1. Clone and create virtual environment

```powershell
cd voxis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,f5tts]"
```

> **Windows note:** If you have multiple Python installs, always use `python -m pip` from the same interpreter that runs the server.

### 2. Prefetch model weights

```powershell
python -m pip install -e ".[download]"
python scripts/download_models.py
```

This downloads the F5-TTS checkpoint and Vocos vocoder into your Hugging Face cache.

### 3. Configure reference voice

```powershell
copy .env.example .env
```

Copy a short reference WAV (~12s, with ~1s trailing silence) to `assets/voices/reference.wav` and set the matching transcript:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURAL_TTS_BACKEND` | `f5tts` | `f5tts` or `mock` |
| `F5TTS_MODEL` | `F5TTS_v1_Base` | Model variant |
| `F5TTS_REF_WAV` | `assets/voices/reference.wav` | Reference clip for voice cloning |
| `F5TTS_REF_TEXT` | (empty) | Transcript of reference WAV; blank = Whisper ASR |
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
cd voxis
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

### CLI baseline (no frontend)

```powershell
python baseline/cli_speak.py --text "Hello from F5-TTS."
```

### Upstream F5-TTS Gradio UI

```powershell
f5-tts_infer-gradio
```

## Usage

1. Enter text in the textarea.
2. Audition Voice A and Voice B using the same script.
3. Adjust **Speaking rate**, **Pitch**, **Energy**, **Warmth**, **Brightness**,
   and **Presence** continuously while audio plays.
4. Choose a candidate and click **Mutate from A/B** to generate two nearby,
   reproducible alternatives.
5. Save the preferred identity in browser storage and continue refining it.

Phase 1 shapes the configured `F5TTS_REF_WAV` with a real-time browser output
chain. These genomes are reproducible tonal profiles; independent learned
speaker embeddings are planned for a later research phase.

## WebSocket protocol (`WS /v1/stream`)

### Client → server (JSON text frames)

**Start synthesis**

```json
{
  "type": "start",
  "text": "Hello, this is a streaming TTS test.",
  "voice": {
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

**Voice update (applied at the next short phrase boundary)**

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

F5-TTS (requires model prefetch + reference WAV):

```powershell
$env:NEURAL_TTS_BACKEND = "f5tts"
python scripts/benchmark.py --backend f5tts
```

## Testing

```powershell
$env:NEURAL_TTS_BACKEND = "mock"
$env:PYTHONPATH = "src"
pytest tests/ -v
```

Tests use the **mock backend** by default — no GPU model required.

## Voice controls

| Control | M1 status |
|---------|-----------|
| `speaking_rate` | Real-time waveform resampling with inverse pitch compensation in the browser |
| `energy` | Real-time browser output gain with a smooth 80 ms transition |
| `pitch` | Real-time granular waveform pitch shift in the browser prototype |
| `warmth`, `brightness`, `presence` | Smooth browser EQ filters |
| `speaker` | Uses `F5TTS_REF_WAV` reference voice |
| `language`, `accent`, `breathiness`, `roughness`, `expressiveness` | **Schema only — not applied** |

The UI/API expose future controls without faking backend support.

## Known limitations (M1)

- **Single GPU job at a time** — concurrent sessions serialize via a lock.
- **Boundary-based mid-stream changes** — updates affect the next short text segment, not audio already playing.
- **Continuous energy changes** — streamed PCM passes through a browser output-processing layer and gain changes are smoothly ramped without regeneration.
- **Prototype pitch/rate DSP** — granular pitch compensation enables continuous changes, but extreme values can introduce modulation artifacts.
- **No crossfade yet** — adjacent segments can have an audible transition when settings differ significantly.
- **Chunk streaming, not token streaming** — F5-TTS synthesizes short text segments and yields fixed-size waveform chunks.
- **8 GB GPU** may require closing other GPU apps.
- **First inference** includes model load/warmup latency.

## What M1 does NOT solve

- Custom TTS training / disentangled embeddings
- Independent acoustic factor control
- Mid-playback seamless voice transitions
- Production deployment, auth, multi-GPU serving

## Roadmap

| Milestone | Focus |
|-----------|-------|
| **M1** (this) | Streaming infra + F5-TTS baseline + metrics |
| **M1.5** | Voice playground — A/B presets, reference library |
| **M2** | Session chunk cache + apply `voice_update` to unplayed audio |
| **M3** | Dataset pipeline + labeled metadata |
| **M4** | Learned voice encoder + adapter fine-tuning |

## Project layout

```text
src/neural_tts/          Python package
  api/                   HTTP + WebSocket
  models/                StreamingTTS + F5TTSBackend + MockBackend
  streaming/             Session + scheduler + state
  voice/                 VoiceConfig schema
frontend/                Vite + TypeScript UI
baseline/                CLI tester (no frontend)
scripts/                 download_models.py, benchmark.py
tests/                   Unit + WebSocket integration tests
configs/                 Model/inference YAML (reference)
assets/voices/           Reference WAV clips for cloning
```

## License

Project scaffolding: MIT. F5-TTS model/code: see [upstream license](https://github.com/SWivid/F5-TTS).
