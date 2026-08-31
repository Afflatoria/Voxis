# Setup Guide (Windows)

Project-specific setup notes that are easy to miss from a plain `pip install`.

## F5-TTS backend

This project uses **F5-TTS** as its default inference backend. Install inference dependencies with:

```powershell
python -m pip install -e ".[f5tts]"
```

See the main [README.md](README.md) for reference voice setup (`F5TTS_REF_WAV`, `F5TTS_REF_TEXT`).

## torchcodec + FFmpeg (required on Windows)

`torchcodec` is used by recent `torchaudio` stacks (including F5-TTS). On Windows it fails at import unless FFmpeg **shared** DLLs are available:

```
RuntimeError: Could not load libtorchcodec
FFmpeg version 7: Could not load library: ...\libtorchcodec_core7.dll
```

### Why this happens

1. **Version pairing** — `torchcodec` must match your PyTorch version. This project currently uses:
   - `torch` 2.11.x + `torchcodec` 0.12.x (see [compatibility table](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec))
2. **Shared FFmpeg DLLs** — On Windows, Python 3.8+ does not search `PATH` for DLL dependencies. `torchcodec` loads `libtorchcodec_core*.dll`, which in turn needs `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, etc.
3. **Static builds do not work** — `winget install BtbN.FFmpeg.GPL` (and most “ffmpeg.exe only” installs) ship a **static** build with no separate DLLs. That is fine for running `ffmpeg` from the shell, but not for `torchcodec`.

### One-time FFmpeg install

Download the **gpl-shared** Windows build from [BtbN FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases/latest) (look for `ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip` or the current `*-win64-gpl-shared-*.zip`).

Extract so the `bin` folder ends up at:

```text
tools/ffmpeg-shared/ffmpeg-n8.1-latest-win64-gpl-shared-8.1/bin/
  avcodec-62.dll
  avformat-62.dll
  avutil-60.dll
  ffmpeg.exe
  ...
```

PowerShell example:

```powershell
$dest = "tools\ffmpeg-shared"
$zip  = "$env:TEMP\ffmpeg-shared.zip"
$url  = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $dest -Force
```

### Venv hooks (already configured in this repo)

After FFmpeg is extracted, two small hooks make `import torchcodec` work automatically:

1. **`.venv/Lib/site-packages/sitecustomize.py`** — calls `os.add_dll_directory()` pointing at the FFmpeg `bin` folder. This runs whenever you use this venv’s Python, even if you call `.venv\Scripts\python.exe` directly (no activation required).

2. **`.venv/Scripts/activate.bat`** — prepends the FFmpeg `bin` folder to `PATH` when you activate the venv. Helps CLI tools and torchcodec’s `which("ffmpeg")` fallback.

If you **recreate the venv**, re-run the FFmpeg download above and recreate `sitecustomize.py`:

```python
# .venv/Lib/site-packages/sitecustomize.py
"""Ensure FFmpeg shared DLLs are discoverable for torchcodec on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FFMPEG_BIN = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "ffmpeg-shared"
    / "ffmpeg-n8.1-latest-win64-gpl-shared-8.1"
    / "bin"
)

if sys.platform == "win32" and _FFMPEG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_FFMPEG_BIN))
```

Also add these two lines to `activate.bat` **before** `set PATH=%VIRTUAL_ENV%\Scripts;%PATH%`:

```bat
set "FFMPEG_BIN=%VIRTUAL_ENV%\..\tools\ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" set "PATH=%FFMPEG_BIN%;%PATH%"
```

### Verify

```cmd
.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); import torchcodec; print('torchcodec:', torchcodec.__version__)"
```

Expected output (versions may differ slightly):

```text
torch: 2.11.0+cu126
torchcodec: 0.12.0+cpu
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not load libtorchcodec` | Install the **shared** FFmpeg build (not static/winget-only). Confirm `avcodec-*.dll` exists under `tools/ffmpeg-shared/.../bin/`. |
| Works after `activate`, fails with bare `python` | Recreate `sitecustomize.py` in the venv (see above). |
| `torchcodec` version error after PyTorch upgrade | Re-check the [torchcodec ↔ torch matrix](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec) and `pip install` a matching `torchcodec`. |
| Download URL 404 | Asset names change between releases — grab the latest `*-win64-gpl-shared-*.zip` from the [releases page](https://github.com/BtbN/FFmpeg-Builds/releases/latest) and update the folder name in `sitecustomize.py` / `activate.bat` if needed. |
