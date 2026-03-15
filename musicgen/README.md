# musicgen — AI Music Generation with ACE-Step

This folder contains everything needed to reproduce the AI music generation pipeline used in BioBeats. It uses **ACE-Step 1.5** — a local, GPU-accelerated music generation model that runs entirely on your machine. No cloud API needed.

The pipeline supports:
- **Text-to-music**: generate a clip from a style description
- **Lyrics-to-music**: add your own lyrics on top
- **Audio-to-audio (cover/remix mode)**: use an existing audio file as a structural reference and transform it

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 8 GB VRAM | RTX 4060+ (8 GB+) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 40 GB free |
| OS | Linux x86_64 (Ubuntu/Pop!_OS) | Pop!_OS 22.04 |

> **Note:** macOS (Apple Silicon, CPU/MPS) is supported but significantly slower. Windows is supported via the same steps with minor path adjustments.

---

## Prerequisites

### 1. Install `uv` (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your terminal
uv --version     # should print uv 0.x.x
```

### 2. Install `jq` (JSON CLI tool — required by the generate script)

```bash
sudo apt install jq    # Ubuntu/Debian/Pop!_OS
# or: brew install jq  # macOS
```

### 3. Install NVIDIA drivers + CUDA (Linux)

ACE-Step requires CUDA 12.8+. If you're on Pop!_OS or Ubuntu with an NVIDIA GPU:

```bash
# Check if CUDA is already available
nvidia-smi

# If not, install via apt (Pop!_OS ships NVIDIA drivers in the ISO):
sudo apt install nvidia-cuda-toolkit
```

---

## Setup

### Step 1 — Clone this repo with submodules

```bash
git clone --recurse-submodules https://github.com/MattJmt/BioBeats.git
cd BioBeats
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This will clone **ACE-Step 1.5** into `musicgen/ACE-Step-1.5/`.

---

### Step 2 — Install ACE-Step dependencies

```bash
cd musicgen/ACE-Step-1.5
uv sync
```

`uv sync` reads `pyproject.toml` and installs all dependencies into a local `.venv`. This includes PyTorch with CUDA 12.8 and all model dependencies. **First run takes several minutes** (downloading ~2 GB of packages).

---

### Step 3 — Download model checkpoints

ACE-Step downloads its model weights automatically on first startup from HuggingFace. This happens when you first start the API server (Step 4). The models are cached to `~/.cache/huggingface/` and are roughly **10–15 GB**.

Alternatively, pre-download manually:

```bash
cd musicgen/ACE-Step-1.5
uv run python -c "from acestep.pipeline_ace_step import ACEStepPipeline; ACEStepPipeline.from_pretrained('ACE-Step/ACE-Step-v1-3.5B-instruct')"
```

---

### Step 4 — Configure the generate script

The generate CLI (`acestep.sh`) needs to know the API URL. Run this once:

```bash
bash musicgen/ACE-Step-1.5/.claude/skills/acestep/scripts/acestep.sh config --set api_url http://127.0.0.1:8001
```

---

### Step 5 — Start the API server

#### Option A: Run manually (for testing)

```bash
cd musicgen/ACE-Step-1.5
uv run acestep-api --host 127.0.0.1 --port 8001
```

Wait until you see: `Application startup complete.` (takes 1–10 min on first boot while loading the model into VRAM).

Verify it's healthy:

```bash
curl http://127.0.0.1:8001/health
```

#### Option B: Run as a systemd service (recommended — auto-starts at login)

```bash
# Copy the unit file
mkdir -p ~/.config/systemd/user
cp musicgen/systemd/acestep-api.service ~/.config/systemd/user/

# Edit the path to point to your ACE-Step clone location
# Replace the ExecStart path if you cloned BioBeats somewhere other than ~
nano ~/.config/systemd/user/acestep-api.service
# Change: --project %h/ACE-Step-1.5
# To:     --project /full/path/to/BioBeats/musicgen/ACE-Step-1.5

# Enable and start
systemctl --user daemon-reload
systemctl --user enable acestep-api
systemctl --user start acestep-api

# Check status
systemctl --user status acestep-api
journalctl --user -u acestep-api -f   # live logs
```

The service will auto-restart on crash and start automatically when you log in.

---

## Generating Music

All generation goes through `acestep.sh`, the CLI wrapper around the ACE-Step API.

```bash
ACESTEP="musicgen/ACE-Step-1.5/.claude/skills/acestep/scripts/acestep.sh"
```

### Text-to-music (no lyrics)

```bash
bash "$ACESTEP" generate \
  -c "lo-fi hip hop, chill piano, 85 BPM" \
  --duration 30
```

### With lyrics

```bash
bash "$ACESTEP" generate \
  -c "indie folk, acoustic guitar, breathy vocals, 90 BPM" \
  -l "[Verse 1]
Your lyrics here
Second line

[Chorus]
Chorus text here" \
  --duration 30
```

### Audio-to-audio (cover/remix mode)

Provide an existing audio file as a structural reference. ACE-Step uses it to guide the generation's style and structure while applying your new caption and lyrics on top.

```bash
bash "$ACESTEP" generate \
  -c "dark synthwave, pulsing bass, heavy reverb" \
  -l "[Verse 1]
Your lyrics here" \
  --src-audio "/path/to/reference.mp3" \
  --duration 30
```

> The more musically rich the reference audio, the more structure ACE-Step has to work with. A pure sine wave gives it almost nothing; a full produced track gives it much more.

### Output

Generated MP3s are saved to `musicgen/ACE-Step-1.5/acestep_output/` as `<job_id>_1.mp3`.

---

## Waiting for the server to be ready

Use the included helper script to block until the server is healthy (useful in automation):

```bash
bash musicgen/scripts/ensure-server.sh
```

- Exits immediately if the server is already healthy
- Polls every 2 seconds, up to 600 seconds
- Prints a timeout error if the server never becomes ready

---

## VRAM Tips (RTX 4060 / 8 GB)

The RTX 4060 Laptop GPU has 7.62 GiB usable VRAM. ACE-Step fits comfortably, but to avoid OOM errors from memory fragmentation:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

This is already set in the systemd unit file. If you're running the server manually, export it before starting.

If you still get OOM on long or complex generations, add `--no-thinking --guidance 4.5` flags:

```bash
bash "$ACESTEP" generate \
  -c "..." \
  --no-thinking --guidance 4.5 \
  --duration 30
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `curl: connection refused` on health check | Server not started yet — run `systemctl --user start acestep-api` or start manually |
| Model download stuck / slow | HuggingFace is rate-limiting. Wait or set `HF_HUB_OFFLINE=1` after first download |
| `HTTP 401` from `acestep.sh` | Re-run: `bash acestep.sh config --set api_url http://127.0.0.1:8001` |
| OOM / CUDA out of memory | Add `--no-thinking --guidance 4.5`; or restart the server (`systemctl --user restart acestep-api`) |
| No MP3 in output dir | Check stderr from `acestep.sh` — generation may have failed silently |
| `jq: command not found` | `sudo apt install jq` |
| `uv: command not found` | Re-run the `uv` install curl command above and restart terminal |
| Submodule folder is empty | Run `git submodule update --init --recursive` |

---

## File Reference

```
musicgen/
├── README.md                          ← this file
├── scripts/
│   └── ensure-server.sh               ← blocks until ACE-Step API is healthy
├── systemd/
│   └── acestep-api.service            ← systemd unit for auto-start at login
└── ACE-Step-1.5/                      ← git submodule (https://github.com/ACE-Step/ACE-Step-1.5)
    ├── .claude/skills/acestep/scripts/
    │   └── acestep.sh                 ← main generation CLI (bash + curl + jq)
    ├── pyproject.toml                 ← Python deps (managed by uv)
    └── acestep_output/                ← generated MP3s land here
```
