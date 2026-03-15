# BioBeats 🫀🎧

**SonoCorpus x MIT HARDMODE** — 48h hackathon project building a multimodal wearable that lets you play music with your body.

Biosignals (IMU, PPG, EMG, piezo) are captured from a glove and wrist sensors, fed into ML models, and used to trigger and shape AI-generated music in real time.

---

## Repository Structure

```
BioBeats/
├── src/                  ← Arduino firmware (PlatformIO — SparkFun Pro Micro)
├── imu/                  ← IMU streaming (LSM6DSO32 via Qwiic Mux → Python)
├── ppg/                  ← PPG sensor capture and live viewer
├── tools/                ← ML training + live inference scripts
│   ├── train_hit_classifier.py   ← Train finger-hit model from CSV data
│   └── live_hit_inference.py     ← Run trained model against live serial stream
├── models/               ← Trained PyTorch model checkpoints (.pt)
├── data/                 ← Captured CSV datasets (finger hit recordings)
├── musicgen/             ← AI music generation pipeline (ACE-Step)
├── requirements.txt      ← Python dependencies for glove/sensor tools
└── platformio.ini        ← Firmware build config
```

---

## Modalities

Each sensor modality lives in its own folder. The intended structure for signal sources:

```
imu/    ← accelerometer/gyro (LSM6DSO32 × 4 via Qwiic Mux)
ppg/    ← photoplethysmography (heart rate / blood flow)
emg/    ← electromyography (muscle activation) — coming soon
bone/   ← bone conduction — coming soon
```

Each modality folder contains its own `readme.md` with hardware wiring, startup sequence, and script usage.

---

## Glove Piezo Capture (root-level tools)

Captures and plots live piezo data from a 4-finger glove (Index, Middle, Ring, Thumb) over serial.

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Run the live plotter

```bash
python3 tools/serial_plotter.py
```

- Auto-detects Arduino/Pro Micro serial port
- Opens a live 4-channel plot
- Writes CSV to `data/glove_capture_YYYYMMDD_HHMMSS.csv`
- Auto-reconnects if the device briefly disconnects

**Useful options:**

```bash
python3 tools/serial_plotter.py --port /dev/tty.usbmodemXXXX  # specify port
python3 tools/serial_plotter.py --port COM3                    # Windows
python3 tools/serial_plotter.py --no-plot                      # CSV only
python3 tools/serial_plotter.py --duration 30                  # record 30s
python3 tools/serial_plotter.py --window-sec 15                # plot window
```

### Train the finger-hit classifier

```bash
python3 tools/train_hit_classifier.py
# → saves model to models/finger_hit_model.pt
```

### Run live inference

```bash
python3 tools/live_hit_inference.py
# → loads models/finger_hit_model.pt, detects hits in real time
```

---

## Arduino Firmware

Built with [PlatformIO](https://platformio.org/). Target board: **SparkFun Pro Micro 16MHz**.

```bash
# Install PlatformIO CLI or use the VSCode extension
pio run             # compile
pio run --target upload   # flash to board
```

Serial format expected by Python tools:
- Optional header: `Index\tMiddle\tRing\tThumb`
- Data rows: 4 tab- or comma-separated numeric values
- Baud rate: `115200`

---

## IMU Streaming

**→ See [`imu/readme.md`](imu/readme.md)** for full hardware wiring and startup sequence.

4x LSM6DSO32 IMUs on a SparkFun Qwiic Mux, streamed from an Arduino UNO Q through App Lab → TCP → Python live plot.

---

## PPG Streaming

**→ See [`ppg/`](ppg/)** for sensor wiring and scripts.

---

## AI Music Generation

**→ See [`musicgen/README.md`](musicgen/README.md)** for full setup and usage.

The `musicgen/` folder contains a fully reproducible local AI music generation pipeline powered by **ACE-Step 1.5**. No cloud API required.

Capabilities:
- **Text-to-music** — describe a style, get a clip
- **Lyrics-to-music** — provide lyrics, ACE-Step sings them
- **Audio-to-audio (cover/remix)** — feed any audio as a structural reference and transform the style on top

> Requires an NVIDIA GPU with 8 GB+ VRAM (tested on RTX 4060 Laptop GPU).

---

## Notes

- macOS serial ports: `/dev/tty.usbmodem*` or `/dev/tty.usbserial*`
- Windows serial ports: `COM3`, `COM4`, etc.
- If streaming drops, close any other app holding the port (Arduino IDE Serial Monitor, PlatformIO monitor, etc.)
- Please **create a new branch** for your work — use your modality name as the branch/folder name
