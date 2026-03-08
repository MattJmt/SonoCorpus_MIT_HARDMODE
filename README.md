# SonoCorpus

Python serial collector and live plotter for glove piezo data.

## Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Arduino serial format

Expected line format from firmware (`src/main.cpp`):

- Header (optional): `Index\tMiddle\tRing\tThumb`
- Data rows: 4 numeric values (tab- or comma-separated)
- Default baud: `115200`

## Run the Python tool

From project root:

```bash
python3 tools/serial_plotter.py
```

Default behavior:

- Auto-detects likely Arduino/Pro Micro serial port
- Opens live 4-channel plot (`Index`, `Middle`, `Ring`, `Thumb`)
- Writes CSV to `data/glove_capture_YYYYMMDD_HHMMSS.csv`
- Auto-reconnects if the serial device briefly disconnects

### Useful options

```bash
python3 tools/serial_plotter.py --port /dev/tty.usbmodemXXXX
python3 tools/serial_plotter.py --port COM3
python3 tools/serial_plotter.py --no-plot
python3 tools/serial_plotter.py --no-csv
python3 tools/serial_plotter.py --duration 30
python3 tools/serial_plotter.py --window-sec 15
python3 tools/serial_plotter.py --reconnect-sec 1.5
python3 tools/serial_plotter.py --csv data/my_capture.csv
```

## Notes

- macOS serial ports are usually `/dev/tty.usbmodem*` or `/dev/tty.usbserial*`.
- Windows serial ports are usually `COM3`, `COM4`, etc.
- If streaming drops, close any other app that might be holding the port (Arduino IDE Serial Monitor/Plotter, PlatformIO monitor, etc.).

# SonoCorpus x MIT HARDMODE

48h hackathon, building a multimodal wearable to play music with your body.
Please create new branch for your work
For now, we'll do modality name as folder:

```
source
|_ modality
   |_ imu
   |_ ppg
   |_ emg
   |_ bone
   |_ ...
|_ model (e.g. rave)
```
