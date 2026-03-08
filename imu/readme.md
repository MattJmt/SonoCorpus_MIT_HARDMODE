# IMU Streaming — Arduino UNO Q → VSCode

4x LSM6DSO32 IMUs via SparkFun Qwiic Mux → real-time plot in VSCode.

---

## Hardware

- Arduino UNO Q
- SparkFun Qwiic Mux (8-channel)
- 4x LSM6DSO32 IMUs connected to Mux channels **2, 4, 6, 7**
- Qwiic cable: UNO Q Qwiic port → Mux input

---

## Software Stack

| Layer | Tool |
|---|---|
| MCU sketch | Arduino App Lab |
| MCU→Python bridge | Arduino RouterBridge |
| Python→outside | TCP socket (port 8888) |
| Port forwarding | ADB |
| Plotting | VSCode + matplotlib |

---

## Startup Sequence (every time)

### Step 1 — Run App Lab

Open Arduino App Lab and run the HARDMODE project.  
Wait until you see IMU values streaming in the App Lab console, e.g.:
```
1234,-567,8901,233,-100,9800,400,200,-500,100,300,9500
```

---

### Step 2 — Bridge the Docker container (Terminal 1)

App Lab runs inside a Docker container at `172.19.0.2`. Bridge it to the UNO Q Linux system:

```bash
adb shell
```

Once inside the UNO Q shell (`arduino@kallo:/$`):

```bash
socat tcp-listen:8888,bind=0.0.0.0,reuseaddr tcp:172.19.0.2:8888
```

Keep this terminal open and running.

---

### Step 3 — Forward port to your Linux machine (Terminal 2)

Open a new terminal on your Linux machine:

```bash
adb forward tcp:8888 tcp:8888
```

---

### Step 4 — Run VSCode Python script (Terminal 3)

```bash
cd /path/to/folder
python imu_detect.py
```

A live plot of all 4 IMUs (X/Y/Z accelerometer) will appear.  
Move the IMUs to see the signals react in real time.

---

## App Lab Files

### sketch.ino
Reads accelerometer data from 4 IMUs via I2C multiplexer and exposes it over the Bridge RPC.

Key settings:
- I2C bus: `Wire1` (Qwiic connector)
- Mux address: `0x70`
- IMU address: `0x6A`
- IMU channels: `{2, 4, 6, 7}`
- Accel: 104Hz, ±4g

### main.py
Calls `get_imu_data` over the Bridge every 20ms and streams the result to any connected TCP client on port 8888.

---

## VSCode Python (imu_detect.py)

Connects to `127.0.0.1:8888`, reads comma-separated IMU values, and plots them live using matplotlib.

Data format per line:
```
ax0,ay0,az0,ax1,ay1,az1,ax2,ay2,az2,ax3,ay3,az3
```
12 raw int16 accelerometer counts per sample at ~50Hz.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` | Make sure App Lab is running first, then redo Steps 2 & 3 |
| `Address already in use` (socat) | Run `sudo fuser -k 8888/tcp` inside adb shell |
| Plot shows all zeros | App Lab Bridge sketch not running — restart App Lab |
| ADB device not found | Reconnect USB-C cable and run `adb devices` |