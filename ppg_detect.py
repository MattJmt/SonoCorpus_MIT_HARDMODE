"""
Real-time PPG plot from Arduino.
Expects lines like "Signal 556" (prefix + space + PPG value).
Sends OSC: one value per update (0.0 or 1.0).
Peak detection uses a windowed (delayed) confirmation for robustness.
"""
import sys
import time
import numpy as np
import serial
from pythonosc import udp_client
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PORT = "/dev/cu.usbmodem141011"  # Change to your Arduino serial port
BAUD = 115200
MAX_POINTS = 500  # number of samples to show in the plot

# OSC output to Max (same style as emg_udp)
TARGET_IP = "10.29.145.118"
UDP_PORT = 8002
OSC_ADDRESS = "/ppg"
osc_client = udp_client.SimpleUDPClient(TARGET_IP, UDP_PORT)

# Open serial with retry (port is often busy if Serial Monitor or another app has it)
ser = None
for attempt in range(5):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
        break
    except serial.SerialException as e:
        if "Resource busy" in str(e) or "could not open" in str(e).lower():
            print(f"Port {PORT} is busy (attempt {attempt + 1}/5).")
            print("  → Close Arduino Serial Monitor and any other app using the port.")
            print("  → Or run: lsof /dev/cu.usb*  then  kill <PID>")
            if attempt < 4:
                time.sleep(2)
            else:
                print("Failed to open port. Exiting.")
                sys.exit(1)
        else:
            raise
ser.dtr = False
ser.rts = False

# Rolling buffer for plot
data_buffer = deque(maxlen=MAX_POINTS)
data_buffer.extend([0] * MAX_POINTS)  # start with zeros

# Adaptive scaling: y-axis follows data range so small bumps stay visible
AUTO_SCALE = True
MIN_Y_SPAN = 30   # minimum y-axis span so tiny variations stay visible
Y_MARGIN = 0.15   # extra space above/below (fraction of range)
YLIM_SMOOTH = 0.12  # smooth follow (0=instant, higher=slower)

# Peak detection (pulse), robust + delayed:
# We confirm a peak only when it is the maximum in a window centered on it.
# This introduces ~PEAK_HALF_WINDOW samples of delay but prevents "shifting" peaks.
PEAK_HALF_WINDOW = 15        # samples on each side (15 @ 100 Hz ≈ 150 ms delay)
PEAK_MIN_DISTANCE = 35       # minimum samples between confirmed peaks (refractory)
PEAK_MIN_PROMINENCE = 8      # center - min(window) must exceed this

_sample_index = -1  # global sample counter
_detect_vals = deque(maxlen=2 * PEAK_HALF_WINDOW + 1)
_detect_idxs = deque(maxlen=2 * PEAK_HALF_WINDOW + 1)
_last_peak_idx = -10**9
_peak_values = {}  # global_idx -> value (for plotting dots)
_peaks_confirmed = deque()  # global indices of confirmed peaks (monotonic)

fig, ax = plt.subplots(figsize=(10, 4))
(line,) = ax.plot(range(MAX_POINTS), list(data_buffer), "b-", linewidth=0.8)
(peak_dots,) = ax.plot([], [], "ro", markersize=6, label="peaks")
_ylim = [450, 550]  # current ylim; adaptive scaling will update smoothly
ax.set_ylim(_ylim[0], _ylim[1])
ax.set_xlim(0, MAX_POINTS)
ax.set_xlabel("Sample")
ax.set_ylabel("PPG")
ax.set_title("PPG real-time")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()


def try_parse(value):
    """Try to parse a numeric value (handles int or float)."""
    try:
        return float(value.strip())
    except ValueError:
        return None


def _process_sample(v):
    """Feed one sample; return True only when a peak is confirmed (delayed)."""
    global _sample_index, _last_peak_idx

    _sample_index += 1
    _detect_vals.append(float(v))
    _detect_idxs.append(_sample_index)

    if len(_detect_vals) < _detect_vals.maxlen:
        return False

    c = PEAK_HALF_WINDOW
    center_v = _detect_vals[c]

    # Local maximum in the window + strict neighbors check (avoid flat tops)
    if center_v != max(_detect_vals):
        return False
    if center_v <= _detect_vals[c - 1] or center_v <= _detect_vals[c + 1]:
        return False

    prominence = center_v - min(_detect_vals)
    if prominence < PEAK_MIN_PROMINENCE:
        return False

    center_idx = _detect_idxs[c]
    if center_idx - _last_peak_idx < PEAK_MIN_DISTANCE:
        return False

    _last_peak_idx = center_idx
    _peaks_confirmed.append(center_idx)
    _peak_values[center_idx] = center_v
    return True


def update_plot(_):
    new_count = 0
    peak_in_this_update = False
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            break
        parts = raw.split()
        token = parts[-1] if parts else raw
        v = try_parse(token)
        if v is not None:
            data_buffer.append(v)
            new_count += 1
            if _process_sample(v):
                peak_in_this_update = True
    ydata = list(data_buffer)
    line.set_ydata(ydata)

    # Plot confirmed peaks (stable; won't "shift" once confirmed)
    if _sample_index >= 0:
        visible_start = _sample_index - (len(data_buffer) - 1)
        # Drop peaks that scrolled out of view
        while _peaks_confirmed and _peaks_confirmed[0] < visible_start:
            old = _peaks_confirmed.popleft()
            _peak_values.pop(old, None)
        px, py = [], []
        for gi in _peaks_confirmed:
            x = gi - visible_start
            if 0 <= x < len(ydata):
                px.append(x)
                py.append(ydata[int(x)])
        peak_dots.set_data(px, py)

    # Send exactly one digit per update (0 or 1), stable (no oscillation)
    if new_count:
        val = 1.0 if peak_in_this_update else 0.0
        osc_client.send_message(OSC_ADDRESS, val)
        print(int(val))

    if AUTO_SCALE and data_buffer:
        ymin, ymax = min(data_buffer), max(data_buffer)
        span = max(ymax - ymin, MIN_Y_SPAN)
        margin = span * Y_MARGIN
        target_lo, target_hi = ymin - margin, ymax + margin
        _ylim[0] += (target_lo - _ylim[0]) * YLIM_SMOOTH
        _ylim[1] += (target_hi - _ylim[1]) * YLIM_SMOOTH
        ax.set_ylim(_ylim[0], _ylim[1])
    return (line, peak_dots)


print("Reading from", PORT)
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, save_count=100)
try:
    plt.show()
finally:
    if ser is not None:
        ser.close()
