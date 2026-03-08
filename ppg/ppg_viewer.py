import subprocess
import sys
from collections import deque
from pathlib import Path
import select

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WINDOW_MS = 20_000

times = deque()
raw_vals = deque()
filtered_vals = deque()
pulse_vals = deque()
beat_times = deque()
beat_vals = deque()

SCRIPT_DIR = Path(__file__).resolve().parent

proc = subprocess.Popen(
    [sys.executable, "-u", str(SCRIPT_DIR / "ppg_detect.py")],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd=SCRIPT_DIR,
)

fig, ax = plt.subplots(figsize=(12, 6))

raw_line, = ax.plot([], [], label="raw")
filtered_line, = ax.plot([], [], label="filtered")
pulse_line, = ax.plot([], [], label="pulse")
beat_scatter = ax.scatter([], [], label="beat", s=40)

ax.set_title("PPG Viewer (20s window)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal")
ax.legend()
ax.grid(True)

def trim_buffers(current_t):
    cutoff = current_t - WINDOW_MS

    while times and times[0] < cutoff:
        times.popleft()
        raw_vals.popleft()
        filtered_vals.popleft()
        pulse_vals.popleft()

    while beat_times and beat_times[0] < cutoff:
        beat_times.popleft()
        beat_vals.popleft()

def read_available_lines(max_lines=300):
    count = 0
    while count < max_lines:
        if not proc.stdout:
            break
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break

        line = proc.stdout.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 5:
            continue

        try:
            t = int(parts[0])
            raw = float(parts[1])
            filtered = float(parts[2])
            beat = int(parts[3])
            pulse = float(parts[4])
        except ValueError:
            continue

        times.append(t)
        raw_vals.append(raw)
        filtered_vals.append(filtered)
        pulse_vals.append(pulse)

        if beat == 1:
            beat_times.append(t)
            beat_vals.append(filtered)

        trim_buffers(t)
        count += 1

def update(_frame):
    read_available_lines()

    if not times:
        return raw_line, filtered_line, pulse_line, beat_scatter

    t0 = times[0]
    x = [(t - t0) / 1000.0 for t in times]

    raw_line.set_data(x, list(raw_vals))
    filtered_line.set_data(x, list(filtered_vals))

    if filtered_vals:
        fmin = min(filtered_vals)
        fmax = max(filtered_vals)
        span = max(fmax - fmin, 1.0)
        pulse_scaled = [fmin + 0.1 * span + p * 0.2 * span for p in pulse_vals]
    else:
        pulse_scaled = list(pulse_vals)

    pulse_line.set_data(x, pulse_scaled)

    bx = [(t - t0) / 1000.0 for t in beat_times]
    if bx:
        offsets = np.column_stack([bx, list(beat_vals)])
    else:
        offsets = np.empty((0, 2))
    beat_scatter.set_offsets(offsets)

    ax.set_xlim(max(0, x[-1] - 20), max(20, x[-1]))

    all_y = list(raw_vals) + list(filtered_vals) + pulse_scaled
    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        pad = max((ymax - ymin) * 0.1, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)

    return raw_line, filtered_line, pulse_line, beat_scatter

ani = FuncAnimation(fig, update, interval=33, blit=False)

try:
    plt.show()
finally:
    proc.terminate()
