import serial
from collections import deque

PORT = "/dev/ttyACM0"
BAUD = 115200

# Baseline removal window
BASELINE_WINDOW = 80     # ~0.4 s at 200 Hz
# Smoothing window
SMOOTH_WINDOW = 8
# Minimum gap between beats
REFRACTORY_MS = 450
# Hold pulse high for music triggering
PULSE_WIDTH_MS = 120

# Detection threshold on filtered signal
THRESHOLD = 4.0

raw_buf = deque(maxlen=BASELINE_WINDOW)
filt_buf = deque(maxlen=SMOOTH_WINDOW)

prev2 = None
prev1 = None
curr = None

last_peak_time = -10_000
pulse_until = -1

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Reading PPG from", PORT)
print("Columns: t,raw,filtered,beat,pulse")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or "," not in line:
        continue

    try:
        t_str, raw_str = line.split(",")
        t = int(t_str)
        raw = int(raw_str)
    except ValueError:
        continue

    raw_buf.append(raw)
    if len(raw_buf) < raw_buf.maxlen:
        continue

    baseline = sum(raw_buf) / len(raw_buf)
    centered = raw - baseline

    filt_buf.append(centered)
    filtered = sum(filt_buf) / len(filt_buf)

    # Shift samples for local-maximum check
    prev2 = prev1
    prev1 = curr
    curr = filtered

    beat = 0

    # Need 3 points to detect a peak at prev1
    if prev2 is not None and prev1 is not None and curr is not None:
        is_local_max = prev1 > prev2 and prev1 > curr
        above_thresh = prev1 > THRESHOLD
        outside_refractory = (t - last_peak_time) > REFRACTORY_MS

        if is_local_max and above_thresh and outside_refractory:
            beat = 1
            last_peak_time = t
            pulse_until = t + PULSE_WIDTH_MS

    pulse = 1 if t < pulse_until else 0

    print(f"{t},{raw},{filtered:.2f},{beat},{pulse}")