import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import numpy as np

HOST = '127.0.0.1'
PORT = 8888
WINDOW = 200

GRAVITY_ALPHA = 0.98
TAP_DISPLAY_DURATION = 0.1
TAP_THRESHOLD = [5000, 5000, 4500, 4000]  # per-IMU (index=thumb, middle, ring, pinky-side)
TAP_COOLDOWN_SAMPLES = 5   # samples to ignore after a tap (per-IMU)
TAP_GLOBAL_LOCKOUT = 3     # samples all other IMUs are suppressed after any tap

# --- Sample rate estimation ---
packet_times = deque(maxlen=200)
sample_rate_est = 0.0

# --- Global lockout: counts down on every sample, blocks all IMUs when > 0 ---
global_lockout = 0

# --- Per-IMU state (gravity only) ---
class IMUState:
    def __init__(self):
        self.gravity = np.zeros(3)
        self.baseline_samples = deque(maxlen=50)
        self.baseline_ready = False
        self.cooldown = 0

states = [IMUState() for _ in range(4)]

data = [deque([0]*WINDOW, maxlen=WINDOW) for _ in range(12)]
grav_data = [deque([0]*WINDOW, maxlen=WINDOW) for _ in range(4)]
imu_labels = [f"IMU{i}_{ax}" for i in range(4) for ax in ['X','Y','Z']]
colors = ['r', 'g', 'b']
grav_colors = ['cyan', 'magenta', 'yellow', 'orange']

tap_active = [False] * 4
tap_lit_time = [0.0] * 4

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
f = sock.makefile()
print("Connected!")

def detect_tap(imu_idx, ax, ay, az):
    global global_lockout
    s = states[imu_idx]
    raw = np.array([float(ax), float(ay), float(az)])

    # Initialize gravity from first 50 resting samples
    if not s.baseline_ready:
        s.baseline_samples.append(raw)
        if len(s.baseline_samples) == 50:
            s.gravity = np.mean(s.baseline_samples, axis=0)
            s.baseline_ready = True
            print(f"IMU {imu_idx} gravity initialized: "
                  f"|g|={np.linalg.norm(s.gravity):.0f} counts  "
                  f"vec={s.gravity.astype(int)}")
        return

    # Update gravity via low-pass filter
    s.gravity = GRAVITY_ALPHA * s.gravity + (1 - GRAVITY_ALPHA) * raw

    # Gravity magnitude for plot
    grav_data[imu_idx].append(float(np.linalg.norm(s.gravity)))

    # Dynamic acceleration (gravity removed)
    dynamic = raw - s.gravity

    # Project dynamic acceleration onto the gravity axis.
    # Negative means accelerating against gravity (finger decelerating into a surface).
    gravity_norm = np.linalg.norm(s.gravity)
    if gravity_norm > 0:
        proj = float(np.dot(dynamic, s.gravity / gravity_norm))
    else:
        proj = 0.0

    if s.cooldown > 0:
        s.cooldown -= 1
        return

    if global_lockout > 0:
        global_lockout -= 1
        return

    if proj < -TAP_THRESHOLD[imu_idx]:
        tap_active[imu_idx] = True
        tap_lit_time[imu_idx] = time.time()
        s.cooldown = TAP_COOLDOWN_SAMPLES
        global_lockout = TAP_GLOBAL_LOCKOUT
        print(f"TAP on IMU {imu_idx}! (proj={proj:.0f}, ~{sample_rate_est:.0f} Hz)")

def read_data():
    global sample_rate_est
    while True:
        try:
            line = f.readline().strip()
            if line:
                vals = list(map(int, line.split(',')))
                if len(vals) == 12:
                    now = time.time()
                    packet_times.append(now)
                    if len(packet_times) >= 10:
                        elapsed = packet_times[-1] - packet_times[0]
                        if elapsed > 0:
                            sample_rate_est = (len(packet_times) - 1) / elapsed
                    for i, v in enumerate(vals):
                        data[i].append(v)
                    for i in range(4):
                        detect_tap(i, vals[i*3], vals[i*3+1], vals[i*3+2])
        except Exception:
            pass

thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# --- Plot layout ---
fig = plt.figure(figsize=(13, 13))
gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 0.8, 0.5], hspace=0.45)

axes = [fig.add_subplot(gs[i]) for i in range(4)]
lines = []

for i, ax in enumerate(axes):
    for j in range(3):
        line, = ax.plot(range(WINDOW), list(data[i*3+j]),
                        color=colors[j], label=imu_labels[i*3+j])
        lines.append(line)
    ax.set_ylabel(f'IMU {i}')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-32768, 32768)
    ax.grid(True, alpha=0.3)
    if i < 3:
        ax.set_xticklabels([])

axes[-1].set_xlabel('Samples')

grav_ax = fig.add_subplot(gs[4])
grav_ax.set_title('Gravity Vector Magnitude per IMU', fontsize=10)
grav_ax.set_ylabel('Counts')
grav_ax.set_ylim(0, 35000)
grav_ax.grid(True, alpha=0.3)
grav_lines = []
for i in range(4):
    gl, = grav_ax.plot(range(WINDOW), list(grav_data[i]),
                       color=grav_colors[i], label=f'IMU {i}')
    grav_lines.append(gl)
grav_ax.legend(loc='upper right', fontsize=7)

tap_ax = fig.add_subplot(gs[5])
tap_ax.set_xlim(0, 4)
tap_ax.set_ylim(0, 1)
tap_ax.axis('off')
tap_ax.set_title('Tap Detection', fontsize=10, pad=4)

squares = []
labels = []
for i in range(4):
    x = i + 0.1
    rect = plt.Rectangle((x, 0.1), 0.8, 0.8,
                          facecolor='#222222',
                          edgecolor='white',
                          linewidth=2)
    tap_ax.add_patch(rect)
    squares.append(rect)
    txt = tap_ax.text(x + 0.4, 0.5, str(i),
                      ha='center', va='center',
                      fontsize=18, fontweight='bold',
                      color='white')
    labels.append(txt)

sr_text = fig.text(0.01, 0.005, 'Sample rate: -- Hz', fontsize=9, color='gray')

fig.suptitle('LSM6DSO32 — 4 IMUs Accelerometer', fontsize=12)

def update(frame):
    now = time.time()

    for i, ln in enumerate(lines):
        ln.set_ydata(list(data[i]))
    for ax in axes:
        ax.relim()
        ax.autoscale_view(scalex=False)

    for i, gl in enumerate(grav_lines):
        gl.set_ydata(list(grav_data[i]))
    grav_ax.relim()
    grav_ax.autoscale_view(scalex=False)

    for i in range(4):
        if tap_active[i]:
            squares[i].set_facecolor('#00FF88')
            labels[i].set_color('black')
            if now - tap_lit_time[i] > TAP_DISPLAY_DURATION:
                tap_active[i] = False
        else:
            squares[i].set_facecolor('#222222')
            labels[i].set_color('white')

    if sample_rate_est > 0:
        sr_text.set_text(f'Sample rate: {sample_rate_est:.1f} Hz')

    return lines + grav_lines + squares + labels

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()
