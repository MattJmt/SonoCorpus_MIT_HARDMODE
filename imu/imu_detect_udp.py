import socket
from collections import deque
import threading
import time
import numpy as np
from pythonosc import udp_client
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = '127.0.0.1'
PORT = 8888

GRAVITY_ALPHA = 0.98
TAP_THRESHOLD = [5000, 5000, 4800, 3500]  # per-IMU (index=thumb, middle, ring, pinky-side)
TAP_COOLDOWN_SAMPLES = 5   # samples to ignore after a tap (per-IMU)
TAP_GLOBAL_LOCKOUT = 3   #3  # samples all other IMUs are suppressed after any tap

# --- OSC UDP output ---
TARGET_IP  = "10.29.145.118"  # replace with receiver's IP
UDP_PORT   = 9000
osc_client = udp_client.SimpleUDPClient(TARGET_IP, UDP_PORT)

# --- Sample rate estimation ---
packet_times = deque(maxlen=200)
sample_rate_est = 0.0

# --- Global lockout ---
global_lockout = 0

# --- Terminal tap display: counts down per IMU after a tap fires ---
tap_flash = [0] * 4

# --- Per-IMU state ---
class IMUState:
    def __init__(self):
        self.gravity = np.zeros(3)
        self.baseline_samples = deque(maxlen=50)
        self.baseline_ready = False
        self.cooldown = 0

states = [IMUState() for _ in range(4)]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
f = sock.makefile()

def detect_tap(imu_idx, ax, ay, az):
    global global_lockout, tap_flash
    s = states[imu_idx]
    raw = np.array([float(ax), float(ay), float(az)])

    if not s.baseline_ready:
        s.baseline_samples.append(raw)
        if len(s.baseline_samples) == 50:
            s.gravity = np.mean(s.baseline_samples, axis=0)
            s.baseline_ready = True
        return

    s.gravity = GRAVITY_ALPHA * s.gravity + (1 - GRAVITY_ALPHA) * raw

    dynamic = raw - s.gravity
    gravity_norm = np.linalg.norm(s.gravity)
    proj = float(np.dot(dynamic, s.gravity / gravity_norm)) if gravity_norm > 0 else 0.0

    if s.cooldown > 0:
        s.cooldown -= 1
        return

    if global_lockout > 0:
        global_lockout -= 1
        return

    if proj < -TAP_THRESHOLD[imu_idx]:
        s.cooldown = TAP_COOLDOWN_SAMPLES
        global_lockout = TAP_GLOBAL_LOCKOUT
        msg = [1 if i == imu_idx else 0 for i in range(4)]
        osc_client.send_message('/taps', msg)
        tap_flash[imu_idx] = TAP_COOLDOWN_SAMPLES

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
                    for i in range(4):
                        detect_tap(i, vals[i*3], vals[i*3+1], vals[i*3+2])
                    display = ''.join(str(1 if tap_flash[i] > 0 else 0) for i in range(4))
                    print(f"\r{display}", end='', flush=True)
                    for i in range(4):
                        if tap_flash[i] > 0:
                            tap_flash[i] -= 1
        except Exception:
            pass

thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# --- Tap indicator plot ---
TAP_DISPLAY_DURATION = 0.01   # seconds the tile stays lit after a tap
tap_active   = [False] * 4
tap_lit_time = [0.0]   * 4

fig, ax = plt.subplots(figsize=(6, 2))
ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('#111111')

squares = []
labels  = []
for i in range(4):
    x    = i + 0.1
    rect = plt.Rectangle((x, 0.1), 0.8, 0.8,
                          facecolor='#222222', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    squares.append(rect)
    txt = ax.text(x + 0.4, 0.5, str(i),
                  ha='center', va='center',
                  fontsize=28, fontweight='bold', color='white')
    labels.append(txt)

def update(frame):
    now = time.time()
    for i in range(4):
        if tap_flash[i] > 0 and not tap_active[i]:
            tap_active[i]   = True
            tap_lit_time[i] = now
        if tap_active[i]:
            squares[i].set_facecolor('#00FF88')
            labels[i].set_color('black')
            if now - tap_lit_time[i] > TAP_DISPLAY_DURATION:
                tap_active[i] = False
        else:
            squares[i].set_facecolor('#222222')
            labels[i].set_color('white')
    return squares + labels

ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
