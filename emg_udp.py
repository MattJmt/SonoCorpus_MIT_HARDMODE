import serial
from pythonosc import udp_client

# --- Serial input from Arduino ---
SERIAL_PORT = "/dev/cu.usbserial-120"   # change if needed
BAUD_RATE = 115200

# --- OSC UDP output to Max ---
TARGET_IP = "10.29.145.118"   # Max on same computer
UDP_PORT = 8001
OSC_ADDRESS = "/emg"

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
osc_client = udp_client.SimpleUDPClient(TARGET_IP, UDP_PORT)

print("Reading EMG from serial and sending via OSC...")

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        value = float(line)
        value = max(0.0, min(1.0, value))  # clamp to 0..1

        osc_client.send_message(OSC_ADDRESS, value)
        print(f"EMG: {value:.3f}")

    except Exception:
        pass