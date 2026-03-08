from pythonosc import udp_client
import time

TARGET_IP = "10.29.145.118"  # replace with their IP
PORT = 8000

client = udp_client.SimpleUDPClient(TARGET_IP, PORT)
print(f"Sending to {TARGET_IP}:{PORT}...")

i = 0
while True:
    client.send_message('/test', 1)
    print(f"sent {i}")
    i += 1
    time.sleep(0.1)