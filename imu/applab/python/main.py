from arduino.app_utils import *
import socket
import threading
import time

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8888))
server.listen(1)
print("TCP server listening on port 8888...")

clients = []

def accept_clients():
    while True:
        try:
            client, addr = server.accept()
            clients.append(client)
            print(f"Client connected: {addr}")
        except Exception as e:
            print(f"Accept error: {e}")

threading.Thread(target=accept_clients, daemon=True).start()

def loop():
    result = Bridge.call("get_imu_data")
    print(f"Sending: {result}")
    
    dead = []
    for c in clients:
        try:
            bytes_sent = c.sendall((result + "\n").encode())
            print(f"Sent OK")
        except Exception as e:
            print(f"Send error: {e}")
            dead.append(c)
    for c in dead:
        clients.remove(c)
    
    time.sleep(0.02)

App.run(user_loop=loop)