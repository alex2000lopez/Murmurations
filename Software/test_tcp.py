import socket

IP = "192.168.137.10"   # <-- ESP32 IP from the log
PORT = 5000              # or whatever SERVER_PORT is in your firmware

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP, PORT))
print("Connected, waiting for data...")

while True:
    data = sock.recv(4096)
    if not data:
        print("Connection closed by peer")
        break

    print(f"Got {len(data)} bytes: {data[:32].hex()} ...")
