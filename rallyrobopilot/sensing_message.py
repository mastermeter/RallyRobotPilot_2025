# rallyrobopilot/sensing_message.py

import pickle
import socket
import lzma
import struct # <-- Import the struct module

class SensingSnapshot:
    def __init__(self):
        self.image = None
        self.raycast_distances = []
        self.car_position = (0, 0, 0)
        self.car_speed = 0
        self.car_angle = 0
        self.current_controls = (False, False, False, False)

# --- MODIFIED CLASS: SensingSnapshotManager ---
class SensingSnapshotManager:
    def __init__(self):
        self.pending_data = b''
        # Header is a 4-byte unsigned integer
        self.header_size = struct.calcsize('!I') 

    def pack(self, snapshot):
        """Packs a snapshot into a binary message with a length header."""
        payload = lzma.compress(pickle.dumps(snapshot))
        header = struct.pack('!I', len(payload)) # '!' for network byte order
        return header + payload

    def parse(self):
        """
        Parses the pending data buffer to extract complete messages.
        Returns a list of unpacked snapshot objects.
        """
        messages = []
        while len(self.pending_data) > self.header_size:
            # Read the length of the next message
            payload_len = struct.unpack('!I', self.pending_data[:self.header_size])[0]
            
            # Check if the full message has been received
            full_message_len = self.header_size + payload_len
            if len(self.pending_data) < full_message_len:
                break # Not enough data yet, wait for more

            # Extract the payload
            payload = self.pending_data[self.header_size:full_message_len]
            
            try:
                snapshot = pickle.loads(lzma.decompress(payload))
                messages.append(snapshot)
            except (lzma.LZMAError, pickle.UnpicklingError) as e:
                print(f"[Parser] Error decoding message: {e}")

            # Remove the processed message from the buffer
            self.pending_data = self.pending_data[full_message_len:]
        
        return messages


class NetworkDataCmdInterface:
    def __init__(self, msg_callback, autoconnect=True):
        self.ip_address = "127.0.0.1"
        self.port = 7654
        self.client_socket = None
        self.msg_callback = msg_callback
        self.snapshot_manager = SensingSnapshotManager()
        self.autoconnect = autoconnect

        if self.autoconnect:
            self.connect_to_server()

    def connect_to_server(self):
        """Initializes the connection to the server."""
        try:
            if self.client_socket is None:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(0.01)
                self.client_socket.connect((self.ip_address, self.port))
                print("[Network] Connection established with the server.")
        except Exception as e:
            print(f"[Network] Failed to connect to server: {e}")
            self.client_socket = None

    def send_cmd(self, command_str):
        """Sends a command string to the server."""
        if self.client_socket:
            try:
                self.client_socket.sendall(command_str.encode('utf-8'))
            except Exception as e:
                print(f"[Network] Error sending command: {e}")
                self.client_socket = None

    # --- MODIFIED METHOD: recv_msg ---
    def recv_msg(self):
        """Receives and processes messages from the server."""
        if self.client_socket:
            try:
                # Receive new data and add it to the buffer
                data = self.client_socket.recv(4096)
                if not data:
                    print("[Network] Server closed the connection.")
                    self.client_socket.close()
                    self.client_socket = None
                    return
                
                self.snapshot_manager.pending_data += data
                
                # Try to parse any complete messages from the buffer
                unpacked_messages = self.snapshot_manager.parse()
                
                for msg in unpacked_messages:
                    if self.msg_callback:
                        self.msg_callback(msg)
            except socket.timeout:
                pass 
            except Exception as e:
                print(f"[Network] Error receiving data: {e}")
                self.client_socket.close()
                self.client_socket = None