import os
import torch
import torch.nn as nn
import numpy as np

from rallyrobopilot import NetworkDataCmdInterface

from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.uic import loadUi
from PyQt6.QtCore import QTimer
from train import RobopilotMLP

# --- 2. The Autopilot Controller Class ---
class AutopilotController:
    def __init__(self, network_interface):
        self.network_interface = network_interface
        self.model = RobopilotMLP()
        
        # Load the trained weights
        model_path = "robopilot_model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"✅ Model loaded successfully from {model_path}")
        else:
            print(f"❌ Error: Model file not found at {model_path}")
            # Exit or handle the error appropriately
            exit()

        self.model.eval()  # Set the model to evaluation mode

        # State tracking for controls to avoid sending redundant commands
        self.key_states = {"forward": False, "back": False, "left": False, "right": False}

    def process_frame(self, snapshot):
        """Processes a single frame of sensor data to decide on controls."""
        # Prepare the input tensor
        ray_distances = np.array(snapshot.raycast_distances, dtype=np.float32)
        input_tensor = torch.from_numpy(ray_distances).unsqueeze(0)

        # Get model prediction
        with torch.no_grad():
            predictions = self.model(input_tensor)
        print(f"Model raw output (F, B, L, R): {predictions[0].numpy().round(2)}")

        # Decide actions based on a 0.5 threshold
        # predictions[0] is the tensor for [F, B, L, R]
        actions = {
            "forward": predictions[0][0].item() > 0.2,
            "back":    predictions[0][1].item() > 0.5,
            "left":    predictions[0][2].item() > 0.5,
            "right":   predictions[0][3].item() > 0.5,
        }

        # Send commands only when the state changes
        for action, should_press in actions.items():
            if should_press and not self.key_states[action]:
                self.network_interface.send_cmd(f"push {action};")
                self.key_states[action] = True
            elif not should_press and self.key_states[action]:
                self.network_interface.send_cmd(f"release {action};")
                self.key_states[action] = False

# --- 3. Simplified UI for Running the Autopilot ---
class AutopilotUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Note: This still uses the DataCollector.ui file, but we'll ignore the other buttons.
        loadUi("scripts/DataCollector.ui", self)

        self.autopiloting = False
        self.controller = None
        self.network_interface = NetworkDataCmdInterface(self.on_message_received)

        self.timer = QTimer()
        self.timer.timeout.connect(self.network_interface.recv_msg)
        self.timer.start(25) # Check for messages every 25ms
        
        self.AutopilotButton.clicked.connect(self.toggle_autopilot)
        # Hide or disable unnecessary widgets from the original UI
        self.recordDataButton.setHidden(True)
        self.saveRecordButton.setHidden(True)
        self.resetButton.setHidden(True)
        self.saveImgCheckBox.setHidden(True)
        self.forgetSnapshotNumber.setHidden(True)
        self.nbrSnapshotSaved.setHidden(True)

    def toggle_autopilot(self):
        self.autopiloting = not self.autopiloting
        self.AutopilotButton.setText("Autopilot:\n" + ("ON" if self.autopiloting else "OFF"))
        if self.autopiloting and self.controller is None:
            # Initialize the controller the first time autopilot is turned on
            self.controller = AutopilotController(self.network_interface)
            print("🤖 Autopilot Activated.")
        elif not self.autopiloting:
            print("🛑 Autopilot Deactivated.")
            # Release all keys when turning off
            if self.controller:
                for key in self.controller.key_states:
                    self.network_interface.send_cmd(f"release {key};")
                self.controller.key_states = {k: False for k in self.controller.key_states}


    def on_message_received(self, msg):
        if self.autopiloting and self.controller:
            self.controller.process_frame(msg)

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = AutopilotUI()
    window.show()
    sys.exit(app.exec())