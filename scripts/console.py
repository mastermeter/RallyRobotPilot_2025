# scripts/console.py

import os
import sys
import torch
import numpy as np
import argparse
import lzma
import pickle

from rallyrobopilot import NetworkDataCmdInterface
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt, QTimer, QThread
from PyQt6.QtGui import QKeyEvent
from rallyrobopilot.ml.model import RobopilotMLP
from colorama import Fore, Style, init

# Stylesheet for a clean, modern look
CLEAN_STYLE = """
    QMainWindow { background-color: #2c2f33; }
    QLineEdit { background-color: #40444b; color: #dcddde; border: 1px solid #202225; border-radius: 3px; padding: 2px; }
    QPushButton {
        background-color: #40444b; color: #b9bbbe; border: none;
        border-radius: 5px; font-family: 'Segoe UI', sans-serif;
        font-size: 10pt; font-weight: bold; padding: 5px;
    }
    QPushButton:hover { background-color: #52575d; }
    QPushButton:pressed { background-color: #3a3e44; }
    QPushButton:disabled { background-color: #202225; color: #72767d; }
    QPushButton#AutopilotButton { color: #ffffff; background-color: #7289da; }
    QPushButton#AutopilotButton:hover { background-color: #677bc4; }
    QPushButton#recordDataButton[recording="true"] { background-color: #f04747; color: white; }
    QLabel { color: #b9bbbe; font-family: 'Segoe UI', sans-serif; font-size: 10pt; }
    QLabel#connectionLed { font-weight: bold; color: white; border-radius: 15px; }
"""

class AutopilotController:
    """Handles the model inference and sends driving commands."""
    def __init__(self, network_interface, model_path, activation_threshold=0.15, num_features=15):
        self.network_interface = network_interface
        self.num_features = num_features
        
        if not os.path.exists(model_path):
            print(f"{Fore.RED}❌ Error: Model file not found at {model_path}{Style.RESET_ALL}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, weights_only=False)

        if 'mean' not in checkpoint or 'std' not in checkpoint:
            print(f"{Fore.RED}❌ Error: Model checkpoint is missing 'mean' or 'std' for normalization.{Style.RESET_ALL}")
            raise ValueError("Checkpoint missing normalization data.")
            
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        
        # --- MODIFICATION: Vérifier la cohérence des features ---
        if len(self.mean) != self.num_features:
            print(f"{Fore.RED}❌ Error: Feature count mismatch! UI expects {self.num_features} but model was trained with {len(self.mean)}.{Style.RESET_ALL}")
            raise ValueError("Feature count mismatch.")

        hidden_layers_from_file = checkpoint.get('hidden_layers', [128, 64])
        print(f"{Fore.CYAN}   Instantiating model with {self.num_features} inputs and architecture: {hidden_layers_from_file}{Style.RESET_ALL}")

        self.model = RobopilotMLP(input_size=self.num_features, hidden_layers=hidden_layers_from_file, dropout_rate=0.0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.activation_threshold = activation_threshold
        print(f"{Fore.GREEN}✅ Model and normalization stats loaded successfully from {model_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   Using activation threshold: {self.activation_threshold}{Style.RESET_ALL}")
        
        self.model.eval()
        self.key_states = {"forward": False, "back": False, "left": False, "right": False}

    def process_frame(self, snapshot):
        ray_distances = np.array(snapshot.raycast_distances, dtype=np.float32)

        if self.num_features == 16:
            car_speed = np.array([snapshot.car_speed], dtype=np.float32)
            features = np.concatenate([ray_distances, car_speed])
        else:
            features = ray_distances
        
        normalized_features = (features - self.mean) / self.std
        features_str = ", ".join([f"{d:.2f}" for d in normalized_features])
        print(f"{Fore.YELLOW}Normalized Input ({self.num_features}): [{features_str}]{Style.RESET_ALL}")

        input_tensor = torch.from_numpy(normalized_features).unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(input_tensor)

        print(f"Model Predictions: Fwd={predictions[0][0]:.2f}, Bck={predictions[0][1]:.2f}, Lft={predictions[0][2]:.2f}, Rgt={predictions[0][3]:.2f}")
        
        actions = {
            "forward": predictions[0][0].item() > self.activation_threshold,
            "back":    predictions[0][1].item() > self.activation_threshold,
            "left":    predictions[0][2].item() > self.activation_threshold,
            "right":   predictions[0][3].item() > self.activation_threshold,
        }

        for action, should_press in actions.items():
            if should_press and not self.key_states[action]:
                self.network_interface.send_cmd(f"push {action};")
                self.key_states[action] = True
            elif not should_press and self.key_states[action]:
                self.network_interface.send_cmd(f"release {action};")
                self.key_states[action] = False

class WorkConsole(QMainWindow):
    def __init__(self, activation_threshold):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "Console.ui")
        loadUi(ui_path, self)

        self.autopiloting = False
        self.recording = False
        self.connected_to_game = False
        self.model_path = None
        self.activation_threshold = activation_threshold
        self.controller = None
        self.recorded_data = []
        self.saving_worker = None
        self.save_path = 'data'

        self.network_interface = NetworkDataCmdInterface(self.on_message_received, autoconnect=True)
        self.msg_timer = QTimer()
        self.msg_timer.timeout.connect(self.network_interface.recv_msg)
        self.msg_timer.start(25)

        self.connection_timer = QTimer()
        self.connection_timer.setSingleShot(True)
        self.connection_timer.timeout.connect(self.on_connection_lost)

        # --- Connexion des nouveaux widgets ---
        self.browseModelButton.clicked.connect(self.browse_for_model)
        self.featuresComboBox.currentIndexChanged.connect(self.update_ui_state)

        self.forwardButton.pressed.connect(lambda: self.send_manual_key("forward", True))
        self.forwardButton.released.connect(lambda: self.send_manual_key("forward", False))
        self.backwardButton.pressed.connect(lambda: self.send_manual_key("back", True))
        self.backwardButton.released.connect(lambda: self.send_manual_key("back", False))
        self.leftButton.pressed.connect(lambda: self.send_manual_key("left", True))
        self.leftButton.released.connect(lambda: self.send_manual_key("left", False))
        self.rightButton.pressed.connect(lambda: self.send_manual_key("right", True))
        self.rightButton.released.connect(lambda: self.send_manual_key("right", False))

        self.AutopilotButton.clicked.connect(self.toggle_autopilot)
        self.recordDataButton.clicked.connect(self.toggle_record)
        self.saveRecordButton.clicked.connect(self.save_record)
        
        self.browseButton.clicked.connect(self.browse_for_folder)
        self.filenameEdit.textChanged.connect(self.update_ui_state)
        self.filenameEdit.setPlaceholderText("e.g., good_run_1")
        self.savePathLabel.setText(f"Path: {self.save_path}")

        self.set_connection_status(False)
        self.update_ui_state() # Appel initial pour tout désactiver

    def browse_for_model(self):
        # Ouvre une boîte de dialogue pour choisir un fichier .pth
        fname, _ = QFileDialog.getOpenFileName(self, 'Choisir un modèle', 'models', "PyTorch Models (*.pth)")
        if fname:
            self.model_path = fname
            # Affiche le nom du fichier (tronqué si trop long)
            display_name = os.path.basename(fname)
            if len(display_name) > 25:
                display_name = "..." + display_name[-22:]
            self.modelPathLabel.setText(display_name)
            self.controller = None # Réinitialiser le contrôleur pour le forcer à se recharger
        self.update_ui_state()

    def browse_for_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "Choisir un dossier", self.save_path)
        if directory:
            self.save_path = directory
            display_path = self.save_path
            if len(display_path) > 25:
                display_path = "..." + display_path[-22:]
            self.savePathLabel.setText(f"Path: {display_path}")
            self.update_ui_state()

    def on_message_received(self, msg):
        if not self.connected_to_game:
            print("Connection to game server established.")
        self.set_connection_status(True)
        self.connection_timer.start(2000)

        # --- MODIFICATION: Logique d'enregistrement ---
        if self.recording:
            # S'assurer que l'image est None et que les données sont enregistrées
            msg.image = None 
            self.recorded_data.append(msg)
            self.nbrSnapshotSaved.setText(str(len(self.recorded_data)))
        elif self.autopiloting and self.controller:
            self.controller.process_frame(msg)
    
    def on_connection_lost(self):
        if self.connected_to_game:
            print(f"{Fore.RED}Connection to game server lost.{Style.RESET_ALL}")
        self.set_connection_status(False)
        self.autopiloting = False
        self.recording = False
        self.update_ui_state()

    def set_connection_status(self, connected):
        self.connected_to_game = connected
        self.connectionLed.setStyleSheet("background-color: %s;" % ("#43b581" if connected else "#f04747"))
        self.connectionStatusLabel.setText("Connected" if connected else "Disconnected")
        self.update_ui_state()
    
    def update_ui_state(self):
        # Activer/désactiver le bouton Autopilot
        can_autopilot = self.connected_to_game and self.model_path is not None
        self.AutopilotButton.setEnabled(can_autopilot)
        self.AutopilotButton.setText("Autopilot (Q):\n" + ("ON" if self.autopiloting else "OFF"))

        # Logique pour l'enregistrement
        self.recordDataButton.setEnabled(self.connected_to_game and not self.autopiloting)
        self.recordDataButton.setText("Stop Rec (R)" if self.recording else "Record (R)")
        self.recordDataButton.setProperty("recording", self.recording)
        self.recordDataButton.style().polish(self.recordDataButton)

        # Contrôles manuels
        enable_manual = self.connected_to_game and not self.autopiloting and not self.recording
        self.forwardButton.setEnabled(enable_manual)
        self.backwardButton.setEnabled(enable_manual)
        self.leftButton.setEnabled(enable_manual)
        self.rightButton.setEnabled(enable_manual)
        
        # Logique pour la sauvegarde
        can_save = len(self.recorded_data) > 0 and not self.recording and self.filenameEdit.text() != ""
        self.saveRecordButton.setEnabled(can_save)

    def toggle_autopilot(self):
        if not self.AutopilotButton.isEnabled(): return
        
        self.autopiloting = not self.autopiloting
        
        if self.autopiloting:
            try:
                # Si le contrôleur n'existe pas ou si le modèle a changé, le charger
                if self.controller is None:
                    # Récupérer la valeur de la ComboBox (0 pour 15, 1 pour 16)
                    num_features = 16 if self.featuresComboBox.currentIndex() == 1 else 15
                    self.controller = AutopilotController(self.network_interface, self.model_path, self.activation_threshold, num_features)
                print(f"{Fore.CYAN}🤖 Autopilot Activated.{Style.RESET_ALL}")
            except Exception as e:
                # Si le chargement échoue (ex: mauvais nb de features), désactiver l'autopilot
                print(f"{Fore.RED}❌ Failed to activate autopilot: {e}{Style.RESET_ALL}")
                self.autopiloting = False
                self.controller = None
        else:
            print(f"{Fore.MAGENTA}🛑 Autopilot Deactivated.{Style.RESET_ALL}")
            if self.controller:
                for key in self.controller.key_states:
                    if self.controller.key_states[key]: self.send_manual_key(key, False)
        
        self.update_ui_state()
        
    def toggle_record(self):
        if not self.recordDataButton.isEnabled(): return
        self.recording = not self.recording
        print(f"{Fore.CYAN}Recording {'ON' if self.recording else 'OFF'}{Style.RESET_ALL}")
        self.update_ui_state()

    def send_manual_key(self, action, is_pressed):
        cmd_type = "push" if is_pressed else "release"
        self.network_interface.send_cmd(f"{cmd_type} {action};")

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat(): return
        key = event.key()
        if key == Qt.Key.Key_R: self.toggle_record()
        elif key == Qt.Key.Key_Q: self.toggle_autopilot()
        elif key == Qt.Key.Key_E and self.saveRecordButton.isEnabled(): self.save_record()

        key_map = {Qt.Key.Key_W: "forward", Qt.Key.Key_S: "back", Qt.Key.Key_A: "left", Qt.Key.Key_D: "right"}
        if self.forwardButton.isEnabled() and key in key_map:
            self.send_manual_key(key_map[key], True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat(): return
        key = event.key()
        key_map = {Qt.Key.Key_W: "forward", Qt.Key.Key_S: "back", Qt.Key.Key_A: "left", Qt.Key.Key_D: "right"}
        if key in key_map: self.send_manual_key(key_map[key], False)
        super().keyReleaseEvent(event)

    def save_record(self):
        # ... (Cette fonction reste inchangée) ...
        if self.saving_worker is not None or not self.saveRecordButton.isEnabled(): return
        if self.recording: self.toggle_record()
        self.saveRecordButton.setText("Saving...")
        base_filename = self.filenameEdit.text()
        full_path = os.path.join(self.save_path, f"{base_filename}.npz")
        class ThreadedSaver(QThread):
            def __init__(self, path, data):
                super().__init__()
                self.path = path
                self.data = data
            def run(self):
                with lzma.open(self.path, "wb") as f:
                    pickle.dump(self.data, f)
        self.saving_worker = ThreadedSaver(full_path, self.recorded_data)
        self.recorded_data = []
        self.nbrSnapshotSaved.setText("0")
        self.saving_worker.finished.connect(self.on_record_save_done)
        self.saving_worker.start()

    def on_record_save_done(self):
        # ... (Cette fonction reste inchangée) ...
        print(f"[+] Recorded data saved to {self.saving_worker.path}")
        self.saving_worker = None
        self.saveRecordButton.setText("Save (E)")
        self.filenameEdit.clear()
        self.update_ui_state()


if __name__ == "__main__":
    init()
    # --- MODIFICATION: Argument de ligne de commande simplifié ---
    parser = argparse.ArgumentParser(description="Run the Rally Robopilot Work Console")
    parser.add_argument("-t", "--threshold", type=float, default=0.15, help="Activation threshold for model predictions (e.g., 0.2)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(CLEAN_STYLE)
    
    window = WorkConsole(activation_threshold=args.threshold)
    window.show()
    sys.exit(app.exec())