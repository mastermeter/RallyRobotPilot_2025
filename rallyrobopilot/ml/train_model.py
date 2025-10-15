# rallyrobopilot/ml/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# Use relative imports to access your other modules
from .model import RobopilotMLP
from .training import prepare_data
from .utils import LossPlotter

# --- Hyperparameters ---
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 100
MODEL_SAVE_PATH = "../../models/robopilot_model_v6.pth" # New model name

def main():
    """Main function to run the training and validation process."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    # --- MODIFICATION: Unpack mean and std ---
    train_loader, val_loader, mean, std = prepare_data(batch_size=BATCH_SIZE, sources=['normal'])
    
    if train_loader is None or val_loader is None:
        print("Data loading failed. Exiting.")
        return

    # 2. Initialize Model, Loss Function, and Optimizer
    hidden_layer_config = [128, 128, 128, 128]
    model = RobopilotMLP(input_size=15,
                         output_size=4,
                         hidden_layers=hidden_layer_config,
                         dropout_rate=0.1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    plotter = LossPlotter()

    print("\n--- Starting Model Training ---")
    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        plotter.update(avg_train_loss, avg_val_loss)

    print("--- Training Complete ---")

    # 4. Save the Model and the Plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_model_path = os.path.join(script_dir, MODEL_SAVE_PATH)
    model_dir = os.path.dirname(final_model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # --- MODIFICATION: Save mean and std instead of max_distance ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_layers': hidden_layer_config,
        'mean': mean,
        'std': std
    }, final_model_path)
    
    print(f"✅ Model and normalization stats saved to {final_model_path}")
    
    final_plot_path = os.path.join(script_dir, "../../report/figures/loss_evolution_v7.png")
    plot_dir = os.path.dirname(final_plot_path)
    os.makedirs(plot_dir, exist_ok=True)
    
    plotter.save(final_plot_path)

if __name__ == "__main__":
    main()