# rallyrobopilot/ml/train_model.py

import os
import sys
# --- MODIFICATION: Import argparse ---
import argparse
# --- END MODIFICATION ---

# CRITICAL: Ajouter le chemin des DLLs PyTorch AVANT tout import
torch_lib = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    # Également ajouter au PATH pour compatibilité
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
import torch
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.optim.lr_scheduler import StepLR

# Use relative imports to access your other modules
from .model import RobopilotMLP
from .training import prepare_data
from .utils import LossPlotter

# --- Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30 
EARLY_STOPPING_MIN_DELTA = 0.0001

# --- MODIFICATION: Removed hardcoded MODEL_SAVE_PATH ---

def main():
    """Main function to run the training and validation process."""

    # --- MODIFICATION: Add command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Train the Robopilot model.")
    parser.add_argument(
        "--name",
        type=str,
        default="run_default",
        help="A name for this training run, used for saving model and figure files (e.g., 'v3_more_data')."
    )
    args = parser.parse_args()
    run_name = args.name
    print(f"🚀 Starting training run: '{run_name}'")
    # --- END MODIFICATION ---

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, mean, std = prepare_data(batch_size=BATCH_SIZE, sources=['bords','normal','reverse','frein-gaz'], remove_outliers=True)
    
    if train_loader is None or val_loader is None:
        print("Data loading failed. Exiting.")
        return

    # 2. Initialize Model, Loss Function, and Optimizer
    hidden_layer_config = [200, 150, 100, 50]
    model = RobopilotMLP(input_size=16,
                         output_size=4,
                         hidden_layers=hidden_layer_config,
                         dropout_rate=0.15).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    plotter = LossPlotter()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    # Early Stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = ""

    print("\n" + "="*50)
    print(f"{'TRAINING CONFIGURATION':^50}")
    print("="*50)
    print(f" Run Name: \t\t{run_name}")
    print(f" Device: \t\t{device}")
    print(f" Data Sources: \t\t{['normal', 'reverse']}")
    print("-" * 50)
    print("Hyperparameters:")
    print(f"  - Learning Rate: \t{LEARNING_RATE}")
    print(f"  - Batch Size: \t{BATCH_SIZE}")
    print(f"  - Max Epochs: \t{NUM_EPOCHS}")
    print(f"  - Dropout Rate: \t{0.15}") # From model init
    print(f"  - Weight Decay: \t{optimizer.defaults['weight_decay']}")
    print(f"  - LR Scheduler: \tStepLR (step={scheduler.step_size}, gamma={scheduler.gamma})")
    print("-" * 50)
    print("Early Stopping:")
    print(f"  - Patience: \t\t{EARLY_STOPPING_PATIENCE} epochs")
    print(f"  - Min Delta: \t\t{EARLY_STOPPING_MIN_DELTA}")
    print("-" * 50)
    print("Model Architecture:")
    print(f"  - Hidden Layers: \t{hidden_layer_config}")
    print("="*50)

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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        plotter.update(avg_train_loss, avg_val_loss)

        # Early Stopping Logic
        if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # --- MODIFICATION: Use run_name for dynamic filename ---
            best_model_path = os.path.join(script_dir, f"../../models/robopilot_model_{run_name}.pth")
            # --- END MODIFICATION ---
            model_dir = os.path.dirname(best_model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_layers': hidden_layer_config,
                'mean': mean,
                'std': std
            }, best_model_path)
            print(f"   -> New best model saved with Val Loss: {best_val_loss:.4f}")

        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n--- Early stopping triggered after {epoch + 1} epochs. ---")
                break

    print("--- Training Complete ---")

    if best_model_path:
        print(f"✅ Best model and normalization stats saved to {best_model_path}")
    else:
        print("No model was saved as validation loss did not improve.")

    # --- MODIFICATION: Use run_name for dynamic filename ---
    final_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../figures/loss_evolution_{run_name}.png")
    # --- END MODIFICATION ---
    plot_dir = os.path.dirname(final_plot_path)
    os.makedirs(plot_dir, exist_ok=True)
    
    plotter.save(final_plot_path)

if __name__ == "__main__":
    main()