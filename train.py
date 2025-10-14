import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import lzma
import pickle
import glob
import os
from sklearn.model_selection import train_test_split

# This class is a PyTorch standard for handling custom data.
# It makes loading, shuffling, and batching your data simple and efficient.
class RallyDataset(Dataset):
    """Custom Dataset for Rally Robopilot recordings."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Returns a single sample of your data
        return self.features[idx], self.labels[idx]
    
class RobopilotMLP(nn.Module):
    """A simple Multi-Layer Perceptron for the Rally Robopilot."""
    def __init__(self, input_size=15, output_size=4):
        super(RobopilotMLP, self).__init__()
        
        # Define the sequence of layers
        self.layers = nn.Sequential(
            # First hidden layer: takes 15 inputs, produces 128 outputs
            nn.Linear(input_size, 128),
            # ReLU activation: introduces non-linearity
            nn.ReLU(),
            # Second hidden layer: takes 128 inputs, produces 64 outputs
            nn.Linear(128, 64),
            nn.ReLU(),
            # Output layer: takes 64 inputs, produces 4 outputs (for each control)
            nn.Linear(64, output_size),
            # Sigmoid activation: squashes each output to a probability (0 to 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.layers(x)
class LossPlotter:
    """A class to handle live plotting of training and validation loss."""
    def __init__(self):
        # Turn on interactive mode
        plt.ion()
        # Create a figure and axis for the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_losses = []
        self.val_losses = []

        # Set up the plot aesthetics
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.grid(True)
        
    def update(self, train_loss, val_loss):
        """Appends new losses and redraws the plot."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Clear the previous plot
        self.ax.cla()
        
        # Plot the new data
        self.ax.plot(self.train_losses, label='Training Loss', color='blue')
        self.ax.plot(self.val_losses, label='Validation Loss', color='orange')
        
        # Redraw the plot with updated aesthetics
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.legend()
        self.ax.grid(True)
        
        # Pause briefly to allow the plot to update
        plt.pause(0.1)

    def save(self, filepath="loss_evolution.png"):
        """Saves the final plot to a file."""
        self.fig.savefig(filepath)
        print(f"Loss plot saved to {filepath}")
        plt.ioff() # Turn off interactive mode
        plt.show() # Display the final plot until the user closes it

def prepare_data(batch_size=32, test_size=0.2):
    """
    Loads all '.npz' recordings, processes them, and returns
    PyTorch DataLoaders for training and validation.
    """
    print("--- Starting Data Preparation ---")
    
    # 1. Find and load all recording files from the root directory
    all_snapshots = []
    # Assumes this script is in 'autopilot/' and recordings are in the parent folder
    search_path = "record_*.npz" 
    for file_path in glob.glob(search_path):
        print(f"Loading data from: {os.path.basename(file_path)}")
        with lzma.open(file_path, "rb") as f:
            data = pickle.load(f)
            # The SensingSnapshot class definition needs to be available
            # We can import it from the project's package
            from rallyrobopilot.sensing_message import SensingSnapshot
            all_snapshots.extend(data)

    if not all_snapshots:
        print("\n[!] No data found! Make sure 'record_*.npz' files are in the root directory.")
        return None, None

    print(f"\nTotal snapshots loaded: {len(all_snapshots)}")

    # 2. Extract Inputs (X) and Labels (Y)
    # X will be our raycast distances, Y will be the controls
    features = [s.raycast_distances for s in all_snapshots]
    labels = [s.current_controls for s in all_snapshots]

    # Convert to NumPy arrays for easier processing
    X = np.array(features, dtype=np.float32)
    # Convert boolean labels to float (0.0 or 1.0)
    Y = np.array(labels, dtype=np.float32)

    # 3. Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_val)} validation samples.")

    # 4. Convert NumPy arrays to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train)
    Y_train_tensor = torch.from_numpy(Y_train)
    X_val_tensor = torch.from_numpy(X_val)
    Y_val_tensor = torch.from_numpy(Y_val)

    # 5. Create Dataset and DataLoader instances
    train_dataset = RallyDataset(X_train_tensor, Y_train_tensor)
    val_dataset = RallyDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    print("--- Data Preparation Complete ---")
    return train_loader, val_loader

if __name__ == '__main__':
    # --- 1. Hyperparameters and Setup ---
    NUM_EPOCHS = 25
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "robopilot_model.pth"

    # --- 2. Data Preparation ---
    train_loader, val_loader = prepare_data(batch_size=BATCH_SIZE)

    if train_loader is None or val_loader is None:
        print("Exiting due to data loading failure.")
    else:
        # --- 3. Model, Loss, and Optimizer Initialization ---
        model = RobopilotMLP()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # --- NEW: Initialize the plotter ---
        plotter = LossPlotter()

        print("\n--- Starting Model Training ---")
        # --- 4. The Training Loop ---
        for epoch in range(NUM_EPOCHS):
            # ... (Training Phase code remains the same) ...
            model.train()
            total_train_loss = 0
            for i, (features, labels) in enumerate(train_loader):
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            # ... (Validation Phase code remains the same) ...
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

            # --- NEW: Update the plot ---
            plotter.update(avg_train_loss, avg_val_loss)

        print("--- Model Training Complete ---")
        
        # --- 5. Save the Trained Model and the Plot ---
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
        
        # --- NEW: Save the final plot ---
        plotter.save()