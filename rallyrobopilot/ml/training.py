# rallyrobopilot/ml/training.py
import torch
import numpy as np
import lzma
import pickle
import glob
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import RallyDataset # Note the relative import

def prepare_data(batch_size=32, test_size=0.2, sources=['normal']):
    """
    Loads '.npz' recordings from specific subdirectories within the 'data' folder,
    processes them, and returns PyTorch DataLoaders for training and validation.
    """
    print(f"--- Starting Data Preparation for sources: {sources} ---")
    
    all_snapshots = []
    
    for source_folder in sources:
        search_path = os.path.join("data", source_folder, "*.npz")
        file_paths = glob.glob(search_path)
        
        if not file_paths:
            print(f"[!] Warning: No data found in 'data/{source_folder}'")
            continue

        for file_path in file_paths:
            print(f"Loading data from: {file_path}")
            with lzma.open(file_path, "rb") as f:
                data = pickle.load(f)
                try:
                    from rallyrobopilot.sensing_message import SensingSnapshot
                except ImportError:
                    # Define a dummy class if the main package isn't installed
                    # This allows the script to unpickle the data
                    class SensingSnapshot: pass
                all_snapshots.extend(data)

    if not all_snapshots:
        print("\n[!] No data was loaded! Check the 'sources' argument and your folder structure.")
        return None, None

    print(f"\nTotal snapshots loaded: {len(all_snapshots)}")

    features = [s.raycast_distances for s in all_snapshots]
    labels = [s.current_controls for s in all_snapshots]

    X = np.array(features, dtype=np.float32)
    Y = np.array(labels, dtype=np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_val)} validation samples.")

    X_train_tensor = torch.from_numpy(X_train)
    Y_train_tensor = torch.from_numpy(Y_train)
    X_val_tensor = torch.from_numpy(X_val)
    Y_val_tensor = torch.from_numpy(Y_val)

    train_dataset = RallyDataset(X_train_tensor, Y_train_tensor)
    val_dataset = RallyDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    print("--- Data Preparation Complete ---")
    return train_loader, val_loader