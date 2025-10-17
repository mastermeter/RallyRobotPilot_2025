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

class SensingSnapshot:
    pass

def prepare_data(batch_size=32, test_size=0.2, sources=['frein-gaz'], remove_outliers=False):
    """
    Loads '.npz' recordings, processes them with Z-score normalization,
    and returns DataLoaders and normalization statistics.

    Args:
        batch_size (int): The batch size for the DataLoader.
        test_size (float): The proportion of the dataset to allocate to the validation set.
        sources (list): A list of subdirectories in the 'data' folder to load data from.
        remove_outliers (bool): If True, removes samples containing features with a Z-score > 3.0.
    """
    print(f"--- Starting Data Preparation for sources: {sources} ---")

    # --- Configuration for Augmentation ---
    AUGMENTATION_FACTOR = 5 # The desired total number of samples per original sample
    NOISE_LEVEL = 5         # Standard deviation of the Gaussian noise. Max raycast is 100.
    # --- End Configuration ---

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_base_path = os.path.join(project_root, "data")

    all_snapshots = []

    for source_folder in sources:
        search_path = os.path.join(data_base_path, source_folder, "*.npz")
        file_paths = glob.glob(search_path)

        if not file_paths:
            print(f"[!] Warning: No data found in '{data_base_path}/{source_folder}'")
            continue

        for file_path in file_paths:
            print(f"Loading data from: {file_path}")
            with lzma.open(file_path, "rb") as f:
                data = pickle.load(f)
                all_snapshots.extend(data)

    if not all_snapshots:
        print("\n[!] No data was loaded! Check the 'sources' argument and your folder structure.")
        return None, None, None, None

    print(f"\nTotal snapshots loaded: {len(all_snapshots)}")

    features = [s.raycast_distances + [s.car_speed] for s in all_snapshots]
    labels = [s.current_controls for s in all_snapshots]

    X = np.array(features, dtype=np.float32)
    Y = np.array(labels, dtype=np.float32)

    # --- Check for Zero Values ---
    zero_count = np.sum(X == 0.0)
    total_count = X.size
    zero_percentage = (zero_count / total_count) * 100
    print(f"Total zero raycast readings: {zero_count} out of {total_count} ({zero_percentage:.2f}%)")
    # ------------------------------

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )

    # --- Data Augmentation Pipeline ---
    if AUGMENTATION_FACTOR > 1:
        print(f"Augmenting training data by a factor of {AUGMENTATION_FACTOR}...")
        X_train_augmented = [X_train]
        Y_train_augmented = [Y_train]

        # We already have 1 copy, so we need to create (AUGMENTATION_FACTOR - 1) more.
        for _ in range(AUGMENTATION_FACTOR - 1):
            noise = np.random.normal(loc=0.0, scale=NOISE_LEVEL, size=X_train.shape)
            noisy_X = np.clip(X_train + noise, 0, 100000).astype(np.float32)
            X_train_augmented.append(noisy_X)
            Y_train_augmented.append(Y_train)

        X_train = np.concatenate(X_train_augmented, axis=0)
        Y_train = np.concatenate(Y_train_augmented, axis=0)
        print(f"Augmentation complete. New training set size: {len(X_train)}")
    # --- END Augmentation ---

    # Z-Score (Gaussian) Normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    print("\nNormalization stats (Mean/Std) calculated from training data.")

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # --- Outlier Detection (Z-score magnitude > 3.0) ---
    outlier_threshold = 3.0

    # Check training data
    outliers_mask_train = np.abs(X_train) > outlier_threshold
    outliers_train_count = np.sum(outliers_mask_train)
    outliers_train_percentage = (outliers_train_count / X_train.size) * 100
    print(f"Train Outliers (>|{outlier_threshold}|): {outliers_train_count} ({outliers_train_percentage:.4f}%)")

    # Check validation data
    outliers_mask_val = np.abs(X_val) > outlier_threshold
    outliers_val_count = np.sum(outliers_mask_val)
    outliers_val_percentage = (outliers_val_count / X_val.size) * 100
    print(f"Validation Outliers (>|{outlier_threshold}|): {outliers_val_count} ({outliers_val_percentage:.4f}%)")
    # ----------------------------------------------------

    # --- NEW: Outlier Removal ---
    if remove_outliers:
        print("\n--- Removing Outlier Samples ---")

        # Identify entire samples (rows) in the training set containing at least one outlier
        outlier_samples_mask_train = np.any(outliers_mask_train, axis=1)
        num_outlier_samples_train = np.sum(outlier_samples_mask_train)

        if num_outlier_samples_train > 0:
            # Use the inverted mask (~) to keep only non-outlier samples
            X_train = X_train[~outlier_samples_mask_train]
            Y_train = Y_train[~outlier_samples_mask_train]
            print(f"✅ Removed {num_outlier_samples_train} samples from the training set.")
        else:
            print("No outlier samples found to remove from the training set.")

        # Identify and remove outlier samples from the validation set
        outlier_samples_mask_val = np.any(outliers_mask_val, axis=1)
        num_outlier_samples_val = np.sum(outlier_samples_mask_val)

        if num_outlier_samples_val > 0:
            X_val = X_val[~outlier_samples_mask_val]
            Y_val = Y_val[~outlier_samples_mask_val]
            print(f"✅ Removed {num_outlier_samples_val} samples from the validation set.")
        else:
            print("No outlier samples found to remove from the validation set.")
        print("--------------------------------")
    # --- END Outlier Removal ---

    print(f"\nData split into {len(X_train)} training samples and {len(X_val)} validation samples.")

    X_train_tensor = torch.from_numpy(X_train)
    Y_train_tensor = torch.from_numpy(Y_train)
    X_val_tensor = torch.from_numpy(X_val)
    Y_val_tensor = torch.from_numpy(Y_val)

    train_dataset = RallyDataset(X_train_tensor, Y_train_tensor)
    val_dataset = RallyDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print("--- Data Preparation Complete ---")
    return train_loader, val_loader, mean, std