# train_wrapper.py
import os
import sys

# Laisser les imports et la configuration des DLLs ici, c'est OK.
torch_lib = os.path.join(sys.executable, '..', 'Lib', 'site-packages', 'torch', 'lib')
torch_lib = os.path.normpath(torch_lib)
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

import torch
from rallyrobopilot.ml.train_model import main

# La fonction principale qui ne sera appelée que par le script principal
def run_training():
    # --- DÉPLACER LES PRINTS ICI ---
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ ROCm: {torch.cuda.is_available()}")
    print(f"✓ GPUs: {torch.cuda.device_count()}\n")
    
    # Lancer l'entraînement
    main()

# Cette protection garantit que seul le script exécuté directement
# (et non les workers) appellera run_training()
if __name__ == "__main__":
    run_training()