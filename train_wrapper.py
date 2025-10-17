# train_wrapper.py
import os
import sys

# Forcer le chargement des DLLs PyTorch AVANT tout
torch_lib = os.path.join(sys.executable, '..', 'Lib', 'site-packages', 'torch', 'lib')
torch_lib = os.path.normpath(torch_lib)
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

# Importer torch en premier
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ ROCm: {torch.cuda.is_available()}")
print(f"✓ GPUs: {torch.cuda.device_count()}\n")

# Maintenant on peut importer le reste
from rallyrobopilot.ml.train_model import main

if __name__ == "__main__":
    main()