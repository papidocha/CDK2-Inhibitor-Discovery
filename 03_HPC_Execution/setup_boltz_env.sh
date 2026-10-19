#!/bin/bash

# =================================================================
# Boltz-2 Environment Setup Script (Compute Canada)
# Targets: Python 3.11.5, Boltz 2.2.1, RDKit 2024.09.6
# Verified for: NVIDIA L40S GPU Architecture
# =================================================================

ENV_NAME="env"

echo "[INFO] Initializing environment in: $(pwd)"

# 1. Load exact cluster modules
module purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6 cuda/12.2 cudnn/8.9.5.29

# 2. Create the virtual environment
if [ ! -d "$ENV_NAME" ]; then
    python -m venv $ENV_NAME
fi
source $ENV_NAME/bin/activate
pip install --upgrade pip

# 3. Link RDKit module
echo "$EBROOTRDKIT/lib/python3.11/site-packages" > ./$ENV_NAME/lib/python3.11/site-packages/rdkit.pth

# 4. Install Core Math & Hardware Engines (Using Cluster Wheels)
echo "[INFO] Installing cluster-optimized hardware engines..."
pip install --no-index \
    torch==2.11.0 \
    torchvision==0.26.0 \
    triton==3.6.0 \
    cuequivariance-ops-cu12==0.7.0 \
    cuequivariance-ops-torch-cu12==0.7.0

# 5. Install Logic & Boltz (Hybrid: Cluster + Internet)
echo "[INFO] Installing Boltz-2 and support libraries..."
pip install typing-extensions packaging platformdirs psutil
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1
pip install boltz==2.2.1 pytorch-lightning==2.5.0 pandas pyyaml

echo "------------------------------------------------"
echo "[SUCCESS] Boltz-2 Setup Complete!"
echo "Environment: $(python -c 'import sys; print(sys.prefix)')"
echo "Boltz Version: $(python -c "from importlib.metadata import version; print(version('boltz'))")"
echo "To start working, run: source $ENV_NAME/bin/activate"
echo "------------------------------------------------"