#!/bin/bash

# =================================================================
# RDKit 3D Similarity Environment Setup Script (Compute Canada)
# Targets: Python 3.11.5, RDKit 2024.09.6
# Verified for: Killarney Cluster (CPU-bound)
# =================================================================

ENV_NAME="chem_env"

echo "🚀 Initializing 3D Similarity environment in: $(pwd)"

# 1. Load exact cluster modules
echo "📦 Loading required cluster modules..."
module purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6

# 2. Create the virtual environment
if [ ! -d "$ENV_NAME" ]; then
    echo "🌱 Creating new virtual environment: $ENV_NAME..."
    python -m venv $ENV_NAME
else
    echo "♻️ Virtual environment '$ENV_NAME' already exists. Activating..."
fi

source $ENV_NAME/bin/activate

echo "⬆️ Upgrading pip..."
pip install --no-index --upgrade pip

# 3. Link RDKit module
echo "🔗 Linking cluster-optimized RDKit..."
echo "$EBROOTRDKIT/lib/python3.11/site-packages" > ./$ENV_NAME/lib/python3.11/site-packages/rdkit.pth

# 4. Install specific project dependencies
echo "⚙️ Installing pip dependencies..."
pip install chembl_webresource_client pandas numpy tqdm joblib

echo "------------------------------------------------"
echo "✅ RDKit 3D Setup Complete!"
echo "Environment path: $(python -c 'import sys; print(sys.prefix)')"
echo "RDKit Version: $(python -c 'import rdkit; print(rdkit.__version__)')"
echo "To start working, run: source $ENV_NAME/bin/activate"
echo "------------------------------------------------"