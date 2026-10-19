#!/bin/bash
# Setup script for CDK2 GNINA Workflow on Compute Canada (Killarney)
# Run from the 03_HPC_Execution root: bash setup_gnina_env.sh

# Exit on any error
set -e

echo "=== [1/3] Loading Host Stack Modules ==="
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2
module load python/3.12 rdkit/2024.03.5 openmm/8.2.0 openbabel/3.1.1 scipy-stack/2024a

echo "=== [2/3] Initializing Isolated Python Environment ==="
ENV_DIR="env_gnina"

if [ ! -d "$ENV_DIR" ]; then
    virtualenv --no-download "$ENV_DIR"
else
    echo "Virtual environment '$ENV_DIR' already exists. Activating..."
fi
source "$ENV_DIR/bin/activate"

# Update baseline pip tools and install pip dependencies
pip install --no-cache-dir --no-index pip -U
pip install --no-cache-dir mdanalysis molscrub "pdbfixer>=1.9"

echo "=== [3/3] Linking System OpenMM Sites ==="
# Inject path reference for standard OpenMM system paths into our local env
OPENMM_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/MPI/gcc12/openmpi4/openmm/8.2.0/lib/python3.12/site-packages"
mkdir -p "$ENV_DIR/lib/python3.12/site-packages"
echo "$OPENMM_PATH" > "$ENV_DIR/lib/python3.12/site-packages/openmm_link.pth"

echo "=== [VERIFICATION] Testing Stack Integrity ==="
python -c "
import openmm
from pdbfixer import PDBFixer
import rdkit
import pandas as pd
from molscrub import Scrub
print('\n---> ALL SYSTEMS GO: Environment successfully built and verified.')
"