#!/bin/bash
#SBATCH --job-name=ExpData_3Runs
#SBATCH --output=logs/ExperimentalData_3Runs_%j.out
#SBATCH --gpus-per-node=l40s:1      
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Submit from this folder: sbatch ExperimentalData_3Runs.sh
# Create directories
mkdir -p results/results_ExperimentalData_3Runs

# Load environment
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6 cuda/12.2
source ../env/bin/activate || { echo "[ERROR] ../env not found. Run 'bash setup_boltz_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# Run the experimental validation script across 3 seeds
python ExperimentalData_3Runs.py \
    --input data/ExperimentalData_57Ligands_IC50.txt \
    --outdir results/results_ExperimentalData_3Runs