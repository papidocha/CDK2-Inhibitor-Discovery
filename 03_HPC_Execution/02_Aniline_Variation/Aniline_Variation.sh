#!/bin/bash
#SBATCH --job-name=Aniline_Var
#SBATCH --output=logs/Aniline_Var_%j.out
#SBATCH --gpus-per-node=l40s:1      
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00

# Submit from this folder: sbatch Aniline_Variation.sh
# Create directories
mkdir -p results/results_Aniline_Variation

# Load environment
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6 cuda/12.2
source ../env/bin/activate || { echo "[ERROR] ../env not found. Run 'bash setup_boltz_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# Run the job
python Aniline_Variation.py \
    --input data/VariationAniline.txt \
    --outdir results/results_Aniline_Variation