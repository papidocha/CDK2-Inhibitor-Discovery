#!/bin/bash
#SBATCH --job-name=ChEMBL_3D_Align
#SBATCH --array=0-5                 
#SBATCH --output=logs/pancdk_3d_%a.out
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Submit from this folder: sbatch submit_job_pancdk.sh

# Load CPU-only modules (RDKit is CPU-bound for conformer gen & ShapeAlign)
module purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6

# Activate the environment built by ../setup_3d_env.sh
source ../chem_env/bin/activate || { echo "[ERROR] ../chem_env not found. Run 'bash setup_3d_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# ==========================================
# PATHFINDER LOGIC
# ==========================================
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "[START] Task 0: Acting as Pathfinder to download and cache ChEMBL Pan-CDK ligands..."
else
    # Give Task 0 two minutes to download the ChEMBL IC50 data so the API doesn't lock us out
    DELAY=$(( 120 + (SLURM_ARRAY_TASK_ID * 10) ))
    echo "Task $SLURM_ARRAY_TASK_ID: Waiting ${DELAY}s for Task 0 to build the ChEMBL cache..."
    sleep $DELAY
fi

# Run the Python script
echo "[INFO] Executing 3D Alignments for Task $SLURM_ARRAY_TASK_ID..."
python run_3d_alignment_pancdk.py \
    --input data/ExperimentalData_57Ligands_IC50.txt \
    --outdir data/outputs \
    --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks 6 || exit 1

echo "[SUCCESS] Task $SLURM_ARRAY_TASK_ID Complete!"