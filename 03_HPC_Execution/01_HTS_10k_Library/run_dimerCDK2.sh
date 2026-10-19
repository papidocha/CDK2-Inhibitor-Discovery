#!/bin/bash
#SBATCH --job-name=HTS_DIMERCDK2_10k
#SBATCH --array=0-31
#SBATCH --output=logs/hts_dimerCDK2/dimerCDK2_%a.out
#SBATCH --gpus-per-node=l40s:1      
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# Submit from this folder: sbatch run_dimerCDK2.sh
mkdir -p results/results_hts_dimerCDK2

module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6 cuda/12.2
source ../env/bin/activate || { echo "[ERROR] ../env not found. Run 'bash setup_boltz_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# ==========================================
# PATHFINDER LOGIC
# ==========================================
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "[START] Task 0: Acting as Pathfinder to populate Dimer CDK2 MSA cache..."
else
    # Tasks 1-31 wait 5 minutes (300s) + a slight stagger to avoid file-read collisions
    DELAY=$(( 300 + (SLURM_ARRAY_TASK_ID * 5) ))
    echo "Task $SLURM_ARRAY_TASK_ID: Waiting ${DELAY}s for Task 0 to build MSA cache..."
    sleep $DELAY
fi

# Run the Python script
python run_hts_dimerCDK2.py \
    --input data/THQ_virtual_library_10000_seed42.csv \
    --outdir results/results_hts_dimerCDK2 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks 32