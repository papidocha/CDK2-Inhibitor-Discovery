#!/bin/bash
#SBATCH --job-name=cdk2_128_triplicate
#SBATCH --output=logs_128/workflow_%j.out
#SBATCH --gpus-per-node=l40s:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00   # 3 seeds x 57 ligands x up to 10 conformers at exhaustiveness 128

# Submit from this folder: sbatch run_gnina_triplicate_128.sh
#
# Steps: prepare the receptor, build a conformer ensemble for each ligand,
# redock the co-crystallised ligand as a control, then dock the 57 synthesized
# ligands at exhaustiveness 128 under three random seeds. Each docking step is
# checkpointed on its output file, so the job can be resubmitted after a
# wall-time limit and will resume where it stopped.

set -euo pipefail

# 1. Directory Setup
mkdir -p receptor results_final_128 results_redock ligands ligands_confs

# 2. Module Environment
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 gnina/1.3.1
module load python/3.12 rdkit/2024.03.5 openmm/8.2.0 openbabel/3.1.1 scipy-stack/2024a
source ../env_gnina/bin/activate || { echo "[ERROR] ../env_gnina not found. Run 'bash setup_gnina_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# 3. Preparation
echo "Step 1: Preparing receptor (CDK2, chain A)..."
python scripts/prep_receptor.py

echo "Step 2: Generating ligand conformer ensembles..."
python scripts/prep_ligands.py

# 4. Docking Configuration
CENTER_X=39.668
CENTER_Y=25.978
CENTER_Z=-10.781
SIZE=20
RECEPTOR="receptor/receptor.pdbqt"
SEEDS=(42 1337 2026)

# 4. Redocking control: native ligand back into the prepared receptor
echo "Step 3: Redocking control (native ligand of 9OB3)..."
CRYSTAL="receptor/9OB3_crystal_ligand.sdf"
if [ -s "$CRYSTAL" ]; then
    for seed in "${SEEDS[@]}"; do
        out_file="results_redock/crystal_seed_${seed}.sdf.gz"
        if [ -s "$out_file" ]; then
            echo "  [SKIP] redock seed $seed already completed."
            continue
        fi
        gnina -r $RECEPTOR -l "$CRYSTAL" \
              --center_x $CENTER_X --center_y $CENTER_Y --center_z $CENTER_Z \
              --size_x $SIZE --size_y $SIZE --size_z $SIZE \
              --exhaustiveness 128 \
              --num_modes 9 \
              --seed "$seed" \
              --out "$out_file"
    done
    python scripts/redock_control.py | tee results_redock/redock_rmsd_summary.txt
else
    echo "  [WARNING] $CRYSTAL missing - redocking control skipped."
fi

# 5. Production docking
echo "Step 4: Triplicate GNINA production run at exhaustiveness 128..."

for lig in ligands_confs/*.sdf; do
    name=$(basename "$lig" .sdf)

    for seed in "${SEEDS[@]}"; do
        out_file="results_final_128/${name}_seed_${seed}_128_final.sdf.gz"

        # Checkpoint: skip if this specific seed output already exists
        if [ -s "$out_file" ]; then
            echo "  [SKIP] $name | Seed $seed already completed."
            continue
        fi

        echo "  --> Processing $name | Seed $seed at 128 exhaustiveness..."
        gnina -r $RECEPTOR -l "$lig" \
              --center_x $CENTER_X --center_y $CENTER_Y --center_z $CENTER_Z \
              --size_x $SIZE --size_y $SIZE --size_z $SIZE \
              --exhaustiveness 128 \
              --num_modes 9 \
              --seed "$seed" \
              --out "$out_file"
    done
done

echo "Workflow complete. Production results in results_final_128/, "
echo "redocking control in results_redock/redock_rmsd_summary.txt"
