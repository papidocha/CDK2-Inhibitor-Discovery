#!/bin/bash
# =================================================================
# CDK2/Cyclin E1 HTS Result Merger
# Targets: results/results_hts_dimerCDK2/
# =================================================================

# Run from this folder after all array tasks finish: bash merge_cdk2_results.sh
module load StdEnv/2023 python/3.11.5 rdkit/2024.09.6 cuda/12.2
source ../env/bin/activate || { echo "[ERROR] ../env not found. Run 'bash setup_boltz_env.sh' in the 03_HPC_Execution folder first."; exit 1; }

# 1. Define paths
VENV_PY="python"
RESULTS_DIR="results/results_hts_dimerCDK2"
OUTPUT_FILE="master_CDK2_HTS_results_10k.csv"

# 2. Check directory
if [ ! -d "$RESULTS_DIR" ]; then
    echo "[ERROR] $RESULTS_DIR not found."
    exit 1
fi

echo "[INFO] Harvesting CDK2 task results..."

# 3. Concatenate and Sort
$VENV_PY << END
import pandas as pd
from pathlib import Path
import os
import sys

results_path = Path("$RESULTS_DIR")
output_file = "$OUTPUT_FILE"

# Find task-specific CSVs
csv_files = list(results_path.glob("summary_dimer_task_*.csv"))

if not csv_files:
    print(f"[ERROR] No task files found in {results_path.absolute()}")
    sys.exit(1)

print(f"[INFO] Found {len(csv_files)} files. Merging...")

df_list = []
for f in sorted(csv_files):
    if os.path.getsize(f) > 0:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

if not df_list:
    print("[ERROR] No valid data found.")
    sys.exit(1)

# Combine and sort by ligand ID
master_df = pd.concat(df_list, ignore_index=True)
if 'id' in master_df.columns:
    master_df = master_df.sort_values(by='id')

master_df.to_csv(output_file, index=False)

print("------------------------------------------------")
print(f"[SUCCESS] {len(master_df)} CDK2 ligands merged.")
print(f"[INFO] File saved: {output_file}")
print("------------------------------------------------")
END