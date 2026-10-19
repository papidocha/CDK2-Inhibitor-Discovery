import math
import sys
import pandas as pd
from pathlib import Path
from rdkit import Chem

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

TXT_IN = SCRIPT_DIR / "ligands" / "ExperimentalData_57Ligands.txt"
CSV_IN = SCRIPT_DIR / "results" / "CDK2_estimated_IC50.csv"

# Full dose-response IC50s (Hill slope fixed at 1), written by plot_curves.py
DOSE_RESPONSE_TSV = SCRIPT_DIR / "results" / "ic50_results_hill1.tsv"

# Copies of the 57-ligand input read by the HPC jobs and the 3D similarity analysis
TXT_OUT = [
    REPO_ROOT / "03_HPC_Execution" / "03_Experimental_Triplicates" / "data" / "ExperimentalData_57Ligands_IC50.txt",
    REPO_ROOT / "03_HPC_Execution" / "04_GNINA_Docking" / "ligands" / "ExperimentalData_57Ligands_IC50.txt",
    REPO_ROOT / "03_HPC_Execution" / "05_ChEMBL_3D_Similarity" / "data" / "ExperimentalData_57Ligands_IC50.txt",
    REPO_ROOT / "07_ChEMBL_3D_Similarity" / "data" / "ExperimentalData_57Ligands_IC50.txt",
]

# ==========================================
# 1. SMILES Standardization Function
# ==========================================
def canonicalize_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    print(f"Warning: Could not parse SMILES -> {smiles_str}")
    return smiles_str

def round_2sigfigs(x):
    return round(x, 1 - math.floor(math.log10(abs(x))))

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print(f"Loading ligand data from {TXT_IN.name}...")
    df_ligands = pd.read_csv(TXT_IN)
    
    print(f"Loading estimated IC50s from {CSV_IN.name}...")
    df_ic50 = pd.read_csv(CSV_IN)

    # Merge the estimated data onto the 57-ligand template
    df_merged = pd.merge(
        df_ligands, 
        df_ic50[['Compound', 'Estimated_IC50_µM']], 
        left_on='expID', 
        right_on='Compound', 
        how='left'
    )
    
    # Standardize column name for the output
    df_merged.rename(columns={'Estimated_IC50_µM': 'expIC50'}, inplace=True)
    df_merged['expIC50'] = df_merged['expIC50'].astype(float)

    # ==========================================
    # 3. Replace Primary-Screen Estimates with Full Dose-Response IC50s
    # ==========================================
    if not DOSE_RESPONSE_TSV.exists():
        print(f"Error: {DOSE_RESPONSE_TSV.name} not found. Run plot_curves.py first.")
        sys.exit(1)

    print(f"Applying dose-response IC50s from {DOSE_RESPONSE_TSV.name}...")
    df_dr = pd.read_csv(DOSE_RESPONSE_TSV, sep="\t")

    for comp_id, ic50 in zip(df_dr['compound'], df_dr['reported_ic50']):
        val = round_2sigfigs(ic50)
        mask = df_merged['expID'] == comp_id
        if mask.any():
            df_merged.loc[mask, 'expIC50'] = val
            print(f"  -> {comp_id}: {val} µM")
        else:
            print(f"  -> Warning: {comp_id} not found in the input list.")

    # ==========================================
    # 4. Canonicalize SMILES
    # ==========================================
    print("Canonicalizing SMILES using RDKit...")
    df_merged['Product_SMILES'] = df_merged['Product_SMILES'].apply(canonicalize_smiles)

    # ==========================================
    # 5. Finalize and Export
    # ==========================================
    # Isolate only the required columns in the exact requested order
    final_cols = ['expID', 'Product_SMILES', 'expIC50']
    df_final = df_merged[final_cols]
    
    for out_path in TXT_OUT:
        df_final.to_csv(out_path, index=False)

    print("\n--- Output Summary ---")
    print(f"Total ligands processed: {len(df_final)}")
    print(f"Total missing IC50 values: {df_final['expIC50'].isna().sum()}")
    for out_path in TXT_OUT:
        print(f"File saved successfully to: {out_path.relative_to(REPO_ROOT)}")
    print("----------------------")

if __name__ == "__main__":
    main()