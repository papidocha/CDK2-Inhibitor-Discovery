import json
import numpy as np
import pandas as pd
from pathlib import Path

# Active Dimer Sequences (PDB: 9OB3)
CDK2_SEQ = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
CYCLIN_E1_SEQ = "GSIIAPSRGSPLPVLSWANREEVWKIMLNKEKTYLRDQHFLEQHPLLQPKMRAILLDWLMEVCEVYKLHRETFYLAQDFFDRYMATQENVVKTLLQLIGISSLFIAAKLEEIYPPKLHQFAYVTDGACSGDEILTMELMIMKALKWRLSPLTIVSWLNVYMQVAYLNDLHEVLLPQYPQQIFIQIAELLDLCVLDVDCLEFPYGILAASALYHFSSSELMQKVSGYQWCDIENCVKWMVPFAMVIRETGSSKLKHFRGVADEDAHNIQTHRDSLDLLDKARAKKA"

SEEDS = [42, 1337, 2026]

def parse_single_seed(seed_dir):
    """Parses JSON confidence and NPZ PAE matrix for a single seed run."""
    res = {
        "boltz_pIC50": None, 
        "boltz_prob": None, 
        "plddt": None, 
        "iptm_complex": None, 
        "min_ipae": None, 
        "iptm_cdk2_lig": None
    }
    
    # 1. Parse JSON metrics
    for jf in Path(seed_dir).rglob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
                if "affinity_pred_value" in data:
                    res["boltz_pIC50"] = 6.0 - float(data["affinity_pred_value"]) # Convert log10(uM) to pIC50
                    res["boltz_prob"] = float(data.get("affinity_probability_binary"))
                if "complex_plddt" in data:
                    res["plddt"] = float(data.get("complex_plddt"))
                if "iptm" in data:
                    res["iptm_complex"] = float(data.get("iptm"))
        except Exception:
            continue

    # 2. Parse NPZ PAE metrics
    cdk2_end = len(CDK2_SEQ)
    complex_prot_end = len(CDK2_SEQ) + len(CYCLIN_E1_SEQ)
    
    for nf in Path(seed_dir).rglob("*.npz"):
        try:
            data = np.load(nf)
            pae = data['pae']
            cdk2_lig_block = pae[:cdk2_end, complex_prot_end:]
            if cdk2_lig_block.size > 0:
                res["min_ipae"] = float(np.min(cdk2_lig_block))
                d0 = 8.0
                res["iptm_cdk2_lig"] = float(np.mean(1 / (1 + (cdk2_lig_block / d0)**2)))
        except Exception:
            continue
            
    return res

def main():
    # Paths to the completed job's output directory
    input_csv = "data/ExperimentalData_57Ligands_IC50.txt"
    results_dir = Path("results/results_ExperimentalData_3Runs/batch_runs")
    output_csv = "results/results_ExperimentalData_3Runs/ExperimentalData_3Runs_WIDE.csv"
    
    if not results_dir.exists():
        print(f"Error: Could not find the results directory at {results_dir}")
        return

    # Load the original input dataframe
    df = pd.read_csv(input_csv)
    wide_results = []

    for idx, row in df.iterrows():
        job_id = str(row['expID'])
        job_dir = results_dir / job_id
        
        # Start with the original data for this ligand
        entry = row.to_dict()
        
        # If the job directory doesn't exist, it means this ligand wasn't processed
        if not job_dir.exists():
            continue
            
        # Extract data for each specific seed
        for s in SEEDS:
            seed_dir = job_dir / f"seed_{s}"
            seed_data = parse_single_seed(seed_dir)
            
            # Append the seed number to the column name (e.g., "boltz_pIC50_seed42")
            for metric_name, value in seed_data.items():
                entry[f"{metric_name}_seed{s}"] = value
                
        wide_results.append(entry)

    # Save the wide format CSV
    wide_df = pd.DataFrame(wide_results)
    wide_df.to_csv(output_csv, index=False)
    print(f"Successfully extracted raw data for {len(wide_df)} ligands.")
    print(f"Saved wide table to: {output_csv}")

if __name__ == "__main__":
    main()