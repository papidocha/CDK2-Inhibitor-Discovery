import os, yaml, subprocess, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Active Dimer Sequences (PDB: 9OB3)
CDK2_SEQ = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
CYCLIN_E1_SEQ = "GSIIAPSRGSPLPVLSWANREEVWKIMLNKEKTYLRDQHFLEQHPLLQPKMRAILLDWLMEVCEVYKLHRETFYLAQDFFDRYMATQENVVKTLLQLIGISSLFIAAKLEEIYPPKLHQFAYVTDGACSGDEILTMELMIMKALKWRLSPLTIVSWLNVYMQVAYLNDLHEVLLPQYPQQIFIQIAELLDLCVLDVDCLEFPYGILAASALYHFSSSELMQKVSGYQWCDIENCVKWMVPFAMVIRETGSSKLKHFRGVADEDAHNIQTHRDSLDLLDKARAKKA"

def calculate_focused_metrics(result_dir):
    """Calculates min-iPAE and Isolated ipTM specifically for CDK2-Ligand."""
    npz_files = list(Path(result_dir).rglob("*.npz"))
    if not npz_files:
        return None, None
    
    try:
        data = np.load(npz_files[0])
        pae = data['pae']
        
        # Dynamic Token Mapping based on string lengths
        cdk2_end = len(CDK2_SEQ)
        complex_prot_end = len(CDK2_SEQ) + len(CYCLIN_E1_SEQ)
        
        # Slice: Rows (CDK2) x Cols (Ligand)
        cdk2_lig_block = pae[:cdk2_end, complex_prot_end:]
        
        if cdk2_lig_block.size == 0:
            return None, None
            
        # 1. min-iPAE (Local "Grip")
        min_ipae = float(np.min(cdk2_lig_block))
        
        # 2. Isolated ipTM (Local Interface Confidence)
        d0 = 8.0
        iptm_isolated = float(np.mean(1 / (1 + (cdk2_lig_block / d0)**2)))
        
        return min_ipae, iptm_isolated
    except Exception as e:
        print(f"Error parsing PAE matrix: {e}")
        return None, None

def extract_metrics(result_dir):
    metrics = {
        "pred_ic50_uM": None, 
        "binding_prob": None, 
        "plddt": None, 
        "iptm_cdk2_lig": None, 
        "min_ipae_cdk2": None
    }
    
    json_files = list(Path(result_dir).rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
                if "affinity_pred_value" in data:
                    metrics["pred_ic50_uM"] = 10 ** data["affinity_pred_value"]
                    metrics["binding_prob"] = data.get("affinity_probability_binary")
                if "complex_plddt" in data:
                    metrics["plddt"] = data.get("complex_plddt")
        except: continue
    
    min_pae, iso_iptm = calculate_focused_metrics(result_dir)
    metrics["min_ipae_cdk2"] = min_pae
    metrics["iptm_cdk2_lig"] = iso_iptm
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/VariationAniline.txt")
    parser.add_argument("--outdir", default="results/results_Aniline_Variation")
    args = parser.parse_args()

    out_base = Path(args.outdir).resolve()
    os.environ["BOLTZ_CACHE_PATH"] = str(out_base / "boltz_cache")
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    # Check if file has headers or is a raw SMILES list
    try:
        df = pd.read_csv(args.input)
        if 'Product_SMILES' not in df.columns:
            # Fallback if it's just a raw list of SMILES
            df = pd.read_csv(args.input, header=None, names=['Product_SMILES'])
            df['expID'] = [f"aniline2_{i:02d}" for i in range(len(df))]
    except:
        print("Error reading input file. Ensure it is a valid CSV or TXT list.")
        return

    csv_path = out_base / "summary_Aniline_Variation.csv"
    processed_ids = set()
    results = []

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        processed_ids = set(existing_df['expID'].astype(str).tolist())
        results = existing_df.to_dict('records')
        print(f"Checkpoint found. Skipping {len(processed_ids)} ligands.")

    for idx, row in df.iterrows():
        eid = str(row['expID'])
        smiles = row['Product_SMILES']
        if eid in processed_ids: continue
            
        job_id = f"job_{eid}"
        job_dir = out_base / "batch_runs" / job_id
        
        input_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": CDK2_SEQ, "modifications": [{"ccd": "TPO", "position": 160}]}},
                {"protein": {"id": "B", "sequence": CYCLIN_E1_SEQ}},
                {"ligand": {"id": "L", "smiles": smiles}}
            ],
            "properties": [{"affinity": {"binder": "L"}}]
        }
        
        yaml_path = out_base / f"{job_id}.yaml"
        with open(yaml_path, 'w') as f: yaml.dump(input_data, f)

        print(f"--- Running {eid} ({idx+1}/{len(df)}) ---")
        subprocess.run(["boltz", "predict", str(yaml_path), "--out_dir", str(job_dir), "--use_msa_server", "--use_potentials"])

        m = extract_metrics(job_dir)
        
        entry = row.to_dict()
        entry.update({
            "boltz_prob": m["binding_prob"],
            "boltz_ic50": m["pred_ic50_uM"],
            "plddt": m["plddt"],
            "iptm_cdk2_ligand": m["iptm_cdk2_lig"],
            "min_ipae_cdk2": m["min_ipae_cdk2"]
        })
        results.append(entry)

        pd.DataFrame(results).to_csv(csv_path, index=False)
        if yaml_path.exists(): yaml_path.unlink()

if __name__ == "__main__":
    main()