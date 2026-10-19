import os, yaml, subprocess, json, argparse, shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Active Dimer Sequences (PDB: 9OB3)
CDK2_SEQ = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
CYCLIN_E1_SEQ = "GSIIAPSRGSPLPVLSWANREEVWKIMLNKEKTYLRDQHFLEQHPLLQPKMRAILLDWLMEVCEVYKLHRETFYLAQDFFDRYMATQENVVKTLLQLIGISSLFIAAKLEEIYPPKLHQFAYVTDGACSGDEILTMELMIMKALKWRLSPLTIVSWLNVYMQVAYLNDLHEVLLPQYPQQIFIQIAELLDLCVLDVDCLEFPYGILAASALYHFSSSELMQKVSGYQWCDIENCVKWMVPFAMVIRETGSSKLKHFRGVADEDAHNIQTHRDSLDLLDKARAKKA"

# Triplicate Seeds for Uncertainty Quantification
SEEDS = [42, 1337, 2026]

def parse_single_seed(seed_dir):
    """Parses JSON confidence and NPZ PAE matrix for a single seed run."""
    res = {
        "aff_log": None, "prob": None, "plddt": None, 
        "iptm_complex": None, "min_ipae": None, "iptm_cdk2_lig": None
    }
    
    # 1. Parse JSON metrics
    for jf in Path(seed_dir).rglob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
                if "affinity_pred_value" in data:
                    res["aff_log"] = float(data["affinity_pred_value"]) # log10(uM)
                    res["prob"] = float(data.get("affinity_probability_binary"))
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

def compute_summary_statistics(seed_results):
    """Computes Mean and Sample Standard Deviation across the 3 seeds."""
    stats = {}
    
    def get_mean_std(vals):
        valid = [v for v in vals if v is not None]
        if not valid:
            return None, None
        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        return mean, std

    # pIC50 = -log10(IC50 in M) = 6 - log10(IC50 in uM)
    pic50_vals = [(6.0 - r["aff_log"]) for r in seed_results if r["aff_log"] is not None]
    stats["boltz_pIC50_mean"], stats["boltz_pIC50_std"] = get_mean_std(pic50_vals)
    
    # Confidence and Interface metrics
    stats["boltz_prob_mean"], stats["boltz_prob_std"] = get_mean_std([r["prob"] for r in seed_results])
    stats["plddt_mean"], stats["plddt_std"] = get_mean_std([r["plddt"] for r in seed_results])
    stats["iptm_complex_mean"], stats["iptm_complex_std"] = get_mean_std([r["iptm_complex"] for r in seed_results])
    stats["iptm_cdk2_lig_mean"], stats["iptm_cdk2_lig_std"] = get_mean_std([r["iptm_cdk2_lig"] for r in seed_results])
    stats["min_ipae_cdk2_mean"], stats["min_ipae_cdk2_std"] = get_mean_std([r["min_ipae"] for r in seed_results])
    
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ExperimentalData_57Ligands_IC50.txt")
    parser.add_argument("--outdir", default="results/results_ExperimentalData_3Runs")
    args = parser.parse_args()

    out_base = Path(args.outdir).resolve()
    os.environ["BOLTZ_CACHE_PATH"] = str(out_base / "boltz_cache")
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(args.input)
    csv_path = out_base / "summary_ExperimentalData_3Runs.csv"
    
    # 2. CHECKPOINT LOGIC
    results = []
    processed_ids = set()
    
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            results = existing_df.to_dict('records')
            processed_ids = set(existing_df['expID'].astype(str).tolist())
            print(f"Found checkpoint. Skipping {len(processed_ids)} previously completed ligands.")
        except Exception:
            print("Starting fresh.")

    # Master MSA Cache donor setup
    master_id = str(df.iloc[0]['expID'])
    master_msa_dir = out_base / "batch_runs" / master_id / f"seed_{SEEDS[0]}" / f"boltz_results_{master_id}" / "msa" / f"{master_id}_unpaired_tmp_env"

    # 3. Execution Loop
    for idx, row in df.iterrows():
        job_id = str(row['expID'])
        smiles = row['Product_SMILES']
        
        if job_id in processed_ids:
            continue
            
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
        out_base.joinpath("batch_runs").mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(input_data, f)

        seed_results = []
        for s in SEEDS:
            seed_dir = job_dir / f"seed_{s}"
            
            # ==========================================
            # LOCAL MSA CACHE "BAIT AND SWITCH"
            # ==========================================
            if not (job_id == master_id and s == SEEDS[0]) and master_msa_dir.exists():
                target_msa_dir = seed_dir / f"boltz_results_{job_id}" / "msa" / f"{job_id}_unpaired_tmp_env"
                target_msa_dir.mkdir(parents=True, exist_ok=True)
                
                for a3m_file in ["bfd.mgnify30.metaeuk30.smag30.a3m", "uniref.a3m"]:
                    src = master_msa_dir / a3m_file
                    dst = target_msa_dir / a3m_file
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)
            # ==========================================

            print(f"--- Predicting {job_id} ({idx+1}/{len(df)}) | Seed {s} ---")
            
            # Execute Boltz for this seed
            subprocess.run([
                "boltz", "predict", str(yaml_path),
                "--out_dir", str(seed_dir),
                "--use_msa_server",
                "--use_potentials",
                "--seed", str(s)
            ])
            
            single_res = parse_single_seed(seed_dir)
            seed_results.append(single_res)

        # Compute combined publication metrics
        summary_stats = compute_summary_statistics(seed_results)
        
        entry = row.to_dict()
        entry.update(summary_stats)
        results.append(entry)

        # LIVE SAVE
        pd.DataFrame(results).to_csv(csv_path, index=False)
        if yaml_path.exists():
            yaml_path.unlink()

    print(f"Job complete! Processed {len(df) - len(processed_ids)} new ligands.")

if __name__ == "__main__":
    main()