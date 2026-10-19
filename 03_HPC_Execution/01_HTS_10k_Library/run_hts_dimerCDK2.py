import os, yaml, subprocess, json, argparse, shutil, time
import numpy as np
import pandas as pd
from pathlib import Path

# Active Dimer Sequences (PDB: 9OB3)
CDK2_SEQ = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
CYCLIN_E1_SEQ = "GSIIAPSRGSPLPVLSWANREEVWKIMLNKEKTYLRDQHFLEQHPLLQPKMRAILLDWLMEVCEVYKLHRETFYLAQDFFDRYMATQENVVKTLLQLIGISSLFIAAKLEEIYPPKLHQFAYVTDGACSGDEILTMELMIMKALKWRLSPLTIVSWLNVYMQVAYLNDLHEVLLPQYPQQIFIQIAELLDLCVLDVDCLEFPYGILAASALYHFSSSELMQKVSGYQWCDIENCVKWMVPFAMVIRETGSSKLKHFRGVADEDAHNIQTHRDSLDLLDKARAKKA"

def calculate_focused_metrics(result_dir):
    """Calculates min-iPAE and Isolated ipTM specifically for CDK2-Ligand."""
    npz_files = list(Path(result_dir).rglob("*.npz"))
    if not npz_files: return None, None
    
    try:
        data = np.load(npz_files[0])
        pae = data['pae']
        
        # Dynamic Token Mapping based on string lengths
        cdk2_end = len(CDK2_SEQ)
        complex_prot_end = len(CDK2_SEQ) + len(CYCLIN_E1_SEQ)
        
        # Slice the matrix: Rows (CDK2) x Cols (Ligand)
        cdk2_lig_block = pae[:cdk2_end, complex_prot_end:]
        
        if cdk2_lig_block.size == 0: return None, None
            
        # 1. min-iPAE: The single strongest contact (lower is better)
        min_ipae = float(np.min(cdk2_lig_block))
        
        # 2. Isolated ipTM: Average interface confidence (higher is better)
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
        "iptm_complex": None,  
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
                if "iptm" in data:
                    metrics["iptm_complex"] = data.get("iptm")
        except: continue
    
    min_pae, iso_iptm = calculate_focused_metrics(result_dir)
    metrics["min_ipae_cdk2"] = min_pae
    metrics["iptm_cdk2_lig"] = iso_iptm
    
    return metrics

def main():
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/THQ_virtual_library_10000_seed42.csv")
    parser.add_argument("--outdir", default="results/results_hts_dimer")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--total_tasks", type=int, default=32)
    args = parser.parse_args()

    out_base = Path(args.outdir).resolve()
    
    # 1. Load Data
    df = pd.read_csv(args.input)
    if 'id' not in df.columns:
        df['id'] = [f"lig_{i:05d}" for i in range(len(df))]

    # 2. Slice for Task
    total_mols = len(df)
    task_size = (total_mols // args.total_tasks) + (1 if total_mols % args.total_tasks > 0 else 0)
    start, end = args.task_id * task_size, min((args.task_id + 1) * task_size, total_mols)
    task_df = df.iloc[start:end].copy()

    # 3. CHECKPOINT LOGIC
    csv_path = out_base / f"summary_dimer_task_{args.task_id}.csv"
    timing_csv_path = out_base / f"timing_ligands_task_{args.task_id}.csv"
    gpu_json_path = out_base / f"timing_gpu_summary_task_{args.task_id}.json"
    
    results = []
    timing_records = []
    processed_smiles = set()
    
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            results = existing_df.to_dict('records')
            processed_smiles = set(existing_df['Product_SMILES'].tolist())
            print(f"Task {args.task_id}: Found checkpoint. Skipping {len(processed_smiles)} molecules.")
            
            if timing_csv_path.exists():
                timing_records = pd.read_csv(timing_csv_path).to_dict('records')
        except:
            print(f"Task {args.task_id}: Starting fresh.")

    # 4. Loop
    for idx, row in task_df.iterrows():
        if row['Product_SMILES'] in processed_smiles:
            continue
            
        job_id = str(row['id'])
        job_dir = out_base / "batch_runs" / job_id
        
        # Boltz-2 Dimer Schema with PTM
        input_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": CDK2_SEQ, "modifications": [{"ccd": "TPO", "position": 160}]}},
                {"protein": {"id": "B", "sequence": CYCLIN_E1_SEQ}},
                {"ligand": {"id": "L", "smiles": row['Product_SMILES']}}
            ],
            "properties": [{"affinity": {"binder": "L"}}]
        }
        
        yaml_path = out_base / f"{job_id}.yaml"
        out_base.joinpath("batch_runs").mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f: yaml.dump(input_data, f)

        # ==========================================
        # LOCAL MSA CACHE "BAIT AND SWITCH"
        # ==========================================
        master_msa_dir = out_base / "batch_runs" / "lig_00000" / "boltz_results_lig_00000" / "msa" / "lig_00000_unpaired_tmp_env"
        
        if job_id != "lig_00000":
            target_msa_dir = job_dir / f"boltz_results_{job_id}" / "msa" / f"{job_id}_unpaired_tmp_env"
            target_msa_dir.mkdir(parents=True, exist_ok=True)
            
            for a3m_file in ["bfd.mgnify30.metaeuk30.smag30.a3m", "uniref.a3m"]:
                src = master_msa_dir / a3m_file
                dst = target_msa_dir / a3m_file
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        # ==========================================

        # TIME TRACKING: Start
        ligand_start_time = time.time()
        
        # Execute
        subprocess.run(["boltz", "predict", str(yaml_path), "--out_dir", str(job_dir), "--use_msa_server", "--use_potentials"])

        # TIME TRACKING: End
        ligand_duration_mins = (time.time() - ligand_start_time) / 60.0
        timing_records.append({
            "job_id": job_id, 
            "Product_SMILES": row['Product_SMILES'], 
            "minutes_per_ligand": ligand_duration_mins
        })

        # Extract Metrics
        m = extract_metrics(job_dir)
        entry = row.to_dict()
        entry.update({
            "pred_ic50_uM": m["pred_ic50_uM"], 
            "binding_prob": m["binding_prob"], 
            "plddt": m["plddt"],
            "iptm_complex": m["iptm_complex"],
            "iptm_cdk2_lig": m["iptm_cdk2_lig"], 
            "min_ipae_cdk2": m["min_ipae_cdk2"]
        })
        results.append(entry)

        # LIVE SAVE (Metrics + Timings)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        pd.DataFrame(timing_records).to_csv(timing_csv_path, index=False)
        
        if yaml_path.exists(): yaml_path.unlink()

    # 5. OVERALL GPU TIME TRACKING
    script_duration_hours = (time.time() - script_start_time) / 3600.0
    avg_mins = sum(t["minutes_per_ligand"] for t in timing_records) / len(timing_records) if timing_records else 0
    
    gpu_summary = {
        "task_id": args.task_id,
        "total_ligands_processed": len(timing_records),
        "overall_duration_hours": script_duration_hours,
        "average_minutes_per_ligand": avg_mins
    }
    
    with open(gpu_json_path, 'w') as f:
        json.dump(gpu_summary, f, indent=4)

if __name__ == "__main__":
    main()