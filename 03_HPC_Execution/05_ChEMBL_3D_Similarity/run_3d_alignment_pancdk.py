import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeAlign
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from chembl_webresource_client.new_client import new_client

# Suppress minor RDKit logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# ==========================================
# 1. 3D Alignment Core Functions
# ==========================================
def generate_3d_with_hs(smiles, max_confs=15):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    mol = rdMolStandardize.FragmentParent(mol)
    mol = Chem.AddHs(mol)
    
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.5
        params.numThreads = 1  
        
        AllChem.EmbedMultipleConfs(mol, numConfs=max_confs, params=params)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1) 
        return mol
    except Exception:
        return None

def compute_combo_score(chembl_smiles, query_mol_3d, max_confs=15):
    chembl_mol_3d = generate_3d_with_hs(chembl_smiles, max_confs=max_confs)
    if not chembl_mol_3d or chembl_mol_3d.GetNumConformers() == 0:
        return {'smiles': chembl_smiles, 'combo_score': 0.0, 'shape_score': 0.0, 'color_score': 0.0}
        
    best_combo = 0.0
    best_shape = 0.0
    best_color = 0.0
    
    for conf in chembl_mol_3d.GetConformers():
        try:
            shape_score, color_score = rdShapeAlign.AlignMol(
                query_mol_3d,             
                chembl_mol_3d,            
                refConfId=0,              
                probeConfId=conf.GetId(), 
                useColors=True            
            )
            
            # Direct addition for standard 0.0 - 2.0 TanimotoCombo
            combo_score = shape_score + color_score
            
            if combo_score > best_combo:
                best_combo = combo_score
                best_shape = shape_score
                best_color = color_score
                
        except Exception:
            continue
            
    return {'smiles': chembl_smiles, 'combo_score': best_combo, 'shape_score': best_shape, 'color_score': best_color}


def main():
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ExperimentalData_57Ligands_IC50.txt")
    parser.add_argument("--outdir", default="data/outputs")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--total_tasks", type=int, default=6)
    args = parser.parse_args()

    out_base = Path(args.outdir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.input).parent
    
    # ==========================================
    # 2. Fetch or Load Pan-CDK Ligands (Cached)
    # ==========================================
    cache_file = data_dir / "pan_cdk_chembl_smiles.txt"
    
    if cache_file.exists():
        print(f"Task {args.task_id}: Loading Pan-CDK ligands from local cache...")
        with open(cache_file, 'r') as f:
            chembl_smiles_raw = [line.strip() for line in f if line.strip()]
        print(f"Task {args.task_id}: Successfully loaded {len(chembl_smiles_raw)} Pan-CDK ligands.")
    else:
        if args.task_id == 0:
            print("Task 0 (Pathfinder): Searching ChEMBL for CDK targets...")
            target_api = new_client.target
            targets = target_api.filter(pref_name__icontains='cyclin-dependent kinase')
            
            kinase_chembl_ids = [t['target_chembl_id'] for t in targets 
                                 if t['target_type'] in ['SINGLE PROTEIN', 'PROTEIN COMPLEX']]
            kinase_chembl_ids = list(set(kinase_chembl_ids))
            
            print(f"Task 0: Found {len(kinase_chembl_ids)} distinct CDK targets. Fetching active compounds...")
            activity_api = new_client.activity
            chembl_smiles_raw = []

            for tid in tqdm(kinase_chembl_ids, desc="Downloading Ligands"):
                try:
                    res = activity_api.filter(target_chembl_id=tid, pchembl_value__isnull=False, standard_type="IC50")
                    smiles = [x['canonical_smiles'] for x in res if x['canonical_smiles'] is not None]
                    chembl_smiles_raw.extend(smiles)
                except Exception:
                    continue

            chembl_smiles_raw = list(set(chembl_smiles_raw))
            print(f"Task 0: Caching {len(chembl_smiles_raw)} unique ligands to {cache_file.name}")
            
            with open(cache_file, 'w') as f:
                for sm in chembl_smiles_raw:
                    f.write(sm + '\n')
        else:
            print(f"[!] Task {args.task_id} Error: Cache file missing. Pathfinder logic failed.")
            sys.exit(1)

    # ==========================================
    # 3. Load Experimental Data & Slice for Task
    # ==========================================
    try:
        df = pd.read_csv(args.input, sep=",")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {args.input}")

    total_mols = len(df)
    task_size = (total_mols // args.total_tasks) + (1 if total_mols % args.total_tasks > 0 else 0)
    start, end = args.task_id * task_size, min((args.task_id + 1) * task_size, total_mols)
    task_df = df.iloc[start:end].copy()
    
    print(f"\n--- Task {args.task_id} Processing Indices {start} to {end-1} ({len(task_df)} molecules) ---")

    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    print(f"Distributing internal 3D alignments across {num_cores} local CPU cores...")

    # ==========================================
    # 4. Process Loop with Checkpointing
    # ==========================================
    for idx, row in task_df.iterrows():
        eid = row['expID']
        sm = row['Product_SMILES']
        
        output_file = out_base / f"{eid}_pancdk_3d_results.csv"
        if output_file.exists():
            print(f"\n[{eid}] Results exist at {output_file.name}. Skipping to save compute.")
            continue
            
        print(f"\n[{eid}] Embedding 3D query...")
        q_mol_3d = generate_3d_with_hs(sm, max_confs=1)
        
        if not q_mol_3d:
            print(f"[{eid}] WARNING: Failed to embed 3D coordinates. Skipping.")
            continue
            
        print(f"[{eid}] Executing Pan-CDK Shape/Color Alignment...")
        ligand_start_time = time.time()
        
        # Parallelize the ChEMBL database loop across the node's CPUs
        results = Parallel(n_jobs=num_cores, batch_size='auto')(
            delayed(compute_combo_score)(ref_sm, q_mol_3d, 15) 
            for ref_sm in tqdm(chembl_smiles_raw, desc=f"Aligning {eid}")
        )
        
        # Save immediately to prevent data loss
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        duration = (time.time() - ligand_start_time) / 60.0
        print(f"[{eid}] Successfully saved in {duration:.1f} minutes.")

if __name__ == "__main__":
    main()