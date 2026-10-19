import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# ==========================================
# 0. Mute RDKit C++ Warnings
# ==========================================
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Global Configuration & SMIRKS Definitions
# ==========================================

# Standard vs Special Reaction definitions (Norbornene scaffold only)
SCAFFOLD_REACTIONS = {
    "Norbornene": {
        "std": "[cH,CH:1][#6:2][NH2:3].[c:4][CH:5]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:5]([C:4])[NH:3][c:2][c:1]1",
        "specials": [
            {
                "name": "amino_pyrazole_tautomer_1",
                "pattern": "[NH2]c1[nH]ncc1",
                "smirks": "[NH2:1][c:2]1[nH:3][n:4][c:5][c:6]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[n;H0:3]1[n;H0:4][c:5][c:6]4"
            },
            {
                "name": "amino_pyrazole_tautomer_2",
                "pattern": "[NH2]c1n[nH]cc1",
                "smirks": "[NH2:1][c:2]1[n:3][nH:4][c:5][c:6]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[n;H0:3]1[n;H0:4][c:5][c:6]4"
            },
            {
                "name": "amino_indazole_type_1",
                "pattern": "[NH2]c1[cH]c2nncc2cc1",
                "smirks": "[NH2:1][c:2]1[cH:3][c:4]2[nH:5][n:6][c:9][c:10]2[c:11][c:12]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:3]1[c:4]5[nH:5][n:6][c:9][c:10]5[c:11][c:12]4"
            },
            {
                "name": "amino_indazole_type_2",
                "pattern": "[NH2]c1[cH]cc2nncc2c1",
                "smirks": "[NH2:1][c:2]1[c:12][c:3][c:4]2[nH:5][n:6][c:9][c:10]2[cH:11]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:12][c:3][c:4]5[nH:5][n:6][c:9][c:10]5[c:11]14"
            },
            {
                "name": "amino_benzimidazole_tautomer_1",
                "pattern": "[NH2]c1ccc2nc[nH]c2[cH]1",
                "smirks": "[NH2:1][c:2]1[c:12][c:3][c:4]2[n:5][c:9][nH:6][c:10]2[cH:11]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:12][c:3][c:4]5[n:5][c:9][nH:6][c:10]5[c:11]14"
            },
            {
                "name": "amino_benzimidazole_tautomer_2",
                "pattern": "[NH2]c1ccc2[nH]cnc2[cH]1",
                "smirks": "[NH2:1][c:2]1[c:12][c:3][c:4]2[nH:5][c:9][n:6][c:10]2[cH:11]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:12][c:3][c:4]5[nH:5][c:9][n:6][c:10]5[c:11]14"
            },
            {
                "name": "amino_indole_type_1",
                "pattern": "[NH2]c1[cH]c2nccc2cc1",
                "smirks": "[NH2:1][c:2]1[cH:3][c:4]2[nH:5][c:6][c:9][c:10]2[c:11][c:12]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:3]1[c:4]5[nH:5][c:6][c:9][c:10]5[c:11][c:12]4"
            },
            {
                "name": "amino_indole_type_2",
                "pattern": "[NH2]c1[cH]cc2nccc2c1",
                "smirks": "[NH2:1][c:2]1[c:12][c:3][c:4]2[nH:5][c:6][c:9][c:10]2[cH:11]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[c:12][c:3][c:4]5[nH:5][c:6][c:9][c:10]5[c:11]14"
            },
            {
                "name": "meta_substituted_steric_control",
                "pattern": "[NH2]c1[cH]c(*)[cH][cH][cH]1",
                "smirks": "[NH2:1][c:2]1[cH:3][c:4][cH:5][cH:6][cH:9]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[cH:3][c:4][cH:5][cH:6][c:9]14"
            },
            {
                "name": "meta_para_substituted_steric_control",
                "pattern": "[NH2]c1[cH]c(*)c(*)[cH][cH]1",
                "smirks": "[NH2:1][c:2]1[cH:3][c:4][c:5][cH:6][cH:9]1.[c:7][CH:8]=O>>[H][C@]12[C@]([C@H]3CC[C@@H]2C3)([H])[C@H:8]([C:7])[NH:1][c:2]4[cH:3][c:4][c:5][cH:6][c:9]14"
            }
        ]
    }
}

# ==========================================
# 2. Virtual Enumeration Engine
# ==========================================

def initialize_reactions():
    """Pre-compiles the RDKit reaction objects and SMARTS patterns for speed."""
    rxn_objects = {}
    for scaffold, rules in SCAFFOLD_REACTIONS.items():
        compiled_specials = []
        for spec in rules["specials"]:
            compiled_specials.append({
                "name": spec["name"],
                "pat_mol": Chem.MolFromSmarts(spec["pattern"]),
                "rxn": AllChem.ReactionFromSmarts(spec["smirks"])
            })
            
        rxn_objects[scaffold] = {
            "std": AllChem.ReactionFromSmarts(rules["std"]),
            "specials": compiled_specials
        }
    return rxn_objects

def generate_balanced_povarov_sample(ani_df, ald_df, rxn_objects, target_size=10000, seed=42):
    """
    Generates a virtual library using a round-robin sampling approach to ensure 
    equal representation of starting materials up to the target_size.
    """
    all_results = []
    error_count = 0 

    # REPRODUCIBILITY: Fixed random_state ensures the same shuffle every time
    ani_shuffled = ani_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    ald_shuffled = ald_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # We will use pointers to cycle through the lists evenly
    ani_idx = 0
    ald_idx = 0
    attempts = 0
    max_attempts = target_size * 20 # Expanded safety break for large enumeration

    print(f"Sampling {target_size} products with balanced representation (Seed: {seed})...")

    while len(all_results) < target_size and attempts < max_attempts:
        attempts += 1

        # Get the current rows
        ani_row = ani_shuffled.iloc[ani_idx]
        ald_row = ald_shuffled.iloc[ald_idx]

        ani_mol = Chem.MolFromSmiles(ani_row['SMILES'])
        ald_mol = Chem.MolFromSmiles(ald_row['SMILES'])

        if ani_mol and ald_mol:
            for scaffold_name, rxns in rxn_objects.items():
                reaction_succeeded = False
                broken_mol = None
                
                # 1. Try ALL matching special rules first (Fallback loop)
                for spec in rxns["specials"]:
                    if ani_mol.HasSubstructMatch(spec["pat_mol"]):
                        products = spec["rxn"].RunReactants((ani_mol, ald_mol))
                        if products:
                            for prod_variant in products:
                                prod_mol = prod_variant[0]
                                try:
                                    Chem.SanitizeMol(prod_mol)
                                    prod_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
                                    
                                    all_results.append({
                                        'Scaffold': scaffold_name,
                                        'Product_SMILES': prod_smiles,
                                        'Aniline_SMILES': ani_row['SMILES'],
                                        'Aniline_CAS': ani_row['CASNum'],
                                        'Aldehyde_SMILES': ald_row['SMILES'],
                                        'Aldehyde_CAS': ald_row['CASNum']
                                    })
                                    reaction_succeeded = True
                                    break # Success for this rule variant
                                except:
                                    broken_mol = prod_mol 
                                    continue # Failed sanitization, try next variant
                        if reaction_succeeded:
                            break # Success overall, break special rules loop

                # 2. Fallback to standard reaction if no specials worked (or none matched)
                if not reaction_succeeded:
                    products = rxns["std"].RunReactants((ani_mol, ald_mol))
                    if products:
                        for prod_variant in products:
                            prod_mol = prod_variant[0]
                            try:
                                Chem.SanitizeMol(prod_mol)
                                prod_smiles = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
                                
                                all_results.append({
                                    'Scaffold': scaffold_name,
                                    'Product_SMILES': prod_smiles,
                                    'Aniline_SMILES': ani_row['SMILES'],
                                    'Aniline_CAS': ani_row['CASNum'],
                                    'Aldehyde_SMILES': ald_row['SMILES'],
                                    'Aldehyde_CAS': ald_row['CASNum']
                                })
                                reaction_succeeded = True
                                break
                            except:
                                broken_mol = prod_mol
                                continue

                # 3. Debug logging if completely failed
                if not reaction_succeeded:
                    if error_count < 10:
                        if broken_mol:
                            print(f"\n[Debug] Sanitization Failed!")
                            try:
                                broken_smi = Chem.MolToSmiles(broken_mol, sanitize=False)
                                print(f"  Broken Product: {broken_smi}")
                            except:
                                print("  Broken Product: (Could not render)")
                        else:
                            print(f"\n[Debug] No products generated (Blocked C-H or strict SMIRKS failure).")
                            
                        print(f"  Aniline : {ani_row['SMILES']}")
                        print(f"  Aldehyde: {ald_row['SMILES']}")
                        error_count += 1
                        
                    elif error_count == 10:
                        print(f"\n[Debug] Reached 10 error prints. Suppressing further errors to keep terminal clean...")
                        error_count += 1

        # Round Robin logic: Advance pointers
        ani_idx = (ani_idx + 1) % len(ani_shuffled)
        ald_idx = (ald_idx + 1) % len(ald_shuffled)

        # Shift offset when aniline list wraps around to ensure new pair combinations
        if ani_idx == 0:
            ald_idx = (ald_idx + 1) % len(ald_shuffled)

    return pd.DataFrame(all_results)

# ==========================================
# 3. Execution Block
# ==========================================

if __name__ == "__main__":
    # Resolve paths reliably based on the script's location
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # Path setup: Grabbing curated inputs from the sister directory
    ald_input_path = PROJECT_ROOT / "01_BB_Curation" / "results" / "aldehydes" / "curated_aldehydes_smiles_cas.csv"
    ani_input_path = PROJECT_ROOT / "01_BB_Curation" / "results" / "amines" / "curated_amines_smiles_cas.csv"
    
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Starting THQ Virtual Library Enumeration ---\n")
    
    if not ald_input_path.exists() or not ani_input_path.exists():
        print(f"[Error] Could not locate input files.")
        print(f"Expected Aldehydes: {ald_input_path}")
        print(f"Expected Amines: {ani_input_path}")
        sys.exit(1)
        
    try:
        # Load the Data
        print("Loading curated building blocks...")
        df_ald = pd.read_csv(ald_input_path)
        df_ani = pd.read_csv(ani_input_path)
        print(f"  -> Loaded {len(df_ald)} Aldehydes.")
        print(f"  -> Loaded {len(df_ani)} Amines.")
        
        # Setup Reactions
        rxn_objects = initialize_reactions()
        
        # Generate the virtual library (Set target_size to a high number for full coverage, or 10000 for a subset)
        MY_SEED = 42
        TARGET_SIZE = 10000 
        
        sampled_library_df = generate_balanced_povarov_sample(df_ani, df_ald, rxn_objects, target_size=TARGET_SIZE, seed=MY_SEED)
        
        if sampled_library_df.empty:
            print("!!! Warning: No products were generated.")
        else:
            print(f"\nGeneration Complete!")
            print(f"Total Successful Products: {len(sampled_library_df):,}")
            
            # Save the results
            output_file = output_dir / f"THQ_virtual_library_{TARGET_SIZE}_seed{MY_SEED}.csv"
            sampled_library_df.to_csv(output_file, index=False)
            print(f"Library successfully saved to:\n{output_file}")
            
    except Exception as e:
        print(f"[Error] Pipeline failed during execution: {e}")