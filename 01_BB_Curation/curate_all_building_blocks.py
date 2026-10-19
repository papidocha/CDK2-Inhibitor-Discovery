from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ==========================================
# 0. Global Configuration & Rule Dictionaries
# ==========================================
RDLogger.DisableLog('rdApp.*')

# General exclusions applied to ALL libraries
GENERAL_EXCLUSIONS = {
    "boron": "[B]",
    "silicon": "[Si]",
    "phosphorus": "[P]",
    "iodine": "[I]",
    "nitro": "[N+](=O)[O-]",
    "quinone": "C1(=O)C(=O)C=CC=C1",
    "catechol": "c1([OX2H])c([OX2H])cccc1",
    "thiol": "[SH]",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl]",
    "carboxylate": "[CX3](=O)[O-]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[S](=O)(=O)[O,OH]"
}

# Library-specific exclusions and their respective "rescue" indices
LIBRARY_CONFIG = {
    "aldehydes": {
        "file": "AM_aromatic_aldehydes_v2.xlsx",
        "exclusions": {
            "multi_aldehyde": "[CX3H1](=O).*[CX3H1](=O)",
            "ortho_hydroxy": "c1([CX3H1](=O))c([OX2H1])cccc1",
            "aniline_block": "c[NH2]"
        },
        "rescues": {
            "aniline_block": [71, 518, 1353, 1570, 2076]
        },
        "manual_drops": [
            2297, 79, 2353, 2158, 2068, 2061, 1815, 1082, 1080, 390, 838, 246, 241, 156, 77, 30, 2007, 1983, 
            1409, 1065, 558, 282, 146, 16, 58, 604, 1004, 1054, 1318, 2006, 2367, 4, 8, 11, 17, 47, 48, 74, 
            114, 115, 118, 138, 223, 262, 270, 284, 302, 321, 326, 361, 525, 527, 582, 598, 626, 632, 649, 
            650, 651, 742, 751, 768, 770, 772, 773, 775, 780, 792, 815, 839, 841, 842, 843, 846, 853, 854, 
            858, 939, 962, 979, 1024, 1025, 1026, 1032, 1061, 1067, 1072, 1076, 1089, 1092, 1093, 1094, 1097, 
            1098, 1108, 1111, 1118, 1130, 1132, 1149, 1168, 1175, 1198, 1209, 1223, 1265, 1269, 1273, 1278, 
            1284, 1302, 1306, 1323, 1329, 1350, 1367, 1379, 1392, 1395, 1404, 1411, 1428, 1432, 1433, 1438, 
            1447, 1468, 1470, 1471, 1474, 1475, 1480, 1485, 1490, 1492, 1508, 1513, 1514, 1526, 1527, 1528, 
            1531, 1542, 1543, 1586, 1602, 1609, 1618, 1620, 1643, 1644, 1675, 1687, 1712, 1746, 1796, 1798, 
            1804, 1805, 1817, 1819, 1887, 1947, 1964, 1974, 2003, 2023, 2036, 2037, 2049, 2052, 2066, 2095, 
            2105, 2106, 2154, 2184, 2194, 2239, 2261, 2327, 2340, 2341, 298, 1095, 1297, 2133, 2164, 2167, 182, 849, 851, 1642, 334
        ]
    },
    "anilines": {
        "file": "AM_anilines.xlsx",
        "exclusions": {
            "aldehyde": "[CX3H1]=O",
            "multi_amine": "[NH2].*[NH2]",
            "specific_diamine": "Nc1ccccc1[NH][CX4]"
        },
        "rescues": {
            "sulfonic_acid": [1311, 1400, 1412, 1452],  # Rescues from GENERAL_EXCLUSIONS
            "multi_amine": [
                72, 75, 140, 187, 191, 193, 311, 339, 345, 373, 390, 400, 444, 468, 491,
                536, 540, 625, 626, 636, 663, 698, 758, 809, 857, 892, 920, 925, 947,
                1020, 1042, 1059, 1060, 1145, 1180, 1190, 1201, 1208, 1211, 1217, 1220,
                1226, 1233, 1252, 1280, 1283, 1292, 1366, 1379, 1388, 1393, 1409, 1472,
                1587, 1794, 1888, 1895, 1959, 2068, 2099
            ]
        },
        "manual_drops": [1767, 268, 906, 949, 1121, 1205, 1314, 1439, 1499, 2046, 14, 1874, 2101, 1, 259, 640, 654, 1084, 1985, 1993, 1415, 1701, 413, 188]
    },
    "heteroaromatic_amines": {
        "file": "AM_heteroaromatic_amines_v2.xlsx",
        "exclusions": {
            "aldehyde": "[CX3H1]=O",
            "aliphatic_amine": "[NH2][CX4]",
            "pyridine": "n1ccccc1"
        },
        "smi_exclusions": {
            "risk_structures": ["NC1=NC=CN=C1", "NC1=NC=NC=C1", "NC1=NN=CC=C1"]
        },
        "rescues": {
            "risk_structures": [26, 69, 99, 107, 134, 357, 421, 498, 539, 565, 578, 806, 843, 887, 1021, 1160, 1252, 1458],
            "pyridine": [7, 173, 236, 369, 379, 380, 499, 539, 577, 580, 590, 685, 689, 843, 1085, 1164, 1372, 1373, 1416, 1446, 1527, 1668],
            "povarov_ortho": [379]
        },
        "manual_drops": [
            1328, 91, 408, 475, 511, 865, 1090, 1658, 1734, 1665, 10, 202, 213, 230, 345, 660,
            678, 835, 838, 907, 953, 968, 1020, 1317, 1379, 1438, 1573, 1704, 1696
        ]
    }
}

# ==========================================
# 1. Unified Engine Functions
# ==========================================
def apply_filters(df, mol_col, smarts_dict, smiles_dict=None, rescues_dict=None):
    """
    Vectorized filtering utilizing Pandas index set operations.
    Handles both SMARTS (general rules) and SMILES (specific risk structures).
    """
    rescues_dict = rescues_dict or {}
    drop_indices = set()

    # 1. SMARTS matching
    for rule_name, smarts in smarts_dict.items():
        pat = Chem.MolFromSmarts(smarts)
        if not pat: continue
        matches = set(df[df[mol_col].apply(lambda mol: mol.HasSubstructMatch(pat))].index)

        if rule_name in rescues_dict:
            matches -= set(rescues_dict[rule_name])

        drop_indices.update(matches)

    # 2. SMILES matching (Critical for RDKit aromaticity matching on explicit rings)
    if smiles_dict:
        for rule_name, smi_list in smiles_dict.items():
            pats = [Chem.MolFromSmiles(s) for s in smi_list if Chem.MolFromSmiles(s)]
            matches = set(df[df[mol_col].apply(lambda mol: any(mol.HasSubstructMatch(p) for p in pats))].index)

            if rule_name in rescues_dict:
                matches -= set(rescues_dict[rule_name])

            drop_indices.update(matches)

    return df.drop(index=drop_indices)

def remove_alkali_salts(df, mol_col):
    """Universal check to drop unwanted alkali salts of carboxylic acids."""
    alkali_pat = Chem.MolFromSmarts('[Li+,Na+,K+]')
    acid_pat = Chem.MolFromSmarts('[CX3](=O)[O]')
    mask = df[mol_col].apply(lambda x: x.HasSubstructMatch(acid_pat) and x.HasSubstructMatch(alkali_pat))
    return df[~mask].copy()

def requires_ortho_ch(mol):
    """Validation for Povarov readiness (Anilines)."""
    aniline_pat = Chem.MolFromSmarts('c[NH2]')
    for match in mol.GetSubstructMatches(aniline_pat):
        atom = mol.GetAtomWithIdx(match[0])
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic() and neighbor.GetAtomicNum() == 6 and neighbor.GetTotalNumHs() >= 1:
                return True
    return False

def run_pains_filter(df, mol_col):
    """Unified PAINS filter that drops hits. Hits are separated out."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog = FilterCatalog(params)

    is_pains = df[mol_col].apply(lambda x: catalog.HasMatch(x) if x else False)

    df_pains_dropped = df[is_pains].copy()
    df_clean = df[~is_pains].copy()

    return df_clean, df_pains_dropped

def filter_pharmacophore(df, mol_col, category):
    """Filters molecules requiring at least 1 extra HBA and 1 extra HBD aside from the main reacting group."""
    if category == "aldehydes":
        pat = Chem.MolFromSmarts('[a][CX3H1]=O')
        hbd_sub_per_match = 0
        hba_sub_per_match = 1
    elif category in ["anilines", "heteroaromatic_amines"]:
        pat = Chem.MolFromSmarts('[a][NH2]')
        hbd_sub_per_match = 1
        hba_sub_per_match = 1
    else:
        return df

    def is_target(mol):
        if not mol: return False

        total_hbd = rdMolDescriptors.CalcNumHBD(mol)
        total_hba = rdMolDescriptors.CalcNumHBA(mol)

        matches = mol.GetSubstructMatches(pat)
        num_matches = len(matches)

        adj_hbd = total_hbd - (num_matches * hbd_sub_per_match)
        adj_hba = total_hba - (num_matches * hba_sub_per_match)

        return adj_hbd >= 1 and adj_hba >= 1

    mask = df[mol_col].apply(is_target)
    return df[mask].copy()

def remove_chiral_carbons(df, mol_col):
    """Removes molecules that contain at least one stereogenic carbon (assigned or unassigned)."""
    def has_chiral_carbon(mol):
        if not mol: return False
        # includeUnassigned=True identifies any atom with 4 unique substituents (CIP rules)
        # even if there is no explicit @ or @@ in the SMILES.
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for idx, _ in chiral_centers:
            # Check if the chiral center is actually a carbon atom
            if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6:
                return True
        return False

    mask = df[mol_col].apply(has_chiral_carbon)
    return df[~mask].copy()

def filter_in_stock(df):
    """Removes any compounds where StockUS is marked as 'Inquiry'."""
    return df[df['StockUS'] != 'Inquiry'].copy()

# ==========================================
# 2. Master Curation Pipeline
# ==========================================
def process_library(category, input_path, config):
    # --- 1. Load Data ---
    cols = ["CASNum", "SMILES", "StockUS", "Price/100mg (USD)", "Price/1g (USD)"]
    df = pd.read_excel(input_path, usecols=cols)
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'ROMol_raw')
    df = df.dropna(subset=['ROMol_raw']).copy()

    # --- 2. Initial Filters (General + Class-Specific) ---
    df = apply_filters(df, 'ROMol_raw', GENERAL_EXCLUSIONS, rescues_dict=config.get("rescues"))
    df = remove_alkali_salts(df, 'ROMol_raw')
    df = apply_filters(df, 'ROMol_raw', config.get("exclusions", {}), config.get("smi_exclusions", {}), config.get("rescues"))

    # Specific Rule: Heteroaromatic Amines cannot have multiple anilines
    if category == "heteroaromatic_amines":
        aniline_pat = Chem.MolFromSmarts('c[NH2]')
        df = df[df['ROMol_raw'].apply(lambda x: len(x.GetSubstructMatches(aniline_pat)) <= 1)]

    # --- 3. Salt Stripping & Deduplication ---
    lfc = rdMolStandardize.LargestFragmentChooser()
    def strip_salt(mol):
        if not mol: return None
        return Chem.MolToSmiles(lfc.choose(mol), isomericSmiles=True)

    df['SMILES_clean'] = df['ROMol_raw'].apply(strip_salt)
    df = df.drop_duplicates(subset=['SMILES_clean'], keep='first').copy()
    df['ROMol_clean'] = df['SMILES_clean'].apply(Chem.MolFromSmiles)

    # --- 4. Class-Specific Structural Validation ---
    if category in ["anilines", "heteroaromatic_amines"]:
        # NEW LOGIC: Check for explicit rescues before applying the requires_ortho_ch filter
        povarov_rescues = set(config.get("rescues", {}).get("povarov_ortho", []))
        
        # Keep the molecule if its index is in the rescue list OR if it passes the ortho check naturally
        mask = df.apply(lambda row: (row.name in povarov_rescues) or requires_ortho_ch(row['ROMol_clean']), axis=1)
        df = df[mask].copy()

    if category == "heteroaromatic_amines":
        tautomer_map = {102: 'Cc1c[nH]c(=O)cc1N', 261: 'Nc1c[nH]cc(Br)c1=O', 800: 'Cc1[nH]c(=O)c(N)cc1Br', 850: 'Nc1c[nH]ccc1=O'}
        for idx, new_smiles in tautomer_map.items():
            if idx in df.index:
                df.at[idx, 'SMILES_clean'] = new_smiles
                df.at[idx, 'ROMol_clean'] = Chem.MolFromSmiles(new_smiles)

    # --- 5. Explicit Manual Drops ---
    # Isolate and format strictly the manual drops for exporting
    active_drops = [i for i in config.get("manual_drops", []) if i in df.index]
    df_manual_dropped = df.loc[active_drops].copy()
    df = df.drop(index=active_drops)

    df_manual_dropped = df_manual_dropped.drop(columns=['SMILES']).rename(columns={'SMILES_clean': 'SMILES'})
    df_dropped_formatted = df_manual_dropped[['SMILES', 'CASNum']].copy()

    # --- 6. PAINS, Pharmacophore, Stereochemistry, and Stock Filtering ---
    # Run PAINS to clean the main dataset (hits are discarded silently)
    df, _ = run_pains_filter(df, 'ROMol_clean')

    # Ensure compounds have at least 1 extra HBA and 1 extra HBD
    df = filter_pharmacophore(df, 'ROMol_clean', category)

    # Remove any compound that has a stereogenic carbon 
    df = remove_chiral_carbons(df, 'ROMol_clean')

    # Remove any compounds that are strictly "Inquiry" status
    df = filter_in_stock(df)

    # --- 7. Final Formatting ---
    df = df.drop(columns=['SMILES']).rename(columns={'SMILES_clean': 'SMILES'})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df_final = df[["SMILES", "CASNum", "StockUS", "Price/100mg (USD)", "Price/1g (USD)"]]

    return df_final, df_dropped_formatted

# ==========================================
# 3. Execution Block
# ==========================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    print("--- Starting Unified Building Block Curation Pipeline ---\n")

    # Define how specific libraries group into final output folders
    group_mapping = {
        "aldehydes": "aldehydes",
        "anilines": "amines",
        "heteroaromatic_amines": "amines"
    }

    # Dictionary to hold the dataframes before saving
    grouped_results = {
        "aldehydes": {"curated": [], "dropped": []},
        "amines": {"curated": [], "dropped": []}
    }

    # 1. Process all libraries and store them in memory
    for category, config in LIBRARY_CONFIG.items():
        input_path = SCRIPT_DIR / "data" / config["file"]

        if not input_path.exists():
            print(f"[Skip] Input file not found for {category} at: {input_path}")
            continue

        print(f"Processing {category.replace('_', ' ').title()}...")
        try:
            df_curated, df_dropped = process_library(category, input_path, config)

            # Assign to the correct super-group
            target_group = group_mapping[category]
            grouped_results[target_group]["curated"].append(df_curated)
            grouped_results[target_group]["dropped"].append(df_dropped)

        except Exception as e:
            print(f"[Error] Failed processing {category}: {e}\n")

    print("\n--- Saving Consolidated Outputs ---")

    # 2. Concatenate and save the grouped results
    for target_group, data in grouped_results.items():
        if not data["curated"]:
            continue

        output_dir = SCRIPT_DIR / "results" / target_group
        output_dir.mkdir(parents=True, exist_ok=True)

        # Combine the lists into single dataframes and drop any cross-library duplicates
        final_curated = pd.concat(data["curated"]).drop_duplicates(subset=['SMILES'])
        final_dropped = pd.concat(data["dropped"]).drop_duplicates(subset=['SMILES'])

        # Define file paths
        out_full = output_dir / f"curated_{target_group}_full.csv"
        out_stripped = output_dir / f"curated_{target_group}_smiles_cas.csv"
        out_dropped = output_dir / f"manually_dropped_{target_group}.csv"

        # Save to CSV (index=False removes the old Excel row numbers so they don't get messy from the merge)
        final_curated.to_csv(out_full, index=False)
        final_curated[["SMILES", "CASNum"]].to_csv(out_stripped, index=False)
        final_dropped.to_csv(out_dropped, index=False)

        print(f"[{target_group.upper()}]")
        print(f"  -> Saved full dataset ({len(final_curated)} mols): results/{target_group}/{out_full.name}")
        print(f"  -> Saved stripped dataset: results/{target_group}/{out_stripped.name}")
        print(f"  -> Saved manual exclusions ({len(final_dropped)} mols): results/{target_group}/{out_dropped.name}\n")

    print("Pipeline execution complete.")