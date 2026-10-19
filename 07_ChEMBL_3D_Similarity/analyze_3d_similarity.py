import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from chembl_webresource_client.new_client import new_client
import sys
import re

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# File containing your original experimental data
EXP_DATA_FILE = DATA_DIR / "ExperimentalData_57Ligands_IC50.txt"
TARGET_HIT = 'SMP210b'

# Manuscript compound numbers (THQ) for the internal compound IDs
ID_MAP = pd.read_csv(SCRIPT_DIR.parent / "compound_id_map.csv").set_index("internal_id")["thq_id"].to_dict()
TARGET_LABEL = ID_MAP[TARGET_HIT]

# Figure Formatting
mm_to_inch = 1 / 25.4
FIG_WIDTH = 170 * mm_to_inch
FIG_HEIGHT = 140 * mm_to_inch # ~5.5 inches for a 2x2 grid aspect ratio

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'svg.fonttype': 'none' # <--- CRITICAL: Forces text to be editable in Illustrator!
})

# ==========================================
# Helper: Fetch Target CDK via ChEMBL API
# ==========================================
def fetch_cdk_targets(smiles):
    """Queries ChEMBL to find which CDKs a specific SMILES is active against."""
    try:
        mol_api = new_client.molecule
        act_api = new_client.activity
        tar_api = new_client.target
        
        # 1. Match SMILES to get ChEMBL ID
        mols = list(mol_api.filter(molecule_structures__canonical_smiles=smiles).only(['molecule_chembl_id']))
        if not mols: return "CDK (Not Found)"
        mol_id = mols[0]['molecule_chembl_id']
        
        # 2. Get active targets (IC50)
        acts = list(act_api.filter(molecule_chembl_id=mol_id, standard_type="IC50").only(['target_chembl_id']))
        t_ids = list(set([a['target_chembl_id'] for a in acts if 'target_chembl_id' in a]))
        
        # 3. Extract pure CDK numbers using regex
        cdks = []
        for tid in t_ids:
            t_info = tar_api.get(tid)
            name = t_info.get('pref_name', '').lower()
            
            if 'cyclin-dependent' in name or 'cdk' in name:
                matches = re.findall(r'(?:kinase[s]?|cdk)\s*([0-9]+)', name)
                for m in matches:
                    cdks.append(f"CDK{m}")
        
        cdks = list(set(cdks))
        cdks.sort()
        
        if len(cdks) > 4:
            return "Pan-CDK"
        elif len(cdks) > 0:
            return "/".join(cdks)
        else:
            return "Unknown CDK Target"
    except Exception as e:
        return "API Lookup Failed"

# ==========================================
# 1. Aggregate Output CSVs (0.0 - 2.0 Scale)
# ==========================================
print(f"Scanning for 3D alignment results in {OUTPUTS_DIR.name}/...")
csv_files = list(OUTPUTS_DIR.glob("*_pancdk_3d_results.csv"))

if not csv_files:
    print(f"[!] Error: No CSV files found in {OUTPUTS_DIR.resolve()}")
    sys.exit(1)

summary_data = []
all_distributions = {}

for f in csv_files:
    eid = f.name.split('_')[0]
    df = pd.read_csv(f)

    # Enforce the industry standard 0.0 - 2.0 scale
    df['combo_score'] = df['shape_score'] + df['color_score']
    all_distributions[eid] = df['combo_score'].values

    best_match = df.loc[df['combo_score'].idxmax()]

    summary_data.append({
        'expID': eid,
        'thq_id': ID_MAP[eid],
        'Max_Combo_Score': best_match['combo_score'],
        'Shape_Score': best_match['shape_score'],
        'Color_Score': best_match['color_score'],
        'Best_ChEMBL_SMILES': best_match['smiles']
    })

df_summary = pd.DataFrame(summary_data)
print(f"Successfully loaded and aggregated data for {len(df_summary)} THQ ligands.")
df_summary.to_csv(RESULTS_DIR / "Summary_ChEMBL_3D_Matches.csv", index=False)

# ==========================================
# 2. Generate Separate Top 5 Reference Grid
# ==========================================
print(f"\nGenerating Top 5 separate reference image for {TARGET_HIT}...")
csv_path = OUTPUTS_DIR / f"{TARGET_HIT}_pancdk_3d_results.csv"

if not csv_path.exists() or not EXP_DATA_FILE.exists():
    print(f"[!] Warning: Missing data to generate {TARGET_HIT} reference grid.")
else:
    df_exp = pd.read_csv(EXP_DATA_FILE, sep=",")
    query_matches = df_exp[df_exp['expID'] == TARGET_HIT]
    
    if len(query_matches) > 0:
        query_smiles = query_matches['Product_SMILES'].values[0]
        
        df_res = pd.read_csv(csv_path)
        df_res['combo_score'] = df_res['shape_score'] + df_res['color_score']
        top_5 = df_res.sort_values(by='combo_score', ascending=False).head(5)
        
        mols_to_draw = [Chem.MolFromSmiles(query_smiles)]
        legends = [f"Our Hit: {TARGET_LABEL}\n(Query Structure)"]
        
        for rank, (_, row) in enumerate(top_5.iterrows(), start=1):
            smi = row['smiles']
            mols_to_draw.append(Chem.MolFromSmiles(smi))
            
            print(f"  > Fetching target info for Match #{rank} from ChEMBL...")
            reported_targets = fetch_cdk_targets(smi)
            
            legends.append(f"#{rank} | {reported_targets}\n"
                           f"Combo: {row['combo_score']:.2f} (S: {row['shape_score']:.2f}, C: {row['color_score']:.2f})")
            
        draw_opts = Draw.rdMolDraw2D.MolDrawOptions()
        draw_opts.legendFontSize = 24 
        draw_opts.clearBackground = False
        
        img = Draw.MolsToGridImage(
            mols_to_draw,
            legends=legends,
            molsPerRow=3,
            subImgSize=(450, 400),
            drawOptions=draw_opts
        )
        ref_image_path = RESULTS_DIR / f"{TARGET_LABEL}_Top5_3D_Matches_Annotated.png"
        img.save(ref_image_path)
        print(f"✅ Saved cleanly annotated reference image to {ref_image_path.name}")

# ==========================================
# 3. Build the 4-Panel Multiplot (SVG)
# ==========================================
print("\nAssembling Multi-panel SI Figure...")
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Fix the exact dimensions and spacing without using tight_layout/tight bounding box
fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.12, hspace=0.45, wspace=0.35)

gs = fig.add_gridspec(2, 2)

def add_panel_letter(ax, letter):
    ax.text(-0.25, 1.05, letter, transform=ax.transAxes, 
            fontsize=8, fontweight='bold', va='bottom', ha='right')

# --- Panel A: Full Library Histogram ---
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df_summary['Max_Combo_Score'], bins=15, kde=True, color='#1f77b4', edgecolor='black', alpha=0.7, ax=ax1)
ax1.set_title('Full THQ Library Similarity', pad=8)
ax1.set_xlabel('Max TanimotoCombo Score (Max 2.0)')
ax1.set_ylabel('Compound Count')
ax1.set_xlim(0.4, 2.0)
sns.despine(ax=ax1)
add_panel_letter(ax1, 'A')

# --- Panel B: Shape vs Color Scatter ---
ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(data=df_summary, x='Shape_Score', y='Color_Score', color='#2ca02c', s=20, edgecolor='black', alpha=0.7, ax=ax2)

if TARGET_HIT in df_summary['expID'].values:
    hit_data = df_summary[df_summary['expID'] == TARGET_HIT]
    sns.scatterplot(data=hit_data, x='Shape_Score', y='Color_Score', color='#d62728', marker='*', s=80, edgecolor='black', ax=ax2, label=TARGET_LABEL)

ax2.set_title('Pharmacophore vs. Steric Contributions', pad=8)
ax2.set_xlabel('Shape Tanimoto (Max 1.0)')
ax2.set_ylabel('Color Tanimoto (Max 1.0)')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.legend(frameon=False, loc='upper left')
sns.despine(ax=ax2)
add_panel_letter(ax2, 'B')

# --- Panel C: Target Specific KDE Profiles ---
ax3 = fig.add_subplot(gs[1, 0])
if TARGET_HIT in all_distributions:
    scores = all_distributions[TARGET_HIT]
    max_score = df_summary[df_summary['expID'] == TARGET_HIT]['Max_Combo_Score'].values[0]
    sns.kdeplot(scores, label=f"{TARGET_LABEL} (Max: {max_score:.2f})", color='#d62728', fill=True, alpha=0.35, linewidth=1.2, ax=ax3)

ax3.set_title(f'Pan-CDK Similarity Landscape of {TARGET_LABEL}', pad=8)
ax3.set_xlabel('TanimotoCombo Score')
ax3.set_ylabel('Density')
ax3.set_xlim(0.0, 2.0)
ax3.legend(frameon=False, loc='upper right')
sns.despine(ax=ax3)
add_panel_letter(ax3, 'C')

# --- Panel D: Structure Drop Zone (Empty Plot) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off') # Hides all axes, borders, and ticks
ax4.set_title(f'Chemical Structure of {TARGET_LABEL}', pad=8)
# Subtle placeholder text that you can easily delete or draw over in Illustrator
ax4.text(0.5, 0.5, f'[ Insert ChemDraw of {TARGET_LABEL} Here ]',
         ha='center', va='center', color='gray', fontsize=7, transform=ax4.transAxes)
add_panel_letter(ax4, 'D')

# ==========================================
# 4. Save and Finish
# ==========================================
final_plot_path = RESULTS_DIR / "Fig_S2_ChEMBL_3D_Similarity_Multiplot.svg"

# REMOVED bbox_inches='tight' so the true 170mm canvas size is completely preserved
plt.savefig(final_plot_path, format='svg')
plt.close(fig)

print(f"✅ Pipeline complete. SI figure saved to {final_plot_path.resolve()}")