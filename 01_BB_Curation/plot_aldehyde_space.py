import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
CURATED_CSV = SCRIPT_DIR / "results" / "aldehydes" / "curated_aldehydes_smiles_cas.csv"
SELECTED1_TXT = SCRIPT_DIR / "data" / "Selected_Aldehydes_Round1.txt"
SELECTED2_TXT = SCRIPT_DIR / "data" / "Selected_Aldehydes_Round2.txt"

OUT_DIR = SCRIPT_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Helper: Canonicalize SMILES
# ==========================================
def get_canonical_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None

# ==========================================
# 2. Load and Deduplicate Data
# ==========================================
print("Loading SMILES data...")

df_curated = pd.read_csv(CURATED_CSV)
curated_smiles = set([s for s in df_curated['SMILES'].apply(get_canonical_smiles) if s])

if SELECTED1_TXT.exists():
    with open(SELECTED1_TXT, 'r') as f:
        raw_selected1 = [line.strip() for line in f if line.strip()]
    selected1_smiles = set([s for s in (get_canonical_smiles(x) for x in raw_selected1) if s])
else:
    print(f"Warning: {SELECTED1_TXT.name} not found.")
    selected1_smiles = set()

if SELECTED2_TXT.exists():
    with open(SELECTED2_TXT, 'r') as f:
        raw_selected2 = [line.strip() for line in f if line.strip()]
    selected2_smiles = set([s for s in (get_canonical_smiles(x) for x in raw_selected2) if s])
else:
    print(f"Warning: {SELECTED2_TXT.name} not found.")
    selected2_smiles = set()

# CRUCIAL FIX: Sort the set to guarantee deterministic input order for the t-SNE seed!
all_unique_smiles = sorted(list(curated_smiles.union(selected1_smiles).union(selected2_smiles)))

data = []
for smi in all_unique_smiles:
    if smi in selected2_smiles:
        data.append({'SMILES': smi, 'Category': 'Round 2 Selected'})
    elif smi in selected1_smiles:
        data.append({'SMILES': smi, 'Category': 'Round 1 Selected'})
    else:
        data.append({'SMILES': smi, 'Category': 'Curated Library'})

df_tsne = pd.DataFrame(data)
print(f"Total unique molecules for t-SNE: {len(df_tsne)}")

# ==========================================
# 3. Compute Morgan Fingerprints (ECFP4)
# ==========================================
print("Computing Morgan Fingerprints (radius=2, 2048 bits)...")
fps = []
valid_indices = []

mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

for idx, row in df_tsne.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol:
        arr = mfp_gen.GetFingerprintAsNumPy(mol)
        fps.append(arr)
        valid_indices.append(idx)

X = np.array(fps)
df_tsne = df_tsne.loc[valid_indices].copy()

# ==========================================
# 4. Dimensionality Reduction (t-SNE)
# ==========================================
print("Running t-SNE projection...")
# The fixed random_state gives a reproducible projection because the input matrix 'X' is always identical (SMILES are sorted above)
tsne = TSNE(n_components=2, perplexity=30, random_state=22, init='pca', max_iter=1000)
X_tsne = tsne.fit_transform(X)

df_tsne['tsne_1'] = X_tsne[:, 0]
df_tsne['tsne_2'] = X_tsne[:, 1]

def get_sort_key(category):
    if category == 'Round 2 Selected': return 2
    if category == 'Round 1 Selected': return 1
    return 0

df_tsne['sort_key'] = df_tsne['Category'].apply(get_sort_key)
df_tsne = df_tsne.sort_values('sort_key')

# ==========================================
# 5. Plotting Engine
# ==========================================
print("Generating plot...")

sns.reset_orig()

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 7

# Double-column width is 183 mm (approx 7.2 inches). 
# 45% width = 3.24 inches. Setting width to 3.24 inches with a square/slightly tall 1:0.9 aspect ratio.
fig, ax = plt.subplots(figsize=(3.24, 2), dpi=300)

# 5a. Plot Background (Curated Library)
bg_data = df_tsne[df_tsne['Category'] == 'Curated Library']
ax.scatter(
    bg_data['tsne_1'], 
    bg_data['tsne_2'], 
    c="#B4B4B4", 
    s=15, 
    alpha=0.8, 
    edgecolor="black", 
    linewidth=0.2,
    label=f'Curated Library (n={len(bg_data)})',
    marker='o'
)

# 5b. Plot Round 1 Selected (Red Circles)
sel1_data = df_tsne[df_tsne['Category'] == 'Round 1 Selected']
if not sel1_data.empty:
    ax.scatter(
        sel1_data['tsne_1'], 
        sel1_data['tsne_2'], 
        c='#D32F2F', 
        s=30, 
        alpha=0.9, 
        edgecolor='black', 
        linewidth=0.5,
        label=f'Round 1 Selected (n={len(sel1_data)})',
        marker='o'
    )

# 5c. Plot Round 2 Selected (Blue Diamonds)
sel2_data = df_tsne[df_tsne['Category'] == 'Round 2 Selected']
if not sel2_data.empty:
    ax.scatter(
        sel2_data['tsne_1'], 
        sel2_data['tsne_2'], 
        c='#1976D2', 
        s=30, 
        alpha=0.9, 
        edgecolor='black', 
        linewidth=0.5,
        label=f'Round 2 Selected (n={len(sel2_data)})',
        marker='D' 
    )

# 5d. Styling Axes & Legend to 7 pt Arial Standards
ax.set_xlabel('t-SNE Dimension 1', fontsize=7, labelpad=4)
ax.set_ylabel('t-SNE Dimension 2', fontsize=7, labelpad=4)

ax.set_xticks([])
ax.set_yticks([])

# Scaled legend with 6 pt text to fit cleanly inside a 45% width panel
ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best', fontsize=6, markerscale=0.7, borderpad=0.4)

# Thin, clean 0.5 pt axes borders
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color('black')

plt.tight_layout()

# Save as SVG ONLY
svg_path = OUT_DIR / 'Fig_Aldehyde_Chemical_Space_tSNE.svg'
plt.savefig(svg_path, format='svg', transparent=True, bbox_inches='tight')
plt.close(fig)

print(f"✅ SVG successfully saved to: {svg_path.resolve()}")