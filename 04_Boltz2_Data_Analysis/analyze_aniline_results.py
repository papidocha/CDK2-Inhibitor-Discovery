import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data" / "summary_Aniline_Variation.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CRUCIAL FONT & VECTOR SETTINGS
# ---------------------------------------------------------
# Reset seaborn FIRST so it doesn't hijack our publication settings
sns.reset_orig() 

mpl.rcParams['svg.fonttype'] = 'none'  # Keep SVG text completely editable!
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.5  
# ---------------------------------------------------------

# ==========================================
# 1. Load Data and Construct Matrix
# ==========================================
print(f"Loading data from {DATA_FILE.name}...")
df = pd.read_csv(DATA_FILE)

df['Aniline_ID'] = [f"AM4.{i}" for i in (df.index // 4) + 1]
df['Aldehyde_ID'] = [f"AL4.{i}" for i in (df.index % 4) + 1]

pivot_prob = df.pivot(index="Aldehyde_ID", columns="Aniline_ID", values="boltz_prob")

# ==========================================
# 2. Plotting Engine
# ==========================================
def render_heatmap(data_matrix, cbar_label, filename, vmin=None, vmax=None, cmap_override="YlGnBu"):
    fig, ax = plt.subplots(figsize=(4.2, 1.9), dpi=300)
    
    ax.set_facecolor('#f0f0f0')

    sns.heatmap(
        data_matrix,
        cmap=cmap_override,
        vmin=vmin, 
        vmax=vmax,
        annot=True,
        annot_kws={"size": 6},  
        fmt=".2f",
        linewidths=0.5,         
        linecolor='white',
        cbar_kws={'label': cbar_label},
        ax=ax,
        mask=data_matrix.isna()
    )

    ax.xaxis.tick_top()
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Tick labels automatically use 7pt from rcParams, just remove rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Ensure tick marks on axes and the colorbar match the 0.5pt spec
    ax.tick_params(axis='both', which='both', length=2, width=0.5, pad=2)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout()
    out_path = RESULTS_DIR / filename
    
    # Save directly as SVG with transparency
    plt.savefig(out_path, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path.name}")

# ==========================================
# 3. Generate Visualization
# ==========================================
print("Generating matrix...")

render_heatmap(
    data_matrix=pivot_prob,
    cbar_label='Boltz-2 Probability',
    filename='Fig_SAR_BoltzProb.svg', # Changed extension to SVG
    vmin=0.0,
    vmax=1.0,
    cmap_override="YlGnBu_r"
)

# ==========================================
# 4. Summary Statistics
# ==========================================
print("\n" + "="*30)
print("     SUMMARY STATISTICS     ")
print("="*30)
if 'boltz_prob' in df.columns:
    print(f"Global Prob Mean: {df['boltz_prob'].mean():.3f}")
print("="*30)