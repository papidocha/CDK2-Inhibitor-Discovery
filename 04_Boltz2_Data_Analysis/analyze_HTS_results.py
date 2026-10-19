import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "data" / "master_CDK2_HTS_results_10k.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Column Configurations
BB1_SMILES_COL = 'Aniline_SMILES'  
BB1_CAS_COL = 'Aniline_CAS'
BB2_SMILES_COL = 'Aldehyde_SMILES'
BB2_CAS_COL = 'Aldehyde_CAS'
PROB_COL = 'binding_prob'

# Publication Styling Colors
HIT_COLOR = "#2C3E50"      # Slate blue
NON_HIT_COLOR = "#BDC3C7"  # Soft gray
PALETTE = {"Hit (>0.20)": HIT_COLOR, "Non-Hit": NON_HIT_COLOR}

# ---------------------------------------------------------
# Slide layout settings for the hit rate bar charts (all in mm)
# ---------------------------------------------------------
MM = 1 / 25.4                 # mm -> inch
SLIDE_W_MM = 338.67           # Standard 16:9 PowerPoint width
SLIDE_H_MM = 190.5            # Standard 16:9 PowerPoint height

BAR_MARGIN_LEFT_MM   = 25     # y-axis label + tick labels
BAR_MARGIN_RIGHT_MM  = 12
BAR_MARGIN_TOP_MM    = 35     # Room for a slide title + panel titles
BAR_MARGIN_BOTTOM_MM = 30     # Room for rotated x-tick labels
BAR_PANEL_GAP_MM     = 15     # Space between the two panels

# Larger fonts for projection (only applied to the slide figure)
SLIDE_RC = {
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'axes.linewidth': 0.75,
}

# ---------------------------------------------------------
# CRUCIAL FONT & VECTOR SETTINGS
# ---------------------------------------------------------
sns.reset_orig() # Reset seaborn FIRST to prevent it from overriding us

mpl.rcParams['svg.fonttype'] = 'none'  # Keep SVG text completely editable!
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 7   # matches the correlation figures in module 06
mpl.rcParams['axes.linewidth'] = 0.5  
# ---------------------------------------------------------

# ==========================================
# 1. Load & Prepare Data
# ==========================================
print(f"Loading HTS results from {INPUT_CSV.name}...")
df = pd.read_csv(INPUT_CSV)

print("Calculating Molecular Weights and LogP...")
def safe_calc(smiles, func):
    mol = Chem.MolFromSmiles(str(smiles))
    return func(mol) if mol else None

df['MW'] = df['Product_SMILES'].apply(lambda x: safe_calc(x, Descriptors.MolWt))
df['LogP'] = df['Product_SMILES'].apply(lambda x: safe_calc(x, Descriptors.MolLogP))

df['Hit_Status'] = df[PROB_COL].apply(lambda x: 'Hit (>0.20)' if x > 0.20 else 'Non-Hit')

hits_df = df[df[PROB_COL] > 0.20].sort_values(PROB_COL, ascending=False).copy()
print(f"Found {len(hits_df)} ligands with {PROB_COL} > 20%")

# ==========================================
# 2. Extract Top Building Blocks (By Hit Rate)
# ==========================================
def get_top_bbs_by_rate(df_full, smiles_col, cas_col, prefix, top_n=20):
    grouped = df_full.groupby([smiles_col, cas_col])
    
    counts = grouped.agg(
        Total_Count=(PROB_COL, 'count'),
        Hit_Count=(PROB_COL, lambda x: (x > 0.20).sum())
    ).reset_index()
    
    counts['Hit_Rate'] = (counts['Hit_Count'] / counts['Total_Count']) * 100
    counts = counts.sort_values(['Hit_Rate', 'Hit_Count'], ascending=[False, False]).head(top_n).reset_index(drop=True)
    counts['Plot_ID'] = [f"{prefix}{i+1}" for i in range(len(counts))]
    counts['Grid_Label'] = counts.apply(
        lambda row: f"Rate: {row['Hit_Rate']:.1f}% ({row['Hit_Count']}/{row['Total_Count']})", axis=1
    )
    return counts

# Adjusted prefixes to AM2. and AL2.
bb1_counts = get_top_bbs_by_rate(df, BB1_SMILES_COL, BB1_CAS_COL, "AM2.")
bb2_counts = get_top_bbs_by_rate(df, BB2_SMILES_COL, BB2_CAS_COL, "AL2.")

bb1_counts.to_csv(RESULTS_DIR / 'Top_Amines_Mapping.csv', index=False)
bb2_counts.to_csv(RESULTS_DIR / 'Top_Aldehydes_Mapping.csv', index=False)

# ==========================================
# 3. Model Validation & Physicochemical Plots
# ==========================================
print("Generating validation plots (SVGs)...")

# A. Binding Probability Density (standalone single-panel figure)
fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=300)
sns.kdeplot(data=df, x=PROB_COL, fill=True, color=NON_HIT_COLOR, alpha=0.6, linewidth=1, ax=ax)
ax.axvline(x=0.20, color=HIT_COLOR, linestyle='--', linewidth=1, label='Hit Threshold (0.20)')
ax.set_title('Distribution of Boltz-2 Binding Probabilities', pad=8)
ax.set_xlabel('Predicted Binding Probability')
ax.set_ylabel('Density')
ax.legend(frameon=False)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'Fig_Validation_Density.svg', format='svg', transparent=True,
            bbox_inches='tight')
plt.close()

# B. Physicochemical Space (double column width, matching the 06 correlation panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.75), dpi=300)

sns.violinplot(data=df, x='Hit_Status', y='MW', hue='Hit_Status', palette=PALETTE, 
               ax=ax1, inner="quartile", legend=False, linewidth=0.5)
ax1.set_title('Molecular Weight Distribution')
ax1.set_xlabel('')
ax1.set_ylabel('Molecular Weight (Da)')
sns.despine(ax=ax1)

sns.violinplot(data=df, x='Hit_Status', y='LogP', hue='Hit_Status', palette=PALETTE, 
               ax=ax2, inner="quartile", legend=False, linewidth=0.5)
ax2.set_title('LogP Distribution')
ax2.set_xlabel('')
ax2.set_ylabel('Calculated LogP')
sns.despine(ax=ax2)

# Panel letters, placed as in the correlation panels of module 06
for ax, panel in ((ax1, 'A'), (ax2, 'B')):
    ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'Fig_Validation_PhysChem.svg', format='svg', transparent=True,
            bbox_inches='tight')
plt.close()

# ==========================================
# 4. Building Block Hit Rate Bar Charts (16:9 PowerPoint slide, side by side)
# ==========================================
print("Generating hit rate bar charts (slide format)...")

global_max_y = min(105, max(bb1_counts['Hit_Rate'].max(), bb2_counts['Hit_Rate'].max()) * 1.15)
y_label_text = r'Ligands with $P_{Boltz} > 20$ (%)'

with mpl.rc_context(SLIDE_RC):
    # Figure = full slide
    fig = plt.figure(figsize=(SLIDE_W_MM * MM, SLIDE_H_MM * MM), dpi=300)
    fig.patch.set_alpha(0)  # Transparent slide background

    # Panel sizes computed from the margins
    panel_w = (SLIDE_W_MM - BAR_MARGIN_LEFT_MM - BAR_MARGIN_RIGHT_MM - BAR_PANEL_GAP_MM) / 2
    panel_h = SLIDE_H_MM - BAR_MARGIN_TOP_MM - BAR_MARGIN_BOTTOM_MM

    def panel_rect(left_mm):
        """Axes position as fractions of the slide."""
        return [left_mm / SLIDE_W_MM, BAR_MARGIN_BOTTOM_MM / SLIDE_H_MM,
                panel_w / SLIDE_W_MM, panel_h / SLIDE_H_MM]

    ax1 = fig.add_axes(panel_rect(BAR_MARGIN_LEFT_MM))
    ax2 = fig.add_axes(panel_rect(BAR_MARGIN_LEFT_MM + panel_w + BAR_PANEL_GAP_MM), sharey=ax1)

    panels = [
        (ax1, bb1_counts, 'Top 20 Amines by Hit Rate'),
        (ax2, bb2_counts, 'Top 20 Aldehydes by Hit Rate'),
    ]
    for ax, data, title in panels:
        sns.barplot(data=data, x='Plot_ID', y='Hit_Rate', color=HIT_COLOR, ax=ax,
                    edgecolor='black', linewidth=0.75)
        ax.set_ylim(0, global_max_y)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=90)
        sns.despine(ax=ax)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    # Shared y-axis: label only on the left panel
    ax1.set_ylabel(y_label_text)
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', labelleft=False)

    # No tight_layout / bbox_inches='tight' -> SVG stays exactly slide-sized
    plt.savefig(RESULTS_DIR / 'Fig_BB_Hit_Rates.svg', format='svg')
    plt.close()

# ==========================================
# 5. Molecular Grids (SVGs sized for a single A4 page)
# ==========================================
# Structures are drawn to the ACS/ChemDraw document standard: 0.508 cm bonds and
# 10 pt atom labels. The grids are emitted with their physical size in mm, so
# they drop onto an A4 page at 1:1 with no rescaling (which would change the
# bond length). A4 is 210 x 297 mm; with 20 mm margins that leaves 170 x 257 mm,
# and each grid is kept well inside the height to leave room for a caption.
ACS_BOND_MM = 5.08          # ChemDraw "ACS Document 1996" chain bond length
ACS_ATOM_PT = 10            # ChemDraw atom-label size at that bond length
LEGEND_PT = 7               # caption-sized text under each structure
PX_PER_MM = 8               # internal drawing resolution
PAGE_WIDTH_MM = 170         # A4 minus 20 mm margins

_PT_MM = 25.4 / 72


def draw_mol_grid_svg(df_mols, smiles_col, label_col_1, label_col_2, filename,
                      per_row, n_max, cell_mm, prefix="", ic50_col=None):
    """Write a molecule grid whose SVG carries its true size in millimetres."""
    mols, legends = [], []
    for i, row in df_mols.head(n_max).iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if not mol:
            continue
        mols.append(mol)
        if prefix:
            legends.append(f"{prefix}{i+1} | {row[label_col_1]}\nCAS: {row[label_col_2]}")
        else:
            # RDKit renders legends as vector paths with a font that has no "µ",
            # so the unit is given in the caption rather than on every panel.
            ic50_val = row[ic50_col] if ic50_col else float("nan")
            legends.append(f"{row[label_col_1]}\n"
                           f"P = {row[label_col_2]:.2f} | IC50 = {ic50_val:.2f}")
    if not mols:
        return

    cell_w_mm, cell_h_mm = cell_mm
    cell_px = (round(cell_w_mm * PX_PER_MM), round(cell_h_mm * PX_PER_MM))

    dopts = Draw.MolDrawOptions()
    dopts.fixedBondLength = ACS_BOND_MM * PX_PER_MM
    dopts.fixedFontSize = round(ACS_ATOM_PT * _PT_MM * PX_PER_MM)
    legend_px = round(LEGEND_PT * _PT_MM * PX_PER_MM)
    dopts.legendFontSize = legend_px
    # RDKit allots the legend only 10% of the panel height by default and shrinks
    # the font to fit, which would render the same 7 pt text at different sizes in
    # grids with different cell heights. Give it exactly the room two lines need.
    dopts.legendFraction = min(0.30, 2.4 * legend_px / cell_px[1])
    # Stereochemistry is conveyed by the wedge/hash bonds; the (R)/(S) labels
    # would add five annotations per product and clutter the grid.
    dopts.addStereoAnnotation = False

    svg = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=cell_px,
                               legends=legends, drawOptions=dopts,
                               returnPNG=False, useSVG=True)

    # Restate the canvas in millimetres, keeping the pixel grid as the viewBox,
    # so the drawing lands on the page at exactly the intended bond length.
    rows = -(-len(mols) // per_row)
    w_px, h_px = per_row * cell_px[0], rows * cell_px[1]
    w_mm, h_mm = per_row * cell_w_mm, rows * cell_h_mm
    # RDKit declares iso-8859-1; the file is written as UTF-8, so correct the
    # declaration or characters such as the µ of "µM" are dropped by readers.
    svg = svg.replace("encoding='iso-8859-1'", "encoding='UTF-8'", 1)
    svg = re.sub(r"width='[^']*'", f"width='{w_mm:.2f}mm'", svg, count=1)
    svg = re.sub(r"height='[^']*'", f"height='{h_mm:.2f}mm'", svg, count=1)
    # RDKit already writes a viewBox; add one only if that ever changes
    svg_tag = svg[svg.index("<svg"):svg.index(">", svg.index("<svg"))]
    if "viewBox" not in svg_tag:
        svg = svg.replace("<svg", f"<svg viewBox='0 0 {w_px} {h_px}'", 1)

    (RESULTS_DIR / filename).write_text(svg, encoding="utf-8")
    print(f"  {filename}: {len(mols)} structures, {per_row} per row, "
          f"{w_mm:.0f} x {h_mm:.0f} mm")


print("Drawing molecular grids (SVGs)...")
# Cell sizes come from the widest structure in each set plus room for the legend:
# building blocks span up to ~7.7 bonds (39 mm), products up to ~10.2 (52 mm).
draw_mol_grid_svg(bb2_counts, BB2_SMILES_COL, 'Grid_Label', BB2_CAS_COL,
                  'Grid_Top_Aldehydes.svg', per_row=4, n_max=20,
                  cell_mm=(PAGE_WIDTH_MM / 4, 39), prefix="AL2.")
draw_mol_grid_svg(bb1_counts, BB1_SMILES_COL, 'Grid_Label', BB1_CAS_COL,
                  'Grid_Top_Amines.svg', per_row=4, n_max=20,
                  cell_mm=(PAGE_WIDTH_MM / 4, 39), prefix="AM2.")
draw_mol_grid_svg(hits_df, 'Product_SMILES', 'id', PROB_COL,
                  'Grid_Top_Products.svg', per_row=3, n_max=12,
                  cell_mm=(PAGE_WIDTH_MM / 3, 50), ic50_col='pred_ic50_uM')

print(f"✅ Analysis complete! All publication SVGs saved to {RESULTS_DIR.resolve()}")