import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
import re
import sys
from pathlib import Path

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CRUCIAL FONT & VECTOR SETTINGS
# ---------------------------------------------------------
sns.reset_orig() # Prevent seaborn from hijacking font settings

mpl.rcParams['svg.fonttype'] = 'none'  # Keep SVG text completely editable
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 7   # Updated to 7pt to match Pearson/Spearman boxes
mpl.rcParams['axes.linewidth'] = 0.5  # 0.5 pt frames
# ---------------------------------------------------------

BOLTZ_CSV = DATA_DIR / "ExperimentalData_3Runs_WIDE.csv"
GNINA_CSV = DATA_DIR / "gnina_128_triplicates_wide.csv"
ACTIVITY_THRESHOLD_UM = 10.0

# GNINA ligands are docked as multi-conformer ensembles, so each ligand has two
# possible summaries per metric: the best-scoring pose over the whole ensemble,
# and the mean over conformers of each conformer's best pose.
#
# The ensemble maximum is used: it is the pose a prospective user of GNINA
# would take. Averaging over conformers instead dilutes the score with poor
# starting geometries and lowers the CNN-score AUROC from 0.68 to 0.56.
#
# Conformer counts differ between ligands (2-10, median 9), which raises the
# question of whether a maximum over more conformers is inflated. In this set it
# is not: Spearman rho between conformer count and CNNscore is -0.17 to -0.23
# across the three seeds, i.e. the more flexible ligands score worse, and
# averaging does not remove that association (-0.15 to -0.21). Set to True to
# reproduce the conformer-averaged analysis as a sensitivity check.
USE_MEANCONF = False
_SUF = "_meanconf" if USE_MEANCONF else ""

if not BOLTZ_CSV.exists() or not GNINA_CSV.exists():
    print(f"Error: Missing input datasets.")
    sys.exit(1)

df_boltz = pd.read_csv(BOLTZ_CSV)
df_gnina = pd.read_csv(GNINA_CSV)

# Merge datasets gracefully
cols_to_use = df_gnina.columns.difference(df_boltz.columns).tolist() + ['expID']
df = pd.merge(df_boltz, df_gnina[cols_to_use], on='expID', how='inner')

metric_runs = {
    'boltz_prob': ['boltz_prob_seed42', 'boltz_prob_seed1337', 'boltz_prob_seed2026'],
    'boltz_pIC50': ['boltz_pIC50_seed42', 'boltz_pIC50_seed1337', 'boltz_pIC50_seed2026'],
    'iptm_cdk2_lig': ['iptm_cdk2_lig_seed42', 'iptm_cdk2_lig_seed1337', 'iptm_cdk2_lig_seed2026'],
    'min_ipae': ['min_ipae_seed42', 'min_ipae_seed1337', 'min_ipae_seed2026'],
    'CNNscore': [f'CNNscore{_SUF}_run{i}' for i in (1, 2, 3)],
    'CNNaffinity': [f'CNNaffinity{_SUF}_run{i}' for i in (1, 2, 3)],
    'VinaAffinity': [f'VinaAffinity{_SUF}_run{i}' for i in (1, 2, 3)]
}

all_cols_needed = [col for runs in metric_runs.values() for col in runs]

missing_cols = [c for c in all_cols_needed if c not in df.columns]
if missing_cols:
    print(f"Error: columns absent from the merged table: {missing_cols}")
    sys.exit(1)

n_before = len(df)
df = df.dropna(subset=['expIC50'] + all_cols_needed).copy()
print(f"[INFO] GNINA columns: {'conformer-averaged' if USE_MEANCONF else 'ensemble maximum'}")
print(f"[INFO] Compounds retained: {len(df)} of {n_before}")
if len(df) < n_before:
    dropped = n_before - len(df)
    print(f"[WARNING] {dropped} compound(s) dropped for missing values. "
          f"Check the GNINA and Boltz tables before interpreting the results.")

def extract_id_number(id_str):
    match = re.search(r'(\d+)', str(id_str))
    return int(match.group(1)) if match else 999

df['id_num'] = df['expID'].apply(extract_id_number)
df = df.sort_values(by='id_num').reset_index(drop=True)

# Compute Empirical Means and StDevs
for base, cols in metric_runs.items():
    df[f'{base}_mean'] = df[cols].mean(axis=1)
    df[f'{base}_std'] = df[cols].std(axis=1)

plot_titles = {
    'boltz_prob': 'Boltz-2 Binding Probability',
    'boltz_pIC50': 'Boltz-2 Predicted pIC$_{50}$',
    'iptm_cdk2_lig': 'Boltz-2 ipTM (CDK2-Ligand)',
    'min_ipae': 'Boltz-2 min-iPAE (CDK2)',
    'CNNscore': 'GNINA CNN Pose Score',
    'CNNaffinity': 'GNINA CNN Predicted Affinity',
    'VinaAffinity': 'GNINA Vina Affinity (kcal/mol)'
}

summary_stats = []

# Metrics where a higher value means greater potency, so the correlation sign is
# flipped to align with pIC50
SIGN_FLIPPED = ['boltz_prob', 'iptm_cdk2_lig', 'CNNscore', 'CNNaffinity', 'boltz_pIC50']

# ==========================================
# 1. Correlation Figures
# ==========================================
# Metrics that also appear in the Fig. 5 quadriplot are not written out a second
# time; the remaining ones are grouped into double-column panels.

def metric_stats(base):
    """Regression and rank statistics for one metric, aligned to potency."""
    x = df['expIC50']
    y = df[f'{base}_mean']
    log_x = np.log10(x)
    fit_y = np.log10(y) if base == 'min_ipae' else y
    slope, intercept, r_val, _, _ = stats.linregress(log_x, fit_y)
    rho, _ = stats.spearmanr(x, y)
    aligned_rho, aligned_r = (-rho, -r_val) if base in SIGN_FLIPPED else (rho, r_val)
    return slope, intercept, aligned_r, aligned_rho


def draw_metric_panel(ax, base, stats_loc='upper left', headroom=0.0):
    """Draw one metric-against-IC50 correlation panel in the standard style.

    stats_loc places the Pearson/Spearman box in whichever corner is free of
    points; headroom expands the y-axis away from the data by that fraction of
    the span, to open up space for the box.
    """
    x = df['expIC50']
    y = df[f'{base}_mean']
    y_std = df[f'{base}_std']
    slope, intercept, aligned_r, aligned_rho = metric_stats(base)

    if base == 'min_ipae':
        ax.set_yscale('log')
        ax.invert_yaxis()
    elif base == 'VinaAffinity':
        ax.invert_yaxis()

    # Single-column points & error bars
    ax.errorbar(x, y, yerr=y_std, fmt='none', ecolor='gray', alpha=0.6, elinewidth=0.5, capsize=1.5, zorder=1)
    ax.scatter(x, y, color='#2c3e50', s=15, edgecolor='none', alpha=0.85, zorder=2)

    x_range = np.logspace(np.log10(x.min()*0.8), np.log10(x.max()*1.2), 100)
    if base == 'min_ipae':
        ax.plot(x_range, 10**(slope * np.log10(x_range) + intercept), color='black', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    else:
        ax.plot(x_range, slope * np.log10(x_range) + intercept, color='black', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

    ax.set_xscale('log')
    ax.invert_xaxis()

    if base == 'min_ipae':
        # Log axis spanning ~1.3 decades. Ticks are set explicitly: with the
        # default log locators only 1.0 is labelled, which hides the scale.
        ipae_ticks = [0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7]
        ax.yaxis.set_major_locator(ticker.FixedLocator(ipae_ticks))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter([f"{t:g}" for t in ipae_ticks]))
        ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5, length=3)
    ax.set_xlabel('Experimental IC$_{50}$ (µM)')
    ax.set_ylabel(plot_titles[base])

    if headroom:
        lo, hi = ax.get_ylim()          # hi < lo on the inverted axes
        if ax.get_yscale() == 'log':
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            ax.set_ylim(lo, 10**(log_hi + headroom * (log_hi - log_lo)))
        else:
            ax.set_ylim(lo, hi + headroom * (hi - lo))

    stats_text = f"Pearson $r$: {aligned_r:.2f}\nSpearman $\\rho$: {aligned_rho:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    # Explicitly matched to 7pt
    if stats_loc == 'lower right':
        pos, va, ha = (0.95, 0.05), 'bottom', 'right'
    else:
        pos, va, ha = (0.05, 0.95), 'top', 'left'
    ax.text(*pos, stats_text, transform=ax.transAxes, fontsize=7,
            verticalalignment=va, horizontalalignment=ha, bbox=props)


# Statistics for every metric, reported at the end of the run
for base in metric_runs.keys():
    _, _, aligned_r, aligned_rho = metric_stats(base)
    tool = "Boltz-2" if 'boltz' in base or 'ip' in base else "GNINA"
    summary_stats.append({'Metric': base, 'Tool': tool, 'Spearman_rho': aligned_rho, 'Pearson_r': aligned_r})

# Paired panels (double column width). The stats box goes wherever the points
# leave room: the GNINA metrics fill the panel, so space is opened at the top,
# while the Boltz-2 interface metrics leave the lower right corner empty.
METRIC_PANELS = [
    ('Correlation_GNINA_Affinity_Panel.svg', ['VinaAffinity', 'CNNaffinity'],
     dict(stats_loc='upper left', headroom=0.22)),
    ('Correlation_Boltz2_Interface_Panel.svg', ['min_ipae', 'iptm_cdk2_lig'],
     dict(stats_loc='lower right')),
]
for filename, bases, opts in METRIC_PANELS:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), dpi=300)
    for ax, base, panel in zip(axes, bases, ['A', 'B']):
        draw_metric_panel(ax, base, **opts)
        ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    plt.tight_layout(pad=1.5)
    plt.savefig(OUT_DIR / filename, format='svg', transparent=True, bbox_inches='tight')
    plt.close()

# ==========================================
# 1b. Correlation Plot With Marginal Distributions
# ==========================================
# Compounds whose 1PL fit ran into the 1 mM solubility bound are ties, not
# measurements, so they are drawn separately from the measured values.
CENSORED_IC50_UM = 1000.0

POINT_COLOR = '#2c3e50'
CENSORED_COLOR = '#D32F2F'


def plot_correlation_with_marginals(base, filename):
    """Scatter of a prediction metric against experimental IC50, with marginal
    histograms. The main panel repeats the standalone Correlation_<base>.svg;
    the marginals show the distribution of experimental potency (x, equivalent
    to a pIC50 histogram because the axis is logarithmic) and of the metric (y).
    """
    x = df['expIC50'].to_numpy()
    y = df[f'{base}_mean'].to_numpy()
    y_std = df[f'{base}_std'].to_numpy()
    log_x = np.log10(x)
    censored = x >= CENSORED_IC50_UM

    # Statistics over all 57 ligands, identical to the standalone figure
    slope, intercept, r_val, _, _ = stats.linregress(log_x, y)
    rho, _ = stats.spearmanr(x, y)
    aligned_rho, aligned_r = (-rho, -r_val) if base in SIGN_FLIPPED else (rho, r_val)

    # Double column width, so the legend can sit clear of the points
    fig = plt.figure(figsize=(6.9, 3.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_legend = fig.add_subplot(gs[0, 1])   # free corner of the joint layout
    ax_legend.axis('off')

    # --- Main panel ---
    ax.errorbar(x, y, yerr=y_std, fmt='none', ecolor='gray', alpha=0.6,
                elinewidth=0.5, capsize=1.5, zorder=1)
    ax.scatter(x[~censored], y[~censored], color=POINT_COLOR, s=15, edgecolor='none',
               alpha=0.85, zorder=2, label='Measured')
    ax.scatter(x[censored], y[censored], color=CENSORED_COLOR, s=15, edgecolor='none',
               alpha=0.85, zorder=3, label='Censored (> 1 mM)')

    x_range = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 1.2), 100)
    ax.plot(x_range, slope * np.log10(x_range) + intercept, color='black',
            linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5, length=3)
    ax.set_xlabel('Experimental IC$_{50}$ (µM)')
    ax.set_ylabel(plot_titles[base])

    stats_text = f"Pearson $r$: {aligned_r:.2f}\nSpearman $\\rho$: {aligned_rho:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='center', frameon=False, fontsize=6,
                     handletextpad=0.4, borderaxespad=0.0, labelspacing=0.6,
                     scatterpoints=1)

    # --- x-marginal: potency distribution, split by censoring ---
    x_bins = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 1.2), 18)
    _, _, patch_groups = ax_top.hist([x[~censored], x[censored]], bins=x_bins, stacked=True,
                                     color=[POINT_COLOR, CENSORED_COLOR],
                                     edgecolor='white', linewidth=0.3)
    for patch in patch_groups[1]:
        patch.set_hatch('///')
        patch.set_edgecolor('white')

    # --- y-marginal: distribution of the predicted metric ---
    ax_right.hist(y, bins=np.linspace(np.nanmin(y), np.nanmax(y), 18),
                  orientation='horizontal', color=POINT_COLOR,
                  edgecolor='white', linewidth=0.3, alpha=0.85)

    for marginal in (ax_top, ax_right):
        marginal.spines['top'].set_visible(False)
        marginal.spines['right'].set_visible(False)
        marginal.tick_params(width=0.5, length=3)
    # Only the count axes get a new locator; the shared axes keep the main panel's
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
    ax_right.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_top.set_ylabel('Count')
    ax_right.set_xlabel('Count')

    plt.savefig(OUT_DIR / filename, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}  ({censored.sum()} of {len(x)} ligands censored at "
          f"{CENSORED_IC50_UM:.0f} µM)")


plot_correlation_with_marginals('boltz_prob', 'Correlation_boltz_prob_marginals.svg')

# --- ROC Helper Function ---
y_true = (df['expIC50'] <= ACTIVITY_THRESHOLD_UM).astype(int)

def compute_empirical_roc(y_true, run_cols, df):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for col in run_cols:
        scores = df[col]
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0) 
    return mean_fpr, mean_tpr, std_tpr, np.mean(aucs), np.std(aucs)

# Color palette mapping
roc_metrics = {
    'boltz_prob': {'label': 'Boltz-2 Probability', 'color': '#D32F2F'}, # Crimson
    'boltz_pIC50': {'label': 'Boltz-2 pIC$_{50}$', 'color': '#2980B9'}, # Azure
    'CNNscore': {'label': 'GNINA CNN Score', 'color': '#F39C12'}        # Golden Yellow
}

# ==========================================
# 2. ROC Statistics
# ==========================================
# The ROC curves themselves are drawn as panel E of the Fig. 5 quadriplot; here
# only the AUC values are collected for the summary printed at the end.
roc_results = {}
for m in roc_metrics:
    _, _, _, mean_auc, std_auc = compute_empirical_roc(y_true, metric_runs[m], df)
    roc_results[m] = {'auc': mean_auc, 'auc_std': std_auc}

# ==========================================
# 3. Generate 2x2 Quadriplot (Publication Panel)
# ==========================================
# 7.2 inches = exactly 183 mm (double column width)
fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.3), dpi=300)

# The fourth element places the Pearson/Spearman box in whichever corner the
# points leave free. In panel D the CNN score trends up to the right, so the
# upper left used by B and C would sit on top of the data.
quad_plots = [
    (axes[0, 0], 'boltz_prob', 'B', 'upper left'),
    (axes[0, 1], 'boltz_pIC50', 'C', 'upper left'),
    (axes[1, 0], 'CNNscore', 'D', 'lower right')
]

for ax, base, panel, box_loc in quad_plots:
    x = df['expIC50']
    y = df[f'{base}_mean']
    y_std = df[f'{base}_std']
    log_x = np.log10(x)

    slope, intercept, r_val, _, _ = stats.linregress(log_x, y)
    rho, _ = stats.spearmanr(x, y)
    aligned_rho, aligned_r = -rho, -r_val 
        
    ax.errorbar(x, y, yerr=y_std, fmt='none', ecolor='gray', alpha=0.6, elinewidth=0.5, capsize=1.5, zorder=1)
    ax.scatter(x, y, color='#2c3e50', s=15, edgecolor='none', alpha=0.85, zorder=2)

    x_range = np.logspace(np.log10(x.min()*0.8), np.log10(x.max()*1.2), 100)
    ax.plot(x_range, slope * np.log10(x_range) + intercept, color='black', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

    ax.set_xscale('log')
    ax.invert_xaxis() 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5, length=3)
    ax.set_xlabel('Experimental IC$_{50}$ (µM)')
    ax.set_ylabel(plot_titles[base])

    stats_text = f"Pearson $r$: {aligned_r:.2f}\nSpearman $\\rho$: {aligned_rho:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    # Explicitly matched to 7pt
    bx, by, ha, va = ((0.05, 0.95, 'left', 'top') if box_loc == 'upper left'
                      else (0.97, 0.04, 'right', 'bottom'))
    ax.text(bx, by, stats_text, transform=ax.transAxes, fontsize=7,
            verticalalignment=va, horizontalalignment=ha, bbox=props)

    ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

# Quadrant D: Empirical ROC Curve
ax_roc = axes[1, 1]

for m, styles in roc_metrics.items():
    fpr, tpr, tpr_std, mean_auc, std_auc = compute_empirical_roc(y_true, metric_runs[m], df)
    ax_roc.plot(fpr, tpr, color=styles['color'], lw=1.0, label=f"{styles['label']} (AUC = {mean_auc:.2f})")
    tpr_upper = np.minimum(tpr + 2*tpr_std, 1)
    tpr_lower = np.maximum(tpr - 2*tpr_std, 0)
    ax_roc.fill_between(fpr, tpr_lower, tpr_upper, color=styles['color'], alpha=0.15, linewidth=0)

ax_roc.plot([0, 1], [0, 1], color='gray', lw=0.5, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')

ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
ax_roc.tick_params(width=0.5, length=3)
# Explicitly matched to 7pt
ax_roc.legend(loc="lower right", frameon=True, edgecolor='gray', fancybox=False, fontsize=7).get_frame().set_linewidth(0.5)
ax_roc.text(-0.15, 1.05, 'E', transform=ax_roc.transAxes, fontsize=8, fontweight='bold', va='bottom')

plt.tight_layout(pad=1.5)
plt.savefig(OUT_DIR / 'Fig5_Quadriplot_Summary.svg', format='svg', transparent=True, bbox_inches='tight')
plt.close()

# ==========================================
# 4. Physicochemical Property Correlations (MW, cLogP)
# ==========================================
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

mols = df['Product_SMILES'].apply(Chem.MolFromSmiles)
df['MolWt'] = mols.apply(lambda m: Descriptors.MolWt(m) if m is not None else np.nan)
df['cLogP'] = mols.apply(lambda m: Crippen.MolLogP(m) if m is not None else np.nan)

property_titles = {
    'MolWt': 'Molecular Weight (Da)',
    'cLogP': 'Crippen cLogP'
}

def plot_property_correlation(ax, prop):
    sub = df.dropna(subset=[prop])
    x = sub['expIC50']
    y = sub[prop]
    log_x = np.log10(x)

    slope, intercept, r_val, _, _ = stats.linregress(log_x, y)
    rho, _ = stats.spearmanr(x, y)
    # Sign aligned to potency (pIC50): positive = higher property value tracks with more potent compounds
    aligned_rho, aligned_r = -rho, -r_val

    ax.scatter(x, y, color='#2c3e50', s=15, edgecolor='none', alpha=0.85, zorder=2)
    x_range = np.logspace(np.log10(x.min()*0.8), np.log10(x.max()*1.2), 100)
    ax.plot(x_range, slope * np.log10(x_range) + intercept, color='black', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

    ax.set_xscale('log')
    ax.invert_xaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5, length=3)
    ax.set_xlabel('Experimental IC$_{50}$ (µM)')
    ax.set_ylabel(property_titles[prop])

    stats_text = f"Pearson $r$: {aligned_r:.2f}\nSpearman $\\rho$: {aligned_rho:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left', bbox=props)
    return aligned_r, aligned_rho

# Side-by-side panel (double column width); the individual properties are not
# written out separately
property_stats = {}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), dpi=300)
for ax, prop, panel in zip(axes, property_titles, ['A', 'B']):
    property_stats[prop] = plot_property_correlation(ax, prop)
    ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
plt.tight_layout(pad=1.5)
plt.savefig(OUT_DIR / 'Correlation_PhysChem_Panel.svg', format='svg', transparent=True, bbox_inches='tight')
plt.close()

# ==========================================
# 5. Command Line Outputs Restored
# ==========================================
print(f"Full analysis complete. All SVGs saved to {OUT_DIR.resolve()}")
print(f"\n--- AUROC Summary ({ACTIVITY_THRESHOLD_UM} µM cutoff) ---")
for m in roc_metrics.keys():
    # 2 decimal rounding applied here
    print(f"{roc_metrics[m]['label']}: {roc_results[m]['auc']:.2f} ± {roc_results[m]['auc_std']:.2f}")
print(f"\n--- Physicochemical Property Correlations (sign aligned to pIC50) ---")
for prop, (r, rho) in property_stats.items():
    print(f"{property_titles[prop]}: Pearson r = {r:.2f}, Spearman rho = {rho:.2f}")