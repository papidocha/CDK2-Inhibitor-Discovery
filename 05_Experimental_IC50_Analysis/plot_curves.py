import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import scipy.stats as stats
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from pathlib import Path

# ==========================================
# 0. Configuration & Paths
# ==========================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data_full_ic50"
PLOTS_DIR = SCRIPT_DIR / "plots"
RESULTS_DIR = SCRIPT_DIR / "results"

# Manuscript compound numbers (THQ) for the internal compound IDs
ID_MAP = pd.read_csv(SCRIPT_DIR.parent / "compound_id_map.csv").set_index("internal_id")["thq_id"].to_dict()

# Every titration is a 10-point, 3-fold dilution series starting at 100 uM.
N_POINTS = 10
CONCENTRATIONS = np.array([100.0 / (3.0**i) for i in range(N_POINTS)])
LOG_CONCENTRATIONS = np.log10(CONCENTRATIONS)

# ---------------------------------------------------------
# Both fitting modes are produced in one run:
#   "hill1" - Hill slope constrained to 1 (one-to-one binding).
#             Used for the main-text dose-response figures.
#   "free"  - all four parameters fitted.
# Each writes its own plot folder and results table.
# ---------------------------------------------------------
MODES = {
    "hill1": {"constrain_hill": True,
              "plots": PLOTS_DIR / "hill_constrained",
              "tsv": RESULTS_DIR / "ic50_results_hill1.tsv"},
    "free":  {"constrain_hill": False,
              "plots": PLOTS_DIR / "hill_free",
              "tsv": RESULTS_DIR / "ic50_results_free.tsv"},
}

# ---------------------------------------------------------
# Datasets to process. Each list entry is one INDEPENDENT
# experiment; rows within a file are TECHNICAL replicate wells.
# ---------------------------------------------------------
DATASETS = {
    "SMP124 CDK2": ["SMP124_CDK2_rep1.txt", "SMP124_CDK2_rep2.txt"],
    "SMP126 CDK2": ["SMP126_CDK2_rep1.txt", "SMP126_CDK2_rep2.txt", "SMP126_CDK2_rep3.txt"],
    "SMP210 CDK2": ["SMP210_CDK2_rep1.txt", "SMP210_CDK2_rep2.txt", "SMP210_CDK2_rep3.txt"],
    "SMP210b CDK2": ["SMP210b_CDK2_rep1.txt", "SMP210b_CDK2_rep2.txt", "SMP210b_CDK2_rep3.txt"],
}

# ---------------------------------------------------------
# CRUCIAL FONT & VECTOR SETTINGS
# ---------------------------------------------------------
mpl.rcParams['svg.fonttype'] = 'none'  # Keep SVG text completely editable
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.5  # 0.5 pt frames
# ---------------------------------------------------------


# ==========================================
# 1. Mathematical Models (Log-Parameterized)
# ==========================================
def logistic_4param_log(log_x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1 + 10**((log_x - log_ic50) * hill))


def format_2sigfigs(value):
    """Two significant figures, keeping trailing zeros (0.0597 -> '0.060')."""
    if value is None:
        return "N/A"
    value = float(value)
    if not np.isfinite(value) or value == 0:
        return "N/A"
    magnitude = int(math.floor(math.log10(abs(value))))
    decimals = max(0, 1 - magnitude)
    return f"{value:.{decimals}f}"


# ==========================================
# 2. Loading & per-experiment normalization
# ==========================================
def load_experiment(file_path):
    """
    Load ONE independent experiment.

    Each experiment is normalized against ITS OWN max/min activity controls,
    because the raw luminescence window differs between days, plates and
    reagent lots. Normalizing per experiment is what makes the experiments
    comparable on the 0-100% scale.

    Returns
    -------
    wells : (n_wells, N_POINTS) normalized activity, one row per technical well
    """
    n_cols = N_POINTS + 2

    df = pd.read_csv(file_path, sep='\t', header=None)
    df = df.dropna(axis=1, how='all')      # a stray trailing tab makes an empty column

    found = df.shape[1]
    if found < n_cols:
        raise ValueError(
            f"{file_path.name}: found {found} columns, expected {n_cols} "
            f"({N_POINTS} concentrations + max control + min control)."
        )
    if found > n_cols:
        print(f"    [{file_path.name}] note: {found} columns present, using first {n_cols}.")
    df = df.iloc[:, :n_cols]

    if df.isna().any().any():
        raise ValueError(f"{file_path.name}: contains blank or non-numeric cells.")

    col_means = df.mean(axis=0)
    signal_max = col_means.iloc[N_POINTS]
    signal_min = col_means.iloc[N_POINTS + 1]
    signal_range = signal_max - signal_min
    if not np.isfinite(signal_range) or signal_range == 0:
        raise ValueError(f"{file_path.name}: degenerate control window (max == min).")

    raw = df.iloc[:, 0:N_POINTS].values
    return (raw - signal_min) / signal_range * 100


def hill_bounds(constrain_hill):
    """Parameter bounds; curve_fit has no fixed-parameter API, so Hill is pinned
    with a tight bound when constrained."""
    if constrain_hill:
        return ([-20.0, 50.0, -10.0, 0.999], [50.0, 150.0, 5.0, 1.001])
    return ([-20.0, 50.0, -10.0, 0.1], [50.0, 150.0, 5.0, 5.0])


def fit_curve(log_concs, activity, constrain_hill):
    """4PL fit. Returns (popt, pcov) or (None, None)."""
    if len(activity) < 4:
        print(f"    fit skipped: only {len(activity)} points.")
        return None, None
    bounds = hill_bounds(constrain_hill)
    p0 = [0.0, 100.0, float(np.median(log_concs)), 1.0]
    try:
        popt, pcov = curve_fit(logistic_4param_log, log_concs, activity,
                               p0=p0, bounds=bounds, maxfev=20000)
        return popt, pcov
    except Exception as e:
        print(f"    fit failed: {e}")
        return None, None


# ==========================================
# 3. Main Processing Function
# ==========================================
def process_and_plot(name, file_names, constrain_hill, plots_dir):
    n_free = 3 if constrain_hill else 4

    paths = [DATA_DIR / f for f in file_names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"[Warning] {name}: missing {', '.join(missing)} - skipped.")
        return None

    try:
        experiments = [load_experiment(p) for p in paths]
    except ValueError as e:
        print(f"[Error] {name}: {e}")
        return None
    n_exp = len(experiments)
    n_wells = [e.shape[0] for e in experiments]

    # Collapse TECHNICAL wells within each experiment -> one curve per experiment.
    # This is the step that prevents technical replicates from inflating n.
    exp_curves = np.vstack([e.mean(axis=0) for e in experiments])   # (n_exp, N_POINTS)

    # Plotted points: mean across INDEPENDENT experiments; error = SD across them.
    activity = exp_curves.mean(axis=0)
    if n_exp > 1:
        activity_err = exp_curves.std(axis=0, ddof=1)
        err_label = f'Mean ± SD (n = {n_exp})'
    else:
        activity_err = experiments[0].std(axis=0, ddof=1)
        err_label = 'Mean ± SD (technical replicates)'

    # --- (A) Global fit across all independent experiments (curve that is drawn) ---
    gx = np.tile(LOG_CONCENTRATIONS, n_exp)
    gy = exp_curves.ravel()

    popt, pcov = fit_curve(gx, gy, constrain_hill)
    bottom = top = hill = log_ic50_val = ic50_val = r_squared = None
    ci_low = ci_high = None

    if popt is not None:
        bottom, top, log_ic50_val, hill = popt
        ic50_val = 10**log_ic50_val

        residuals = gy - logistic_4param_log(gx, *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((gy - gy.mean())**2))
        r_squared = 1 - ss_res / ss_tot

        lo, hi = hill_bounds(constrain_hill)
        pinned = [p for p, v, l, h in zip(["Bottom", "Top", "logIC50", "Hill"], popt, lo, hi)
                  if (np.isclose(v, l) or np.isclose(v, h))
                  and not (constrain_hill and p == "Hill")]
        if pinned:
            print(f"[{name}] WARNING: parameter(s) at fit bound: {', '.join(pinned)}. CIs unreliable.")

        se = np.sqrt(np.diag(pcov))[2]
        if np.isfinite(se):
            t_val = stats.t.ppf(0.975, len(gy) - n_free)
            ci_low = 10**(log_ic50_val - t_val * se)
            ci_high = 10**(log_ic50_val + t_val * se)
        else:
            print(f"[{name}] WARNING: covariance not estimable; CI unavailable.")

    # --- (B) Per-experiment fits -> geometric mean IC50 (the reported statistic) ---
    per_exp_ic50 = []
    if n_exp > 1:
        for i, curve in enumerate(exp_curves, start=1):
            p_i, _ = fit_curve(LOG_CONCENTRATIONS, curve, constrain_hill)
            if p_i is not None:
                per_exp_ic50.append(10**p_i[2])
            else:
                print(f"[{name}] experiment {i}: fit failed.")

    geo_mean = geo_sd = fold_range = ci_lo_g = ci_hi_g = None
    if len(per_exp_ic50) >= 2:
        logs = np.log10(per_exp_ic50)
        geo_mean = 10**logs.mean()
        geo_sd = 10**logs.std(ddof=1)                  # multiplicative SD
        fold_range = max(per_exp_ic50) / min(per_exp_ic50)
        t_val = stats.t.ppf(0.975, len(logs) - 1)
        sem = logs.std(ddof=1) / np.sqrt(len(logs))
        ci_lo_g = 10**(logs.mean() - t_val * sem)
        ci_hi_g = 10**(logs.mean() + t_val * sem)
        if fold_range > 3:
            print(f"[{name}] WARNING: independent experiments differ "
                  f"{fold_range:.1f}-fold in IC50. Check for a systematic "
                  f"difference between runs before pooling.")

    # --- PUBLICATION QUALITY PLOTTING ---
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.25, 2.75), dpi=300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5, length=3)
    plt.xscale('log')
    rng = np.random.default_rng(42)

    # 1. One jittered point per INDEPENDENT experiment
    for i in range(n_exp):
        x_jit = CONCENTRATIONS * rng.uniform(0.92, 1.08, size=N_POINTS)
        ax.scatter(x_jit, exp_curves[i], color='#95a5a6', alpha=0.5,
                   s=10, zorder=2, edgecolors='none')

    # 2. Mean +/- SD error bars
    ax.errorbar(CONCENTRATIONS, activity, yerr=activity_err,
                fmt='o', color='#2c3e50', capsize=2, elinewidth=0.5, markeredgewidth=0.5,
                markersize=4, zorder=4, label=err_label)

    # 3. Fitted curve
    if ic50_val is not None:
        x_range = np.logspace(np.log10(CONCENTRATIONS.min()),
                              np.log10(CONCENTRATIONS.max()), 200)
        ax.plot(x_range, logistic_4param_log(np.log10(x_range), *popt),
                color='#D32F2F', linewidth=1, zorder=3, label='4PL fit (Hill = 1)' if constrain_hill else '4PL fit')

    # 4. Reported IC50 with its 95% CI. Geometric mean of the per-experiment fits
    #    where there is more than one experiment; otherwise the single fit value.
    handles_extra = []
    if geo_mean is not None:
        ic50_txt = (rf"Geo. mean IC$_{{50}}$: {format_2sigfigs(geo_mean)} µM "
                    rf"(95% CI: {format_2sigfigs(ci_lo_g)}-{format_2sigfigs(ci_hi_g)})")
        handles_extra.append(Line2D([], [], linestyle='none', marker='none', label=ic50_txt))
    elif ic50_val is not None:
        ci_txt = (rf" (95% CI: {format_2sigfigs(ci_low)}-{format_2sigfigs(ci_high)})"
                  if ci_low is not None else "")
        handles_extra.append(Line2D([], [], linestyle='none', marker='none',
                                    label=rf"IC$_{{50}}$: {format_2sigfigs(ic50_val)} µM{ci_txt}"))

    ax.set_xlabel('Inhibitor Concentration (µM)')
    ax.set_ylabel('Normalized Kinase Activity (%)')
    max_y = 120 if ("SMP124" in name or "SMP126" in name) else 130
    min_y = min(-15, bottom - 10) if bottom is not None else -15
    ax.set_ylim(min_y, max_y)

    ax.axhline(50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    ax.axhline(100, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=1)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=1)

    h, l = ax.get_legend_handles_labels()
    for extra in handles_extra:
        h.append(extra); l.append(extra.get_label())
    ax.legend(h, l, loc='lower left', frameon=False, borderaxespad=1, handlelength=1.5)
    plt.tight_layout()

    compound, _, target = name.rpartition(" ")
    out_stem = f"{ID_MAP[compound]}_{target}"
    plt.savefig(plots_dir / f"{out_stem}_curve.svg", format='svg', transparent=True)
    plt.close()

    # --- Reporting ---
    print(f"\n[{name}]  n = {n_exp} independent experiment(s), {n_wells} technical well(s)"
          f"{'  [Hill fixed at 1.0]' if constrain_hill else ''}")
    if ic50_val is not None:
        ci_txt = (f"(95% CI {format_2sigfigs(ci_low)}-{format_2sigfigs(ci_high)})"
                  if ci_low is not None else "(CI unavailable)")
        print(f"    Global fit : IC50 {format_2sigfigs(ic50_val):>6} uM {ci_txt} | "
              f"pIC50 {6 - log_ic50_val:.2f} | Hill {hill:5.2f} | "
              f"Top {top:5.1f}% | Bottom {bottom:5.1f}% | R2 {r_squared:.3f}")
    if per_exp_ic50:
        print(f"    Per-expt   : {', '.join(format_2sigfigs(v) for v in per_exp_ic50)} uM")
        print(f"    Geo. mean  : {format_2sigfigs(geo_mean)} uM "
              f"(x/÷ {geo_sd:.2f}, {fold_range:.1f}-fold range; "
              f"95% CI {format_2sigfigs(ci_lo_g)}-{format_2sigfigs(ci_hi_g)})")

    return {
        "dataset": name,
        "compound": name.rpartition(" ")[0] or name,
        "n_exp": n_exp,
        "n_wells": "/".join(str(w) for w in n_wells),
        "reported_ic50": geo_mean if geo_mean is not None else ic50_val,
        "reported_ci_low": ci_lo_g if geo_mean is not None else ci_low,
        "reported_ci_high": ci_hi_g if geo_mean is not None else ci_high,
        "reported_kind": "geometric mean" if geo_mean is not None else "single fit",
        "geo_sd": geo_sd,
        "fold_range": fold_range,
        "per_exp_ic50": per_exp_ic50,
        "global_ic50": ic50_val,
        "global_ci_low": ci_low,
        "global_ci_high": ci_high,
        "pic50": (6 - log_ic50_val) if log_ic50_val is not None else None,
        "hill": hill,
        "top": top,
        "bottom": bottom,
        "r2": r_squared,
    }


# ==========================================
# 4. Execution Block
# ==========================================
if __name__ == "__main__":
    for mode, cfg in MODES.items():
        label = "Hill constrained to 1" if cfg["constrain_hill"] else "Hill unconstrained"
        print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")
        rows = []
        for name, files in DATASETS.items():
            r = process_and_plot(name, files, cfg["constrain_hill"], cfg["plots"])
            if r is not None:
                rows.append(r)
        if rows:
            df = pd.DataFrame(rows)
            df["per_exp_ic50"] = df["per_exp_ic50"].apply(
                lambda v: ", ".join(format_2sigfigs(x) for x in v) if v else "-")
            df.to_csv(cfg["tsv"], sep="\t", index=False)
            print(f"\n  -> {cfg['tsv'].name}   SVGs in {cfg['plots'].relative_to(SCRIPT_DIR)}/")

    print("\nProcessing complete.")