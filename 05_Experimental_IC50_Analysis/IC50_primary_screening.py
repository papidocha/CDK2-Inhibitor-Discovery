import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# ==========================================
# 1. Parsing and Activity Functions
# ==========================================

def get_compound_columns(header, n_columns):
    compound_columns = {}
    
    # Find where each compound starts
    compound_positions = [
        i for i, x in enumerate(header)
        if x.startswith("SMP")
    ]
    
    for i, start in enumerate(compound_positions):
        compound = header[start]
        # End is the next compound or before max/blank controls
        if i < len(compound_positions) - 1:
            end = compound_positions[i + 1]
        else:
            end = n_columns - 2
            
        compound_columns[compound] = list(range(start, end))
        
    return compound_columns

def parse_plate(filename):
    # Read raw file
    with open(filename) as f:
        lines = [line.rstrip("\n").split("\t") for line in f]
        
    # First two rows
    header = lines[0]
    concentration_row = lines[1]
    
    # Determine compound columns
    compound_columns = get_compound_columns(
        header,
        len(concentration_row)
    )
    
    # Extract concentrations
    concentrations = {}
    for compound, cols in compound_columns.items():
        concentrations[compound] = [
            float(concentration_row[c])
            for c in cols
        ]
        
    # Extract RLU values
    rlu = np.array(lines[2:], dtype=float)
    
    return compound_columns, concentrations, rlu

def calculate_activity(compound_columns, concentrations, rlu):
    # Average plate controls
    mean_max = rlu[:, -2].mean()
    mean_blank = rlu[:, -1].mean()
    
    results = {}
    for compound, cols in compound_columns.items():
        results[compound] = {}
        for conc, col in zip(concentrations[compound], cols):
            # Average replicate RLU values
            mean_rlu = rlu[:, col].mean()
            # Calculate % activity
            activity = 100 * (mean_rlu - mean_blank) / (mean_max - mean_blank)
            results[compound][conc] = round(activity, 1)
            
    return results

def activity_to_dataframe(activity):
    # Collect all concentrations encountered across all compounds
    all_concentrations = sorted(
        {conc for compound in activity.values() for conc in compound.keys()},
        reverse=True
    )
    
    # Create dataframe
    df_activity = pd.DataFrame(
        index=activity.keys(),
        columns=all_concentrations
    )
    
    # Fill values
    for compound, values in activity.items():
        for conc, act in values.items():
            df_activity.loc[compound, conc] = float(act)
            
    # Name columns nicely
    df_activity.columns = [
        f"{int(c)} µM" for c in df_activity.columns
    ]
    
    # Move compound names from index into first column
    df_activity = df_activity.reset_index()
    df_activity = df_activity.rename(columns={"index": "Compound"})
    
    return df_activity

# ==========================================
# 2. IC50 Calculation Functions
# ==========================================

def round_sig(x, sig=1):
    """Rounds a float to the specified significant digits and removes trailing .0 for display."""
    if pd.isna(x) or x == 0.0:
        return np.nan
        
    # Standard sig fig rounding
    rounded = round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
    
    # Use the '%g' string format to automatically strip trailing '.0'
    # Returning it as a string prevents Pandas from forcing it back to a float (e.g. 10 -> 10.0)
    return f"{rounded:g}"

# Constrained Fitting Function (Fixed Top=100, Bottom=0, Hill=1)
def fixed_sigmoid(x, ic50):
    return 100 / (1 + (x / ic50))

def calculate_all_ic50s(df):
    # Identify concentration columns (those containing "µM")
    conc_cols = [c for c in df.columns if "µM" in c]
    
    # Extract the numeric values from the column headers
    concentrations = np.array([float(c.replace(" µM", "")) for c in conc_cols])
    
    results = []
    
    # Process Each Compound
    for index, row in df.iterrows():
        compound = row["Compound"]
        
        # Extract % activity values for this row
        y_data = row[conc_cols].values.astype(float)
        x_data = concentrations
        
        # Filter out any missing/NaN values if some concentrations weren't tested
        valid_mask = ~np.isnan(y_data)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        
        est_ic50 = np.nan
        
        # We need at least 1 valid point to attempt a fit
        if len(x_valid) > 0:
            try:
                # Fit with an initial guess of 10 µM and upper bound capped at 1000 µM
                popt, _ = curve_fit(
                    fixed_sigmoid, 
                    x_valid, 
                    y_valid, 
                    p0=[10.0], 
                    bounds=(1e-4, 1000)
                )
                est_ic50 = popt[0]
            except Exception:
                # If the fit fails (e.g., perfectly flat data), it remains NaN
                pass
                
        results.append({
            "Compound": compound,
            "Estimated_IC50_µM": round_sig(est_ic50, 1)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. Main Execution Block
# ==========================================

if __name__ == "__main__":
    # --- Robust Relative Pathing ---
    SCRIPT_DIR = Path(__file__).parent
    data_dir = SCRIPT_DIR / "data"
    results_dir = SCRIPT_DIR / "results"
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing text files from {data_dir.name}/ and calculating primary activity...")
    
    all_activity = []
    
    # Find and process all .txt files in the target data directory
    # If the data folder doesn't exist, this will just skip gracefully without crashing
    # Sorted, because compounds tested on several plates keep the first value read
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.txt")):
            compound_columns, concentrations, rlu = parse_plate(file)
            activity = calculate_activity(compound_columns, concentrations, rlu)
            df = activity_to_dataframe(activity)
            all_activity.append(df)
    
    if not all_activity:
        print(f"No .txt files found in the '{data_dir}' directory.")
        exit()

    # Combine all plates
    final_df = pd.concat(all_activity, ignore_index=True)
    
    # Merge duplicate compounds
    final_df = (
        final_df
        .groupby("Compound", as_index=False)
        .first()
    )
    
    # Sort compounds numerically by the number after SMP
    final_df["SMP_number"] = (
        final_df["Compound"]
        .str.extract(r"SMP(\d+)")[0]
        .astype(int)
    )
    
    final_df = (
        final_df
        .sort_values("SMP_number")
        .drop(columns="SMP_number")
        .reset_index(drop=True)
    )
    
    # Sort concentration columns from high to low
    concentration_columns = sorted(
        [c for c in final_df.columns if "µM" in c],
        key=lambda x: float(x.replace(" µM", "")),
        reverse=True
    )
    
    final_df = final_df[["Compound"] + concentration_columns]
    
    print("Estimating IC50 values...")
    
    # Calculate IC50s
    ic50_df = calculate_all_ic50s(final_df)
    
    # Merge the IC50 values back with the raw activity data
    combined_df = pd.merge(final_df, ic50_df, on="Compound", how="left")

    # Add the manuscript compound numbers (THQ) next to the internal IDs
    id_map = pd.read_csv(SCRIPT_DIR.parent / "compound_id_map.csv").set_index("internal_id")["thq_id"]
    combined_df.insert(1, "THQ_ID", combined_df["Compound"].map(id_map))

    # Display the final summary in the terminal
    print("\n--- Final Data Summary ---")
    print(combined_df.to_string())
    print("--------------------------\n")
    
    # Save Final Outputs using relative paths
    csv_out = results_dir / "CDK2_estimated_IC50.csv"
    xlsx_out = results_dir / "CDK2_estimated_IC50.xlsx"
    
    combined_df.to_csv(csv_out, index=False)
    combined_df.to_excel(xlsx_out, index=False)
    
    print(f"Processing complete. Saved outputs to {results_dir.name}/")