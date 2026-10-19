"""
Collate GNINA triplicate results into the wide-format table consumed by
06_Correlation_Analysis/analyze_correlations.py.

Each ligand is docked as a multi-conformer ensemble, so an output file contains
NUM_MODES poses for each input conformer, written in conformer order.

Two summaries are produced per metric:

  *_runN          best pose over the whole ensemble, selected by CNNscore.
  *_meanconf_runN mean over conformers of each conformer's best pose.

The ensemble maximum is the value used downstream: it is the pose a prospective
user of GNINA would take. The conformer-averaged column is provided as a
sensitivity check, since ligands do not all yield the same number of conformers
(2-10 across this set, median 9) and a maximum over more conformers could in
principle be inflated. In this set it is not - conformer count correlates
negatively with CNNscore (Spearman rho -0.17 to -0.23 across seeds), because the
more flexible ligands also dock worse, and averaging does not remove that
association.
"""
import gzip
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem

SEEDS = [42, 1337, 2026]
RUN_NAMES = ["run1", "run2", "run3"]
SEED_TO_RUN = dict(zip(SEEDS, RUN_NAMES))

DATA_FILE = "ligands/ExperimentalData_57Ligands_IC50.txt"
GNINA_DIR = Path("results_final_128")
OUTPUT_FILE = "gnina_128_triplicates_wide.csv"
NUM_MODES = 9        # must match --num_modes in run_gnina_triplicate_128.sh

METRICS = ("CNNscore", "CNNaffinity", "VinaAffinity")
EMPTY = {m: None for m in METRICS}
EMPTY.update({f"{m}_meanconf": None for m in METRICS})
EMPTY.update({"n_poses": 0, "n_conformers": 0})


def _read_poses(sdf_path):
    """All poses in file order as dicts of the three metrics."""
    poses = []
    with gzip.open(sdf_path, "rb") as gz:
        for mol in Chem.ForwardSDMolSupplier(gz, sanitize=False):
            if mol is None or not mol.HasProp("CNNscore"):
                continue
            poses.append({
                "CNNscore": float(mol.GetProp("CNNscore")),
                "CNNaffinity": float(mol.GetProp("CNNaffinity")),
                "VinaAffinity": float(mol.GetProp("minimizedAffinity")),
            })
    return poses


def best_pose(sdf_path):
    """Ensemble maximum and conformer-averaged metrics for one ligand/seed."""
    if not os.path.exists(sdf_path):
        return dict(EMPTY)

    try:
        poses = _read_poses(sdf_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {Path(sdf_path).name}: {exc}")
        return dict(EMPTY)

    if not poses:
        return dict(EMPTY)

    # Ensemble maximum: the single best pose, all metrics from that same pose
    top = max(poses, key=lambda p: p["CNNscore"])
    out = {m: top[m] for m in METRICS}

    # Conformer-averaged: split into NUM_MODES-sized chunks, best pose per chunk
    chunks = [poses[i:i + NUM_MODES] for i in range(0, len(poses), NUM_MODES)]
    per_conf = [max(c, key=lambda p: p["CNNscore"]) for c in chunks if c]
    for m in METRICS:
        vals = [p[m] for p in per_conf]
        out[f"{m}_meanconf"] = sum(vals) / len(vals) if vals else None

    out["n_poses"] = len(poses)
    out["n_conformers"] = len(per_conf)
    return out


def main():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Input data file not found: {DATA_FILE}")
        return

    df_exp = pd.read_csv(DATA_FILE, sep=None, engine="python")
    print(f"[INFO] Loaded {len(df_exp)} compounds. Extracting per-run metrics...")

    wide_records, missing, pose_counts, conf_counts = [], [], [], []

    for _, row in df_exp.iterrows():
        eid = str(row["expID"])
        record = row.to_dict()

        for seed in SEEDS:
            run_lbl = SEED_TO_RUN[seed]
            path = GNINA_DIR / f"{eid}_seed_{seed}_128_final.sdf.gz"
            data = best_pose(path)

            for m in METRICS:
                record[f"{m}_{run_lbl}"] = data[m]
                record[f"{m}_meanconf_{run_lbl}"] = data[f"{m}_meanconf"]
            record[f"n_conformers_{run_lbl}"] = data["n_conformers"]

            if data["CNNscore"] is None:
                missing.append(f"{eid}/seed{seed}")
            else:
                pose_counts.append(data["n_poses"])
                conf_counts.append(data["n_conformers"])

        wide_records.append(record)

    df_wide = pd.DataFrame(wide_records)
    df_wide.to_csv(OUTPUT_FILE, index=False)

    print(f"[SUCCESS] Wide-format replicate dataset saved to '{OUTPUT_FILE}'.")
    if pose_counts:
        print(f"[INFO] Poses evaluated per ligand/seed: "
              f"min {min(pose_counts)}, max {max(pose_counts)}.")
        print(f"[INFO] Conformers per ligand/seed: "
              f"min {min(conf_counts)}, max {max(conf_counts)}.")
        if min(conf_counts) != max(conf_counts):
            print("[NOTE] Conformer counts differ between ligands. The "
                  "'*_meanconf_*' columns summarise the same poses averaged "
                  "over conformers, for use as a sensitivity check.")
    if missing:
        print(f"[WARNING] {len(missing)} ligand/seed combinations had no usable pose:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    else:
        print("[INFO] All ligand/seed combinations produced a pose.")


if __name__ == "__main__":
    main()
