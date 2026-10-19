"""
Ligand preparation for GNINA docking.

Docking samples torsions but not ring conformations, so a single input
conformer would fix the norbornene bridge and the saturated tetrahydroquinoline
pucker for the whole run, and minimising that one conformer only descends to
the nearest local minimum. Instead an ensemble of ETKDGv3 conformers is
generated, MMFF-minimised, pruned by RMSD and energy, and written as a
multi-record SDF, so GNINA docks every ring geometry.

Ligands are written as SDF rather than PDBQT: GNINA reads SDF directly, which
avoids a round trip through OpenBabel that can alter the protonation state set
by Scrub.
"""
import os
import sys

import numpy as np
import pandas as pd
from molscrub import Scrub
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

DATA_FILE = "ligands/ExperimentalData_57Ligands_IC50.txt"
OUT_DIR = "ligands_confs"

# Conformers are embedded at arbitrary coordinates near the origin, far from the
# docking box. GNINA then has to randomise them into the box, which produced
# repeated "initial pose not within box" / "ligand outside box" messages.
# Translating each conformer's centroid onto the box centre avoids this.
BOX_CENTER = (39.668, 25.978, -10.781)   # must match run_gnina_triplicate_128.sh

N_EMBED = int(os.environ.get("N_EMBED", 20))   # conformers attempted per ligand
N_KEEP = int(os.environ.get("N_KEEP", 10))     # lowest-energy conformers retained
PRUNE_RMS = 0.5       # Å, duplicate-conformer pruning during embedding
SEED = 42             # fixed for reproducibility

# Optional subset for the smoke test: LIGAND_IDS="SMP210b,SMP126,SMP047"
LIGAND_IDS = [s.strip() for s in os.environ.get("LIGAND_IDS", "").split(",") if s.strip()]

if not os.path.exists(DATA_FILE):
    sys.exit(f"[ERROR] {DATA_FILE} not found.")

df = pd.read_csv(DATA_FILE, sep=None, engine="python")
if LIGAND_IDS:
    df = df[df["expID"].astype(str).isin(LIGAND_IDS)]
    missing = set(LIGAND_IDS) - set(df["expID"].astype(str))
    if missing:
        sys.exit(f"[ERROR] LIGAND_IDS not found in {DATA_FILE}: {sorted(missing)}")
    print(f"[INFO] Restricted to {len(df)} ligand(s): {', '.join(df['expID'])}")

scrubber = Scrub(ph_low=7.4, ph_high=7.4)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"--- Preparing {len(df)} ligands "
      f"({N_EMBED} conformers embedded, up to {N_KEEP} kept) ---")


def conformer_ensemble(mol, seed=SEED):
    """Embed, minimise and prune a conformer ensemble for one molecule."""
    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = PRUNE_RMS
    params.useSmallRingTorsions = True     # relevant for the bridged bicycle

    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_EMBED, params=params))
    if not cids:
        return None

    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    energies = np.array([e for (_converged, e) in results])
    order = np.argsort(energies)[:N_KEEP]

    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for idx in order:
        out.AddConformer(mol.GetConformer(int(cids[int(idx)])), assignId=True)

    # Centre every conformer on the docking box
    target = np.array(BOX_CENTER)
    for conf in out.GetConformers():
        pos = conf.GetPositions()
        shift = target - pos.mean(axis=0)
        for i in range(out.GetNumAtoms()):
            x, y, z = pos[i] + shift
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return out


n_ok, n_fail, counts = 0, 0, []

for _, row in df.iterrows():
    eid, smi = row["expID"], row["Product_SMILES"]

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"[FAILED] {eid}: SMILES could not be parsed.")
        n_fail += 1
        continue

    # Scrub assigns the protonation state at pH 7.4; geometry is regenerated below
    states = scrubber(mol)
    if not states:
        print(f"[FAILED] {eid}: Scrub returned no state.")
        n_fail += 1
        continue

    ensemble = conformer_ensemble(states[0])
    if ensemble is None or ensemble.GetNumConformers() == 0:
        print(f"[FAILED] {eid}: conformer embedding failed.")
        n_fail += 1
        continue

    out_path = os.path.join(OUT_DIR, f"{eid}.sdf")
    writer = Chem.SDWriter(out_path)
    for conf in ensemble.GetConformers():
        writer.write(ensemble, confId=conf.GetId())
    writer.close()

    counts.append(ensemble.GetNumConformers())
    n_ok += 1
    if ensemble.GetNumConformers() < 3:
        print(f"  [CHECK] {eid}: only {ensemble.GetNumConformers()} conformer(s) "
              f"- rigid molecule or difficult embedding.")

print(f"\n[SUMMARY] {n_ok} ligands written to '{OUT_DIR}/', {n_fail} failed.")
if counts:
    print(f"[SUMMARY] Conformers per ligand: min {min(counts)}, "
          f"median {int(np.median(counts))}, max {max(counts)}.")
    if min(counts) != max(counts):
        print("[NOTE] Ligands differ in conformer count. "
              "get_wide_gnina_results.py also writes '_meanconf' columns, "
              "averaged over conformers, for use as a sensitivity check.")
