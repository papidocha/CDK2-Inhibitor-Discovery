"""
Redocking control: dock the co-crystallised ligand of 9OB3 back into the
prepared receptor and report the RMSD of the top-ranked pose to the crystal
geometry.

This separates "the scoring function performs poorly on this chemotype" from
"the receptor preparation or box definition is wrong". A top-pose RMSD below
~2 A is the usual acceptance criterion.

RMSD is computed with obrms (Open Babel), which is symmetry-aware and tolerant
of the bond-perception differences that arise because the reference ligand is
derived from a PDB record without explicit bond orders. RDKit is used as a
fallback if obrms is unavailable.

Run after the redocking step of run_gnina_triplicate_128.sh.
"""
import glob
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REFERENCE = "receptor/9OB3_crystal_ligand.sdf"
RESULTS_GLOB = "results_redock/crystal_seed_*.sdf.gz"
RMSD_THRESHOLD = 2.0


def decompress(path):
    """Write a gzipped SDF to a temporary plain SDF and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    with gzip.open(path, "rb") as gz:
        shutil.copyfileobj(gz, tmp)
    tmp.close()
    return tmp.name


def rmsd_obrms(ref, poses_sdf):
    """All pose RMSDs against the first molecule of the reference file."""
    if shutil.which("obrms") is None:
        return None
    try:
        out = subprocess.run(["obrms", "-f", ref, poses_sdf],
                             capture_output=True, text=True, timeout=300)
    except Exception as exc:
        print(f"[WARN] obrms failed to run: {exc}")
        return None
    if out.returncode != 0:
        print(f"[WARN] obrms exited {out.returncode}: {out.stderr.strip()[:200]}")
        return None
    vals = [float(m) for m in re.findall(r"RMSD\s+([0-9.]+)", out.stdout)]
    return vals or None


def rmsd_rdkit(ref_path, poses_sdf):
    """Fallback: RDKit symmetry-corrected RMSD, sanitising both molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign
    except ImportError:
        return None
    for sanitize in (True, False):
        try:
            ref = Chem.MolFromMolFile(ref_path, removeHs=True, sanitize=sanitize)
            if ref is None:
                continue
            vals = []
            for mol in Chem.SDMolSupplier(poses_sdf, removeHs=True, sanitize=sanitize):
                if mol is None:
                    continue
                vals.append(rdMolAlign.CalcRMS(mol, ref))
            if vals:
                return vals
        except Exception:
            continue
    return None


def pose_properties(path):
    """CNNscore and Vina affinity of each pose, in file order."""
    try:
        from rdkit import Chem
    except ImportError:
        return []
    props = []
    with gzip.open(path, "rb") as gz:
        for mol in Chem.ForwardSDMolSupplier(gz, sanitize=False):
            if mol is None:
                props.append((None, None))
                continue
            props.append((
                mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else None,
                mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else None,
            ))
    return props


if not os.path.exists(REFERENCE):
    sys.exit(f"[ERROR] Reference ligand not found: {REFERENCE}")

files = sorted(glob.glob(RESULTS_GLOB))
if not files:
    sys.exit(f"[ERROR] No redocking results matching {RESULTS_GLOB}.")

print("--- Redocking control (native ligand of 9OB3) ---")
print(f"Reference: {REFERENCE}")
print(f"Acceptance criterion: top-pose RMSD <= {RMSD_THRESHOLD:.1f} A\n")

top_rmsds, any_failure = [], False

for path in files:
    seed = os.path.basename(path).replace("crystal_seed_", "").replace(".sdf.gz", "")
    plain = decompress(path)
    try:
        vals = rmsd_obrms(REFERENCE, plain)
        method = "obrms"
        if vals is None:
            vals = rmsd_rdkit(REFERENCE, plain)
            method = "rdkit"
    finally:
        os.unlink(plain)

    if not vals:
        print(f"  seed {seed}: RMSD could not be computed by obrms or RDKit.")
        any_failure = True
        continue

    props = pose_properties(path)
    top_rmsd = vals[0]                 # GNINA writes poses in rank order
    best_rmsd = min(vals)
    cnn = props[0][0] if props else "n/a"
    vina = props[0][1] if props else "n/a"

    top_rmsds.append(top_rmsd)
    verdict = "PASS" if top_rmsd <= RMSD_THRESHOLD else "FAIL"
    if verdict == "FAIL":
        any_failure = True

    print(f"  seed {seed}: top-pose RMSD {top_rmsd:.2f} A [{verdict}] "
          f"| best of {len(vals)} poses {best_rmsd:.2f} A "
          f"| CNNscore {cnn} | Vina {vina} kcal/mol  ({method})")

if top_rmsds:
    mean = np.mean(top_rmsds)
    sd = np.std(top_rmsds, ddof=1) if len(top_rmsds) > 1 else 0.0
    print(f"\nMean top-pose RMSD across {len(top_rmsds)} seed(s): "
          f"{mean:.2f} +/- {sd:.2f} A")

if top_rmsds and not any_failure:
    print("[PASS] Receptor preparation and box reproduce the crystal pose.")
    sys.exit(0)

print("[FAIL] The crystal pose was not reproduced within the threshold, or the "
      "RMSD could not be computed. The job script runs under 'set -e', so this "
      "exit status deliberately stops the run before the production docking. "
      "Check the box centre/size and the receptor protonation, then resubmit.")
sys.exit(1)
