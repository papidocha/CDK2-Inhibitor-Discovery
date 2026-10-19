"""
Receptor preparation for GNINA docking against CDK2 (chain A of PDB 9OB3).

Docking is performed against the CDK2 monomer. Cyclin E1 (chain B) is distant
from the orthosteric ATP site and is excluded; set KEEP_CHAINS below to
("A", "B") to retain the full heterodimer instead.

Chains are selected by name rather than by size. Fragment-based selection would
also leave chain A here, but only because CDK2 (298 residues) is larger than
Cyclin E1 (274 residues), which is incidental to the structure rather than a
property of the selection.

The crystal ligand and solvent are deleted while the topology is still derived
from the CIF. Doing it at this stage keeps the 5-character ligand code A1CAH
out of the 3-character residue field of the PDB file that follows, so no
renaming is needed and the phospho-threonine is written directly as TPO.

The co-crystallised ligand is exported separately as the reference for the
redocking control (see redock_control.py).
"""
import os
import sys

from openmm.app import Modeller, PDBFile
from pdbfixer import PDBFixer

CIF_PATH = "receptor/9OB3.cif"
LIGAND_RESNAME = "A1CAH"            # co-crystallised ligand in 9OB3
SOLVENT_RESNAMES = {"HOH"}
KEEP_CHAINS = ("A",)                # chain A = CDK2 (carries pThr160 as TPO);
                                    # chain B = Cyclin E1

CLEAN_PDB = "receptor/temp_clean.pdb"
FIXED_PDB = "receptor/9OB3_fixed.pdb"
RECEPTOR_PDBQT = "receptor/receptor.pdbqt"
CRYSTAL_PDB = "receptor/9OB3_crystal_ligand.pdb"
CRYSTAL_SDF = "receptor/9OB3_crystal_ligand.sdf"

os.makedirs("receptor", exist_ok=True)
print(f"--- Receptor preparation (9OB3, chains {', '.join(KEEP_CHAINS)}) ---")

if not os.path.exists(CIF_PATH):
    sys.exit(f"[ERROR] {CIF_PATH} not found.")

fixer = PDBFixer(filename=CIF_PATH)

present = {r.name for r in fixer.topology.residues()}
if LIGAND_RESNAME not in present:
    print(f"[WARNING] No residue named {LIGAND_RESNAME} in the structure; "
          f"the redocking control cannot be run.")

# 1. Receptor: keep the selected chains, drop the crystal ligand and solvent
modeller = Modeller(fixer.topology, fixer.positions)
modeller.delete([r for r in modeller.topology.residues()
                 if r.chain.id not in KEEP_CHAINS
                 or r.name == LIGAND_RESNAME
                 or r.name in SOLVENT_RESNAMES])

chains = sorted({c.id for c in modeller.topology.chains()})
n_atoms = modeller.topology.getNumAtoms()
has_tpo = any(r.name == "TPO" for r in modeller.topology.residues())
print(f"[INFO] Chains retained: {chains} ({n_atoms} atoms)")
print(f"[INFO] Phospho-threonine (TPO) present: {has_tpo}")
if set(chains) != set(KEEP_CHAINS):
    print(f"[WARNING] Retained chains {chains} do not match KEEP_CHAINS "
          f"{list(KEEP_CHAINS)}.")
if not has_tpo:
    print("[WARNING] TPO not found - the activation-loop phosphorylation is "
          "missing from the receptor.")

with open(CLEAN_PDB, "w") as fh:
    PDBFile.writeFile(modeller.topology, modeller.positions, fh)

# 2. Crystal ligand, for the redocking RMSD reference
lig = Modeller(fixer.topology, fixer.positions)
lig.delete([r for r in lig.topology.residues() if r.name != LIGAND_RESNAME])
if lig.topology.getNumAtoms():
    with open(CRYSTAL_PDB, "w") as fh:
        PDBFile.writeFile(lig.topology, lig.positions, fh)
    print(f"[INFO] Crystal ligand extracted: {lig.topology.getNumAtoms()} atoms")
    os.system(f"obabel {CRYSTAL_PDB} -O {CRYSTAL_SDF} --silent")

# 3. Rebuild missing sidechains and add hydrogens at pH 7.4
print("[INFO] Adding missing atoms and hydrogens...")
fixer2 = PDBFixer(filename=CLEAN_PDB)
fixer2.findMissingResidues()
fixer2.findMissingAtoms()
fixer2.addMissingAtoms()
fixer2.addMissingHydrogens(7.4)

with open(FIXED_PDB, "w") as fh:
    PDBFile.writeFile(fixer2.topology, fixer2.positions, fh)

final_chains = sorted({c.id for c in fixer2.topology.chains()})
print(f"[INFO] After fixing: chains {final_chains}, "
      f"{fixer2.topology.getNumAtoms()} atoms")

# 4. PDBQT for GNINA (-xr keeps the receptor rigid and preserves residue names)
os.system(f"obabel {FIXED_PDB} -O {RECEPTOR_PDBQT} -xr")
print(f"[FINAL] Wrote {RECEPTOR_PDBQT}")
