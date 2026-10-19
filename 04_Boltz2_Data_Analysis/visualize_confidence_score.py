import sys
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# Configuration
# ==========================================
# Input and Output Directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# Ensure the results folder exists before trying to save to it
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LIGANDS = [
    {
        "id": "lig_06170",
        "cif": "lig_06170_structure.cif",
        "smiles": "[H][C@@]12[C@H]3CC[C@H](C3)[C@]1([H])c1ccc3[nH]nc(Br)c3c1N[C@H]2c1cccc(NS(C)(=O)=O)c1"
    },
    {
        "id": "lig_02744",
        "cif": "lig_02744_structure.cif",
        "smiles": "[H][C@@]12[C@H]3CC[C@H](C3)[C@]1([H])c1ccc3[nH]ncc3c1N[C@H]2c1c[nH]c2cccc(OC)c12"
    },
    {
        "id": "lig_00217",
        "cif": "lig_00217_structure.cif",
        "smiles": "[H][C@@]12[C@H]3CC[C@H](C3)[C@]1([H])c1ccc3[nH]ncc3c1N[C@H]2c1ccc(OCCO)cc1"
    }
]

LIGAND_RESNAME = "LIG1" 

# ==========================================
# Functions
# ==========================================
def get_molstar_color(score):
    """Maps pLDDT to the standard AlphaFold/Mol* RGB color scheme for RDKit."""
    if score >= 90:   return (0.000, 0.325, 0.839)  # Very High (Dark Blue)
    elif score >= 70: return (0.396, 0.796, 0.953)  # Confident (Light Blue)
    elif score >= 50: return (1.000, 0.878, 0.400)  # Low (Yellow)
    else:             return (1.000, 0.490, 0.270)  # Very Low (Orange)

def generate_plddt_colorbar():
    """Generates a standalone publication-quality color bar legend."""
    out_png = RESULTS_DIR / "Fig_S1_pLDDT_ColorBar.png"
    print(f"\nGenerating Color Bar Legend: {out_png}")
    
    # Use Arial for consistency with the other figures
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(6, 0.8), dpi=600)
    
    # Exact Hex codes matching the RDKit RGB values
    cmap = mcolors.ListedColormap(['#FF7D45', '#FFE066', '#65CBF3', '#0053D6'])
    bounds = [0, 50, 70, 90, 100]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                      cax=ax, orientation='horizontal',
                      ticks=[0, 50, 70, 90, 100])
    
    cb.set_label('Boltz-2 Predicted Local Distance Difference Test (pLDDT)', fontsize=12, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(" -> Success!")

def process_ligand(ligand_data):
    """Processes a single ligand to generate a pLDDT mapped 2D image."""
    lig_id = ligand_data["id"]
    cif_file = DATA_DIR / ligand_data["cif"]
    smiles = ligand_data["smiles"]
    out_png = RESULTS_DIR / f"Fig_S1_{lig_id}_pLDDT.png"
    
    print(f"\n--- Processing {lig_id} ---")
    
    if not cif_file.exists():
        print(f"[!] Error: Could not find {cif_file}")
        return

    print(" 1. Generating 2D structure...")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"[!] Error: Invalid SMILES for {lig_id}.")
        return
        
    mol = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)

    print(" 2. Parsing CIF...")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(lig_id, cif_file)
    
    cif_heavy_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() == LIGAND_RESNAME:
                    for atom in residue:
                        if atom.element.upper() != 'H':
                            cif_heavy_atoms.append(atom)

    if not cif_heavy_atoms:
        print(f"[!] Error: Could not find '{LIGAND_RESNAME}' heavy atoms in {cif_file.name}.")
        return

    if len(cif_heavy_atoms) != mol.GetNumAtoms():
        print(f"[!] Error: Atom mismatch! SMILES has {mol.GetNumAtoms()} heavy atoms, CIF has {len(cif_heavy_atoms)}.")
        return

    print(" 3. Mapping scores...")
    atom_colors = {}
    for i, atom in enumerate(mol.GetAtoms()):
        score = cif_heavy_atoms[i].get_bfactor()
        atom_colors[atom.GetIdx()] = get_molstar_color(score)

    print(" 4. Rendering image...")
    d2d = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    opts = d2d.drawOptions()
    opts.useBWAtomPalette() 
    opts.padding = 0.1
    opts.highlightRadius = 0.35 
    opts.clearBackground = True
    
    d2d.DrawMolecule(
        mol, 
        highlightAtoms=list(atom_colors.keys()), 
        highlightAtomColors=atom_colors
    )
    d2d.FinishDrawing()
    
    with open(out_png, 'wb') as f:
        f.write(d2d.GetDrawingText())

    print(f" -> Saved {out_png}")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    for lig in LIGANDS:
        process_ligand(lig)
        
    generate_plddt_colorbar()
    print("\nAll tasks complete!")