# Discovery of Nanomolar CDK2 Inhibitors Using Boltz-2

Analysis code and data for the manuscript *Discovery of Nanomolar CDK2 Inhibitors Using Boltz-2*.

This repository covers the full computational pipeline of the study: curation of commercial building blocks, enumeration of a virtual tetrahydroquinoline (THQ) library, Boltz-2 cofolding predictions and GNINA docking on an HPC cluster, the analysis of the biochemical CDK2/Cyclin E1 assay data, the correlation of predictions with experiment, a 3D similarity comparison against known CDK inhibitors, and the Opentrons Flex protocols used for the assays. All figure-generating scripts and the data they consume are included, so the manuscript figures can be reproduced locally.

> Citation: *to be added upon publication.*
> Raw computational outputs: Zenodo, [10.5281/zenodo.23406572](https://doi.org/10.5281/zenodo.23406572)

---

## Pipeline Architecture

This pipeline is designed as a highly modular, hybrid computational and experimental workflow. Modules `01`, `02`, and `04–07` are executed locally for data preparation, virtual library generation, and downstream statistical and visual analysis. Module `03` serves as a centralized execution hub requiring a Linux HPC environment to handle GPU-accelerated deep learning (Boltz-2, GNINA) and similarity scoring. Module `08` bridges the computational predictions with physical wet-lab validation by providing automated liquid handling protocols.

| Module | Execution Environment | Hardware Requirements | Environment Setup |
|--------|-----------------------|-----------------------|-------------------|
| `01_BB_Curation` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `02_Library_Generation` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `03_HPC_Execution` | SLURM Cluster | Multi-core CPUs & NVIDIA GPUs (e.g., L40S) | Run specific `setup_*_env.sh` at folder root |
| `04_Boltz2_Data_Analysis` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `05_Experimental_IC50_Analysis` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `06_Correlation_Analysis` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `07_ChEMBL_3D_Similarity` | Local Workstation | Standard CPU | `pip install -r local_requirements.txt` |
| `08_Opentrons_Assay_Automation`| Opentrons App | Opentrons Flex Robot | Load directly into Opentrons App |

---

## Installation

Tested with Python 3.12 on Windows and Linux.

```bash
git clone https://github.com/papidocha/CDK2-Inhibitor-Discovery.git
cd CDK2-Inhibitor-Discovery
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r local_requirements.txt
```

The HPC module (`03`) uses its own environments, built on the cluster; see [`03_HPC_Execution/README.md`](03_HPC_Execution/README.md).

---

## Running the Analysis

All scripts resolve their paths relative to their own location and can be run from anywhere, e.g. `python 06_Correlation_Analysis/analyze_correlations.py`. Modules `04`, `06`, and `07` are independent of each other and can be run directly; the HPC summary files they need are already in their `data/` folders.

**01_BB_Curation**
1. `curate_all_building_blocks.py` — filters the supplier catalogues into the curated building blocks. Requires the commercial catalogues, which are not redistributable (see below), and is skipped on a fresh clone.
2. `plot_aldehyde_space.py` — t-SNE of the aldehyde chemical space.

**02_Library_Generation**
1. `generate_thq_library.py` — enumerates the 10,000-member virtual THQ library from the curated building blocks.

**04_Boltz2_Data_Analysis**
1. `analyze_HTS_results.py` — hit rates, building-block enrichment, and validation plots for the 10k screen.
2. `analyze_aniline_results.py` — SAR heatmap of the focused amine series.
3. `visualize_confidence_score.py` — per-atom pLDDT figures for three representative ligands.

**05_Experimental_IC50_Analysis** (order matters)
1. `IC50_primary_screening.py` — parses the raw plate readings (`data/RLU*.txt`) and estimates IC50 values from the 1PL fit.
2. `plot_curves.py` — fits the full 10-point dose-response data (`data_full_ic50/`) and writes the curves plus the results tables.
3. `make_table.py` — builds the Supplementary Word tables from the output of step 2.
4. `prepare_boltz_input.py` — merges the measured IC50 values into the 57-ligand input file used by the HPC jobs. Needs steps 1 and 2.

**06_Correlation_Analysis**
1. `analyze_correlations.py` — correlations and ROC analysis of Boltz-2 and GNINA predictions against experiment.

**07_ChEMBL_3D_Similarity**
1. `analyze_3d_similarity.py` — aggregates the 3D alignment scores. Requires internet access: the top-5 reference image queries the ChEMBL API for the CDK targets of each match.

---

## Manuscript Figures

| Figure | Produced by | Output |
|--------|-------------|--------|
| Fig. 1D | — | Reaction matrix, measured by NMR (not in this repository) |
| Fig. 2A | `04/analyze_HTS_results.py` | `04_Boltz2_Data_Analysis/results/Fig_BB_Hit_Rates.svg` |
| Fig. 2B | `04/analyze_HTS_results.py` | `04_Boltz2_Data_Analysis/results/Grid_Top_Amines.svg` |
| Fig. 2C | `01/plot_aldehyde_space.py` | `01_BB_Curation/analysis/Fig_Aldehyde_Chemical_Space_tSNE.svg` |
| Fig. 2D | `05/IC50_primary_screening.py` | `05_Experimental_IC50_Analysis/results/CDK2_estimated_IC50.csv` (values) |
| Fig. 3 | `05/plot_curves.py` | `05_Experimental_IC50_Analysis/plots/hill_constrained/THQ21_*.svg`, `THQ6_*.svg` |
| Fig. 4C | `04/analyze_aniline_results.py` | `04_Boltz2_Data_Analysis/results/Fig_SAR_BoltzProb.svg` |
| Fig. 4E | `05/plot_curves.py` | `05_Experimental_IC50_Analysis/plots/hill_constrained/THQ25_*.svg`, `THQ26_*.svg` |
| Fig. 5A | `05/IC50_primary_screening.py` | `05_Experimental_IC50_Analysis/results/CDK2_estimated_IC50.csv` (values) |
| Fig. 5B–E | `06/analyze_correlations.py` | `06_Correlation_Analysis/results/Fig5_Quadriplot_Summary.svg` |
| Suppl. Fig. S1 | `04/visualize_confidence_score.py` | `04_Boltz2_Data_Analysis/results/Fig_S1_*.png` |
| Suppl. Fig. S2 | `07/analyze_3d_similarity.py` | `07_ChEMBL_3D_Similarity/results/Fig_S2_*.svg`, `THQ25_Top5_3D_Matches_Annotated.png` |
| Suppl. Tables 1–2 | `05/make_table.py` | `05_Experimental_IC50_Analysis/results/IC50_table_hill_*.docx` |

Panels were assembled and annotated for publication in Adobe Illustrator; the scripts above produce the underlying plots. Additional outputs not shown in the manuscript (individual correlation plots, standalone ROC curve, physicochemical panels, molecular grids) are also written to the `results/` folders.

Figures are rendered in Arial. On systems without that font, matplotlib falls back to a default font and the figures will differ slightly in appearance.

---

## Data Handoff & Execution Workflow

To seamlessly bridge the local, HPC, and wet-lab environments, this pipeline is structured for immediate reproducibility:

1. **Upload to Cluster:** The required input files (e.g., the 10k virtual library and experimental SMILES) are already pre-populated within the respective `data/` subdirectories of `03_HPC_Execution/`. Transfer this entire module directory to your Linux HPC cluster; see [`03_HPC_Execution/README.md`](03_HPC_Execution/README.md) for the setup and submission steps.
2. **Execute:** Submit the jobs to the SLURM scheduler from their respective subfolders.
3. **Retrieve Summaries:** Once execution completes, large structural files (such as `.cif` structures and `.sdf` docking poses) remain on the cluster. Download **only the lightweight summary `.csv` files** generated by the jobs back to your local machine.
4. **Local Analysis (Ready to Run):** The summary files generated by the HPC are already provided in the `data/` directories of modules `04`, `06`, and `07`. You can execute the local visualization and statistical scripts immediately to reproduce manuscript figures without needing to run the cluster jobs. The three Boltz-2 `.cif` structures in `04_Boltz2_Data_Analysis/data/` are representative outputs of the 10k screen, kept for the per-atom confidence figures.
5. **Experimental Validation:** The optimized assay protocols for the Opentrons Flex are located in `08_Opentrons_Assay_Automation`. The resulting raw experimental data is already populated in `05_Experimental_IC50_Analysis` (`data/` for the primary screening plates, `data_full_ic50/` for the 10-point dose-response experiments), allowing immediate regeneration of the IC50 curves and their correlation to computational predictions.

### Compound Identifiers
Data files, file names and scripts refer to compounds by internal IDs (e.g., `SMP124`). [`compound_id_map.csv`](compound_id_map.csv) maps each internal ID to the compound number used in the manuscript (e.g., `THQ21`). Figures and tables that appear in the manuscript or Supporting Information are labeled with the manuscript numbers.

### ChEMBL Reference Set
The ChEMBL reference set used in `07_ChEMBL_3D_Similarity` (10,402 unique ligands with reported IC50 values against human CDKs) was retrieved via the ChEMBL API on 13 September 2026 (ChEMBL_37). The exact set is included as `03_HPC_Execution/05_ChEMBL_3D_Similarity/data/pan_cdk_chembl_smiles.txt`; the alignment job loads this cache instead of querying the API, so the analysis reproduces the published reference set. Deleting the cache makes the job download the current ChEMBL release instead, which will differ.

### Note on Commercial Building-Block Data
The building-block curation in `01_BB_Curation` starts from commercial supplier catalogues that contain pricing information and cannot be redistributed. These catalogues are not included, so `curate_all_building_blocks.py` cannot be rerun from this repository alone. The manual exclusions in the script refer to row positions in the original catalogue files and do not apply to other catalogue versions.

The outcome of the curation is provided in full: the curated building blocks (`01_BB_Curation/results/*/curated_*_smiles_cas.csv`, SMILES and CAS numbers) and the manually excluded compounds (`01_BB_Curation/results/*/manually_dropped_*.csv`). The library generation in `02_Library_Generation` and all downstream steps run from these files.

---

## Data Availability

Lightweight tabular data are included here: the enumerated 10,000-member virtual THQ library, the experimental dataset of all 57 biochemically evaluated ligands with their Boltz-2 and GNINA scores, the raw plate readings, and the ChEMBL 3D similarity scores.

The GNINA docking outputs are small enough to version with the code and are included in full: the prepared receptor, the multi-conformer ligand inputs, all 171 pose files (57 ligands x 3 seeds) in `03_HPC_Execution/04_GNINA_Docking/results_final_128/`, and the redocking control in `results_redock/`. The wide summary table can therefore be regenerated from a clean clone without a GPU.

The bulk computational outputs that are too large for this repository (Boltz-2 `.cif` structures, affinity and confidence `.json` files, PAE `.npz` files, and the ChEMBL 3D-similarity score tables) are deposited on Zenodo: [10.5281/zenodo.23406572](https://doi.org/10.5281/zenodo.23406572). This DOI always resolves to the most recent version of the deposit; the version used for the manuscript is [10.5281/zenodo.23406573](https://doi.org/10.5281/zenodo.23406573).

## License

* **Code:** MIT License (see [`LICENSE`](LICENSE)).
* **Data and figures** generated in this work: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Third-party data keep their original licenses:

* **ChEMBL:** The ChEMBL-derived structures in `07_ChEMBL_3D_Similarity/data/outputs/` and `07_ChEMBL_3D_Similarity/results/` were retrieved from ChEMBL (EMBL-EBI, https://www.ebi.ac.uk/chembl/) and are redistributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
* **PDB:** The structure `03_HPC_Execution/04_GNINA_Docking/receptor/9OB3.cif` is PDB entry 9OB3 from the wwPDB, available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
