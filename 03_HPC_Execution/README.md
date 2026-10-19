# 03_HPC_Execution: High-Performance Computing Hub

This directory serves as the execution hub for the computationally intensive phases of the CDK2 inhibitor discovery pipeline. Because of the extreme memory and compute requirements (particularly for structural prediction models), these steps are decoupled from the local environment and are designed to run on a SLURM-managed Linux HPC cluster.

*Note: The pipeline was configured and run on the Killarney supercomputer (SciNet HPC Consortium) using NVIDIA L40S GPUs.*

The outputs of these jobs are already provided in the downstream analysis folders, so the local analysis can be reproduced without running anything on a cluster.

---

## Directory Structure

The HPC hub is modularized into five distinct execution pathways to organize data flow and SLURM submissions:

* **01_HTS_10k_Library:** Boltz-2 virtual screening of the 10,000-compound combinatorial THQ library.
* **02_Aniline_Variation:** Boltz-2 structural probing of the indazole hinge-binding motif.
* **03_Experimental_Triplicates:** Boltz-2 inference for the 57 synthesized compounds (three seeds).
* **04_GNINA_Docking:** Baseline 3D molecular docking evaluation against CDK2 (three seeds). Docking uses chain A of 9OB3, the CDK2 monomer carrying phospho-Thr160; Cyclin E1 (chain B) is distant from the orthosteric ATP site and is excluded. Set `KEEP_CHAINS` in `scripts/prep_receptor.py` to `("A", "B")` to dock against the full heterodimer instead. The run begins with a redocking control of the co-crystallised ligand, which must reproduce the crystal pose before the production results are interpreted.
* **05_ChEMBL_3D_Similarity:** CPU-parallelized RDKit shape/color alignment against the Pan-CDK ChEMBL database.

All input files are included in each subfolder (`data/`, or `ligands/` and `receptor/` for GNINA).

---

## Environment Setup

Because the dependencies vary drastically between tasks (e.g., GPU-accelerated PyTorch for Boltz-2 vs. CPU-compiled RDKit for 3D alignment), the environment initialization scripts are located at the root of the `03_HPC_Execution` directory.

* `setup_boltz_env.sh` — Builds the shared GPU environment `env/` utilized by folders **01, 02, and 03**.
* `setup_gnina_env.sh` — Builds the dedicated docking environment `env_gnina/` for folder **04**.
* `setup_3d_env.sh` — Builds the CPU-parallelized chemoinformatics environment `chem_env/` for folder **05**.

Each environment is created at the `03_HPC_Execution` root. The SLURM scripts activate their environment themselves (e.g., `source ../env/bin/activate`) and stop with an error message if it has not been built yet.

**Cluster-specific settings:** The `module load` lines, the GPU request (`--gpus-per-node=l40s:1`), the cluster wheelhouse installs (`pip install --no-index`), and the OpenMM path in `setup_gnina_env.sh` are specific to the Alliance Canada clusters and must be adapted for other systems.

---

## Execution Workflow & SLURM Submission

### 1. Getting Started on the Cluster

1. **Copy the whole `03_HPC_Execution` folder to the cluster** (or clone the repository there). The job scripts expect this folder layout, with each environment one level above the job folders.
2. **Build the environment** for the jobs you want to run, once, on a login node (the setup scripts install packages from the internet):
   ```bash
   cd 03_HPC_Execution
   bash setup_boltz_env.sh    # folders 01-03
   bash setup_gnina_env.sh    # folder 04
   bash setup_3d_env.sh       # folder 05
   ```
3. **Set your allocation:** replace `YOUR_ACCOUNT` in the `#SBATCH --account=YOUR_ACCOUNT` line of each SLURM script with your own cluster allocation. (SLURM reads `#SBATCH` lines before the script runs, so this cannot be set through a shell variable.)
4. **Submit from the job's subfolder:**
   ```bash
   cd 01_HTS_10k_Library
   sbatch run_dimerCDK2.sh
   ```

All paths in the scripts are relative to their subfolder. The empty `logs/` folders are part of the repository because SLURM needs them to exist before a job starts.

| Folder | Job script |
|--------|------------|
| `01_HTS_10k_Library` | `run_dimerCDK2.sh` (array, 32 tasks) |
| `02_Aniline_Variation` | `Aniline_Variation.sh` |
| `03_Experimental_Triplicates` | `ExperimentalData_3Runs.sh` |
| `04_GNINA_Docking` | `run_gnina_triplicate_128.sh` |
| `05_ChEMBL_3D_Similarity` | `submit_job_pancdk.sh` (array, 6 tasks) |

**Internet access on compute nodes:** Boltz-2 is run with `--use_msa_server`, which queries the ColabFold MSA server from the compute node, and task 0 of the ChEMBL job downloads the reference ligands from the ChEMBL API. Both require outbound internet access on compute nodes, which some clusters block.

### 2. Pathfinder Logic in Array Jobs

The two array jobs use task 0 as a pathfinder, while tasks 1+ wait a few minutes before starting:

* **01_HTS_10k_Library:** Task 0 builds the Multiple Sequence Alignment (MSA) cache for the protein targets, which the other tasks then copy instead of querying the MSA server again.
* **05_ChEMBL_3D_Similarity:** Task 0 downloads the ChEMBL reference ligands and caches them in `data/pan_cdk_chembl_smiles.txt` for the other tasks.

The single jobs in folders 02 and 03 process their ligands sequentially. In folder 03, the MSA computed for the first ligand is reused for all later ligands and seeds.

### 3. Fault Tolerance (Checkpointing)
The Python execution engines are designed with strict checkpointing. If the cluster experiences a timeout or a node failure, simply resubmit the `.sh` script. The engine will automatically parse the existing output directories and skip all previously processed ligands.

---

## Post-Processing & Data Retrieval

After the jobs finish, the following steps produce the summary files used by the local analysis. Run them on the cluster from the job's subfolder, then copy the summary file into the listed folder of the repository.

| Folder | Post-processing step | Summary file | Copy to |
|--------|----------------------|--------------|---------|
| `01_HTS_10k_Library` | `bash merge_cdk2_results.sh` | `master_CDK2_HTS_results_10k.csv` | `04_Boltz2_Data_Analysis/data/` |
| `02_Aniline_Variation` | none (written by the job) | `results/results_Aniline_Variation/summary_Aniline_Variation.csv` | `04_Boltz2_Data_Analysis/data/` |
| `03_Experimental_Triplicates` | `python extract_wide_results.py` | `results/results_ExperimentalData_3Runs/ExperimentalData_3Runs_WIDE.csv` | `06_Correlation_Analysis/data/` |
| `04_GNINA_Docking` | `python scripts/get_wide_gnina_results.py` | `gnina_128_triplicates_wide.csv` | `06_Correlation_Analysis/data/` |
| `05_ChEMBL_3D_Similarity` | none (written by the job) | `data/outputs/*_pancdk_3d_results.csv` | `07_ChEMBL_3D_Similarity/data/outputs/` |

For the Python steps, activate the folder's environment first (e.g., `source ../env/bin/activate` in folder 03, `source ../env_gnina/bin/activate` in folder 04).

Folder 04 docks each ligand as a multi-conformer ensemble, so `get_wide_gnina_results.py` writes two summaries per metric: the best-scoring pose over the whole ensemble (`CNNscore_run1`, ...) and the mean over conformers of each conformer's best pose (`CNNscore_meanconf_run1`, ...). `06_Correlation_Analysis` uses the ensemble maximum (`USE_MEANCONF = False` in `analyze_correlations.py`), which is the pose a prospective user of GNINA would take; setting the flag to `True` reproduces the conformer-averaged analysis as a sensitivity check.

Conformer counts differ between ligands (2-10, median 9), so a maximum taken over more conformers could in principle be inflated. In this set it is not: Spearman rho between conformer count and `CNNscore` is -0.17 to -0.23 across the three seeds, i.e. the more flexible ligands also dock worse, and averaging over conformers does not remove that association (-0.15 to -0.21). Averaging does lower the CNN-score AUROC, from 0.68 to 0.56.

The job also writes `results_redock/redock_rmsd_summary.txt`, the redocking control of the co-crystallised ligand (mean top-pose RMSD 1.26 +/- 0.67 A across the three seeds); it is kept in the repository as a record and is not consumed by any downstream script.

The Boltz-2 jobs also generate large raw files (`.cif` structures, confidence `.json` and PAE `.npz` files, and temporary MSA caches). Only the summary files above are needed for the local analysis; those bulk outputs are deposited on Zenodo (see the Data Availability statement). The GNINA outputs are far smaller and are included in this repository in full, so `get_wide_gnina_results.py` can be re-run from a clean clone without a GPU.
