# Disordered Protein Dataset Preparation

This directory contains instructions and scripts to prepare the dataset of intrinsically disordered proteins (IDPs) for unconditional generation training. The raw data is sourced from the IDRome database.

## Step 1: Download Data

Download the disordered protein ensembles from the IDRome repository:
*   **Paper:** [Conformational ensembles of the human intrinsically disordered proteome (Nature, 2024)](https://www.nature.com/articles/s41586-023-07004-5)
*   **Download Link:** [SID - University of Copenhagen](https://sid.erda.dk/cgi-sid/ls.py?share_id=AVZAJvJnCO)

**Steps:**
1.  Download the PDB files (typically provided as `human_idrome_v1.zip` or split directories).
2.  Extract them into a temporary directory named `raw_pdb/` in this folder.

---

## Step 2: Reconstruct Full-Atom Structures (PDBFixer)

The raw structures from IDRome often contain only backbone atoms (CA) or have missing side-chains. We use **PDBFixer** to reconstruct full-atom representations.

### Prerequisites
You need to install `pdbfixer` and `openmm`.
```bash
conda install -c conda-forge pdbfixer openmm
```

## Step 3: process files
```python
python process_disorder_data.py \
    --input_dir ./fixed_pdb \
    --output_pdb_dir ./disorder_pdb_final \
    --output_csv disorder_data.csv \
    --val_ratio 0.05
```
