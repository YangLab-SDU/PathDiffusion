"""
Script to process Disordered Protein PDB files.
Features:
1. Renumber residues starting from 1 (IDRome data usually starts at 0).
2. Extract sequence and metadata.
3. Split into train/val sets.
4. Save metadata to CSV and renumbered PDBs to a new folder.
"""

import os
import argparse
import pandas as pd
import warnings
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Polypeptide import PPBuilder, is_aa
from sklearn.model_selection import train_test_split

# Suppress PDB construction warnings (common with some disorder files)
warnings.simplefilter('ignore')


def extract_sequence_and_renumber(pdb_file, output_pdb_path):
    """
    Parses PDB, renumbers residues starting from 1, extracts sequence,
    and saves the modified PDB.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # 1. Renumber Residues (1-based)
    # IDRome structures usually have a single model and single chain
    for model in structure:
        for chain in model:
            # Sort residues to ensure correct order before renumbering
            # (Though IDRome files are usually sorted)
            residues = sorted([r for r in chain], key=lambda x: x.id[1])

            for i, residue in enumerate(residues):
                # Detach to allow ID modification if needed, though usually safe direct mod
                # Standard PDB format: residue ID is a tuple (hetero_flag, seq_num, insert_code)
                res_id = list(residue.id)
                res_id[1] = i + 1  # Set sequence number to 1-based index
                residue.id = tuple(res_id)

    # 2. Save Renumbered PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

    # 3. Extract Sequence (using the renumbered structure)
    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(structure):
        sequences.append(str(pp.get_sequence()))

    return ''.join(sequences)


def process_pdb_files(input_dir, output_pdb_dir):
    """Process all PDB files in the directory"""
    data = []

    # Ensure output directory exists
    os.makedirs(output_pdb_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    print(f"Found {len(files)} PDB files. Processing...")

    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_pdb_dir, filename)

        try:
            # Renumber and save, get sequence
            sequence = extract_sequence_and_renumber(input_path, output_path)

            seqlen = len(sequence)
            chain_name = os.path.splitext(filename)[0]  # Remove extension

            if seqlen > 0:
                data.append({
                    'chain_name': chain_name,
                    'seqlen': seqlen,
                    'seqres': sequence
                })
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser(description="Process Disorder PDBs: Renumber and Generate Metadata")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing PDB files (e.g., after pdbfixer)"
    )
    parser.add_argument(
        "--output_pdb_dir",
        type=str,
        required=True,
        help="Path to save the renumbered (1-based) PDB files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="disorder_data.csv",
        help="Path to save the output metadata CSV"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for validation (default: 0.05)"
    )

    args = parser.parse_args()

    print(f"Input Directory: {args.input_dir}")
    print(f"Output PDB Directory: {args.output_pdb_dir}")

    df = process_pdb_files(args.input_dir, args.output_pdb_dir)

    # Add train/val split
    if not df.empty:
        # Assign 'train' by default
        df['train_val_test'] = 'train'

        if args.val_ratio > 0:
            # Create a random validation set
            val_indices = df.sample(frac=args.val_ratio, random_state=42).index
            df.loc[val_indices, 'train_val_test'] = 'val'

        # Add required columns for compatibility with RCSBDataset
        df['coil_ratio'] = 1.0  # Disordered proteins are mostly coil

        df.to_csv(args.output_csv, index=False)
        print(f"Processing complete.")
        print(f"Total sequences: {len(df)}")
        print(f"Training samples: {len(df[df['train_val_test'] == 'train'])}")
        print(f"Validation samples: {len(df[df['train_val_test'] == 'val'])}")
        print(f"Metadata saved to {args.output_csv}")
    else:
        print("No valid PDB files processed.")


if __name__ == "__main__":
    main()