import sys
import os
import math
import subprocess
import re
import argparse
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import QuantileTransformer


def run_nwalign(query_seq_file: str, target_seq_file: str, nwalign_path: str) -> Tuple[str, str]:
    try:
        result = subprocess.run(
            [nwalign_path, query_seq_file, target_seq_file],
            capture_output=True,
            text=True,
            check=True
        )

        output_lines = result.stdout.split('\n')
        aligned_seq1 = ""
        aligned_seq2 = ""
        match_line_found = False
        in_sequence_section = False

        for line in output_lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if "Sequence identity:" in stripped_line:
                in_sequence_section = True
                continue

            if not in_sequence_section:
                continue

            if any(c in stripped_line for c in [":", "."]):
                match_line_found = True
                continue

            if not aligned_seq1 and not match_line_found:
                aligned_seq1 = stripped_line
            elif match_line_found and not aligned_seq2:
                aligned_seq2 = stripped_line

        if not aligned_seq1 or not aligned_seq2:
            raise ValueError("Failed to parse alignment sequences from NWalign output")

        return aligned_seq1, aligned_seq2

    except subprocess.CalledProcessError as e:
        raise ValueError(f"NWalign execution failed: {e.stderr}")
    except Exception as e:
        raise ValueError(f"NWalign parsing failed: {str(e)}")


def read_fasta_sequence(fasta_file: str) -> str:
    sequence = ""
    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequence += line.strip()
    return sequence


def map_scores_to_target(query_scores: List[float], aligned_query: str, aligned_target: str) -> List[float]:
    target_scores = [0.0] * len(aligned_target.replace("-", ""))

    query_pos = target_pos = 0
    for q, t in zip(aligned_query, aligned_target):
        if t != "-":
            if q != "-":
                if query_pos < len(query_scores):
                    target_scores[target_pos] = query_scores[query_pos]
                query_pos += 1
            target_pos += 1
        elif q != "-":
            query_pos += 1

    return target_scores


def find_existing_pdb_protein(input_file: str, afdb_pdb_dir: str, mode: str) -> Optional[str]:
    protein_ids = []
    seen = set()

    print(f"Parsing input file: {input_file}, mode: {mode}")

    try:
        with open(input_file, 'r') as f:
            for line in f:
                pid = None

                match = re.search(r"afdb50_([^\s]+)", line)
                if match:
                    pid = match.group(1)

                if pid and pid not in seen:
                    protein_ids.append(pid)
                    seen.add(pid)

        if not protein_ids:
            print(f"Error: No protein ID matching format (afdb50_XXX) found in file {input_file} (mode={mode})")
            return None

        print(f"Extracted {len(protein_ids)} unique protein IDs, looking for local PDB...")

        for protein_id in protein_ids:
            dir1 = protein_id[0:2]
            dir2 = protein_id[2:4]
            dir3 = protein_id[4:6]
            pdb_file = f"{afdb_pdb_dir}/{dir1}/{dir2}/{dir3}/{protein_id}.pdb"
            if os.path.exists(pdb_file):
                print(f"[{mode}] Found first existing PDB file: {protein_id}")
                return protein_id

        print(f"Error: None of the protein IDs found in file {input_file} exist in the PDB library")
        return None

    except IOError as e:
        print(f"Cannot read input file: {e}")
        return None


def extract_sequence_from_pdb(pdb_file: str) -> str:
    sequence = ""
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    try:
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res_name = line[17:20].strip()
                    sequence += aa_dict.get(res_name, 'X')
    except IOError as e:
        raise ValueError(f"Cannot read PDB file: {e}")

    if not sequence:
        raise ValueError("No valid sequence extracted from PDB file")

    return sequence


def save_sequence_to_fasta(sequence: str, output_file: str, header: str = ">sequence") -> None:
    with open(output_file, 'w') as f:
        f.write(f"{header}\n{sequence}\n")


def process_rmsd_files(rmsd_dir: str, temp_output: str) -> None:
    files = [f for f in os.listdir(rmsd_dir) if f.endswith('.rmsd')]

    for file in files:
        file_path = os.path.join(rmsd_dir, file)
        rm_file = False

        try:
            with open(file_path) as f, open(temp_output, 'a+') as outfile:
                for line in f:
                    if line.startswith("Chain_1TM-score="):
                        tmscore = float(line.split()[1])
                        outfile.write(f"{file[:-5]}\t{tmscore}\n")
                        if tmscore < 0.3:
                            rm_file = True

            if rm_file:
                os.remove(file_path)
        except IOError as e:
            print(f"Error processing file {file}: {e}")


def calculate_alignment_scores(rmsd_dir: str, seq_length: int, sigma: float = 1.0) -> tuple:
    res_scores = [0.0] * seq_length
    num_files = 0

    if not os.path.exists(rmsd_dir):
        return res_scores, 0

    for file in os.listdir(rmsd_dir):
        if not file.endswith('.rmsd'):
            continue
        if "temp" in file or "fasta" in file:
            continue

        file_path = os.path.join(rmsd_dir, file)

        if not os.path.isfile(file_path): continue

        try:
            with open(file_path) as f:
                content = f.read()
                if "align_result:" not in content:
                    continue

            num_files += 1
            with open(file_path) as f:
                t1, t2, t3 = " ", " ", " "
                align_i = 0
                stop_read_res_rmsd = False
                res_rmsd = []

                for line in f:
                    if line.startswith("RMSD"):
                        stop_read_res_rmsd = True
                    if line.startswith("align_result:"):
                        align_i = 1

                    if not stop_read_res_rmsd and line.strip() and not line.startswith("Chain"):
                        try:
                            val = float(line.split()[1])
                            res_rmsd.append(val)
                        except:
                            pass

                    if align_i == 2:
                        t1 = line.strip()
                    elif align_i == 3:
                        t2 = line.strip()
                    elif align_i == 4:
                        t3 = line.strip()

                    if align_i >= 1:
                        align_i += 1

                res_idx = 0
                align_idx = 0

                iter_len = min(len(t1), len(t3))

                for i in range(iter_len):
                    if t1[i] != "-":
                        if t3[i] != "-":
                            if align_idx < len(res_rmsd):
                                di = res_rmsd[align_idx]
                                score = math.exp(-(di ** 2) / (2 * sigma ** 2))
                                if res_idx < len(res_scores):
                                    res_scores[res_idx] += score
                                align_idx += 1
                        res_idx += 1

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    return res_scores, num_files


def write_results(output_file: str, scores: List[float], num_files: int) -> None:
    try:
        if num_files == 0:
            print("Warning: No valid alignment files for score calculation, outputting all-zero results.")
            normalized_scores = [0.0] * len(scores)
        else:
            scores_arr = np.array(scores).reshape(-1, 1)
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            normalized_scores = qt.fit_transform(scores_arr).flatten()

        with open(output_file, 'w') as outfile:
            for i, score in enumerate(normalized_scores):
                outfile.write(f"{i}\t{score:.3f}\n")

        print(f"Scores normalized to [0,1] via quantile normalization and written to {output_file}")

    except IOError as e:
        print(f"Failed to write result file: {e}")
    except Exception as e:
        print(f"Error during quantile normalization: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process sequence alignment and related score calculation')
    parser.add_argument('--mode', type=str, choices=['m8', 'hmmer'], required=True, help='Input file format mode')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path (.m8 or .hmmer)')
    parser.add_argument('--RMSD_path', type=str, required=True, help='RMSD path')
    parser.add_argument('--seq_path', type=str, required=True, help='Target sequence FASTA file path')
    parser.add_argument('--msta_outpath', type=str, required=True, help='msta result output file path')
    parser.add_argument('--afdb_pdb_dir', type=str, required=True, help='AFDB50_pdb directory path')
    parser.add_argument('--nwalign_path', type=str, required=True, help='NWalign executable path')

    args = parser.parse_args()

    try:
        target_seq = read_fasta_sequence(args.seq_path)
        print(f"Successfully read target sequence, length: {len(target_seq)}")
    except IOError as e:
        print(f"Error: Cannot read target FASTA file {args.seq_path}: {e}")
        sys.exit(1)

    protein_id = find_existing_pdb_protein(args.input_file, args.afdb_pdb_dir, args.mode)
    if not protein_id:
        print("Program terminated: Cannot determine reference protein.")
        sys.exit(1)

    dir1 = protein_id[0:2]
    dir2 = protein_id[2:4]
    dir3 = protein_id[4:6]
    pdb_file = f"{args.afdb_pdb_dir}/{dir1}/{dir2}/{dir3}/{protein_id}.pdb"
    try:
        query_seq = extract_sequence_from_pdb(pdb_file)
        print(f"Successfully read query sequence from PDB file, length: {len(query_seq)}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    query_fasta = os.path.join(args.RMSD_path, "query_temp.fasta")
    try:
        save_sequence_to_fasta(query_seq, query_fasta, f">{protein_id}")
    except IOError as e:
        print(f"Error: Cannot save query sequence to temporary file: {e}")
        sys.exit(1)

    try:
        aligned_query, aligned_target = run_nwalign(query_fasta, args.seq_path, args.nwalign_path)
        print("NWalign alignment result:")
        print(aligned_query)
        print("".join(["|" if a == b else " " for a, b in zip(aligned_query, aligned_target)]))
        print(aligned_target)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    temp_output = f"{args.RMSD_path}/temp_tmscore"
    if not os.path.exists(args.RMSD_path):
        print(f"Error: RMSD directory does not exist {args.RMSD_path}")
        sys.exit(1)
    process_rmsd_files(args.RMSD_path, temp_output)

    query_scores, num_files = calculate_alignment_scores(args.RMSD_path, len(query_seq))
    print(f"Processing complete, used {num_files} valid alignment files")

    target_scores = map_scores_to_target(query_scores, aligned_query, aligned_target)

    write_results(args.msta_outpath, target_scores, num_files)
    print(f"Normalized results written to {args.msta_outpath}")


if __name__ == "__main__":
    main()