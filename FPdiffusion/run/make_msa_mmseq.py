#!/usr/bin/env python

import os
import subprocess
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    parser = argparse.ArgumentParser(description='Run mmseqs2 for sequence search')
    parser.add_argument('--query_dir', required=True, help='Query FASTA directory')
    parser.add_argument('--target_db', required=True, help='Target database path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--tmp_dir_base', required=True, help='Temporary file base directory')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of concurrent threads')
    parser.add_argument('--threads_per_task', type=int, default=10, help='Number of threads per task')
    parser.add_argument('--evalue', type=float, default=1.0, help='E-value threshold')
    parser.add_argument('--sensitivity', type=float, default=6.5, help='Sensitivity threshold')
    parser.add_argument('--max_seqs', type=int, default=10000, help='Maximum number of sequences')

    args = parser.parse_args()
    print(f"----------------args")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir_base, exist_ok=True)

    fasta_files = [f for f in os.listdir(args.query_dir) if f.endswith(".fasta")]

    print(f"----------------fasta_files: {fasta_files}")

    def run_mmseqs(fasta_file):
        query_path = os.path.join(args.query_dir, fasta_file)
        output_path = os.path.join(args.output_dir, "msa.m8")
        tmp_dir = os.path.join(args.tmp_dir_base, fasta_file.replace(".fasta", ""))

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        cmd = [
            "mmseqs", "easy-search",
            query_path,
            args.target_db,
            output_path,
            tmp_dir,
            "-e", str(args.evalue),
            "-s", str(args.sensitivity),
            "--threads", str(args.threads_per_task),
            "--max-seqs", str(args.max_seqs)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            shutil.rmtree(tmp_dir)
            return f"{fasta_file} Success -> {output_path}"
        except subprocess.CalledProcessError as e:
            return f"{fasta_file} Failed: {e.stderr}"

    print(f"Starting processing of {len(fasta_files)} FASTA files")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_mmseqs, fasta): fasta for fasta in fasta_files}

        for future in as_completed(futures):
            result = future.result()
            print(result)

    print("All tasks completed")


if __name__ == "__main__":
    main()