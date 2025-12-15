#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    parser = argparse.ArgumentParser(description='Run JackHMMER for sequence search')
    parser.add_argument('--query_file', required=True, help='Query FASTA file path')
    parser.add_argument('--target_db', required=True, help='Target database path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--tmp_dir_base', required=True, help='Temporary file base directory')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of concurrent threads')
    parser.add_argument('--iterations', type=int, default=3, help='JackHMMER iterations (corresponds to -N)')
    parser.add_argument('--evalue', type=float, default=1.0, help='E-value threshold (corresponds to -E)')
    parser.add_argument('--cpu', type=int, default=10, help='Number of CPU threads to use (corresponds to --cpu)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir_base, exist_ok=True)

    query_file = args.query_file
    if not os.path.exists(query_file):
        print(f"Error: Query file {query_file} does not exist. Exiting!")
        return

    if not query_file.endswith(".fasta"):
        print(f"Warning: Query file {query_file} is not a .fasta file, which may cause errors")

    def run_jackhmmer(file_path):
        file_name = os.path.basename(file_path)

        output_path = os.path.join(args.output_dir, "msa.hmmer")

        tmp_dir = os.path.join(args.tmp_dir_base, os.path.splitext(file_name)[0])
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_log = os.path.join(tmp_dir, "jackhmmer_run.log")

        if not os.path.exists(args.target_db):
            return f"❌ Failed: {file_name} → Target database does not exist: {args.target_db}"

        cmd = [
            "jackhmmer",
            "-o", output_path,
            "-N", str(args.iterations),
            "-E", str(args.evalue),
            "--cpu", str(args.cpu),
            file_path,
            args.target_db
        ]

        print(f"Executing command: {' '.join(cmd)}")

        try:
            print(f"\nProcessing {file_name} → Output to {output_path}")
            with open(tmp_log, "w", encoding="utf-8") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            if result.returncode != 0:
                error_msg = "Unknown error"
                if os.path.exists(tmp_log):
                    with open(tmp_log, "r", encoding="utf-8") as f:
                        error_msg = f.read(1000)
                return f"❌ Failed: {file_name} → Command returned non-zero status code {result.returncode}\nError log: {error_msg}\nFull log at: {tmp_log}"

            shutil.rmtree(tmp_dir)
            return f"✅ Success: {file_name} → Output file saved to {output_path}"

        except Exception as e:
            return f"❌ Exception: {file_name} → Exception info: {str(e)}\nFull log at: {tmp_log}"

    print(f"\n=== Processing Started ===")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_jackhmmer, query_file): query_file}

        for future in as_completed(futures):
            result = future.result()
            print(result)

    print(f"\n=== All Tasks Completed ===")
    print(f"Final results directory: {args.output_dir}")


if __name__ == "__main__":
    main()