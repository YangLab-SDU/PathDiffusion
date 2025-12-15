#!/usr/bin/env python3
import os
import sys
import csv
import argparse


def parse_fasta(fasta_file):
    try:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error: Cannot read file {fasta_file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        return None, None

    seq_id = lines[0].strip()[1:] if lines[0].startswith('>') else ''
    sequence = ''.join(line.strip() for line in lines[1:])

    return seq_id, sequence


def main():
    parser = argparse.ArgumentParser(description='Generate sequence info in CSV format from FASTA file')

    parser.add_argument('--fasta_file', required=True,
                        help='Input FASTA file path')
    parser.add_argument('--chain_name', required=True,
                        help='Chain name')
    parser.add_argument('--train_val_test', required=True,
                        help='Dataset split type (train/val/test)')

    parser.add_argument('--csv_file', default='test_data.csv',
                        help='Output CSV file path (default: test_data.csv)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing CSV file (default: append)')

    args = parser.parse_args()

    if not os.path.isfile(args.fasta_file):
        print(f"Error: File '{args.fasta_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    seq_id, sequence = parse_fasta(args.fasta_file)
    seqlen = len(sequence)

    try:
        mode = 'w' if args.overwrite else 'a'
        file_exists = os.path.isfile(args.csv_file)

        with open(args.csv_file, mode, newline='') as csvfile:
            fieldnames = ['chain_name', 'train_val_test', 'seqlen', 'seqres']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists or args.overwrite:
                writer.writeheader()

            writer.writerow({
                'chain_name': args.chain_name,
                'train_val_test': args.train_val_test,
                'seqlen': seqlen,
                'seqres': sequence
            })

        action = "overwritten" if args.overwrite else "appended to"
        print(f"Successfully {action} {args.csv_file}")
    except Exception as e:
        print(f"Error: Cannot write to CSV file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()