#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Error: Please provide target_path and BaseDir as arguments"
    echo "Usage: $0 /path/to/target_directory /path/to/base_directory"
    exit 1
fi

target_path="$1"
base_dir="$2"

if [ ! -d "$base_dir" ]; then
    echo "Error: BaseDir does not exist: $base_dir"
    exit 1
fi

output_pdb_dir="$target_path/stage2"
tran_files_dir="$target_path/tran_files"
tran_out_pdb_dir="$target_path/tran_out_pdb"
tmscore_path="$base_dir/model/TMscore"

if [ ! -f "$tmscore_path" ]; then
    echo "Error: TMscore tool not found: $tmscore_path"
    exit 1
fi

mkdir -p "$tran_files_dir" "$tran_out_pdb_dir" || { echo "Failed to create directories"; exit 1; }

cleanup() {
    echo "Cleaning up temporary files..."
    if [ -d "$tran_files_dir" ]; then
        rm -rf "$tran_files_dir"
        echo "Removed temporary directory: $tran_files_dir"
    else
        echo "Temporary directory does not exist, no cleanup needed"
    fi
}

trap cleanup EXIT SIGINT SIGTERM

pdb_files=($(ls "$output_pdb_dir"/Step_*_sample.pdb 2>/dev/null | sort -V -k 1.5))
file_count=${#pdb_files[@]}

if [ "$file_count" -eq 0 ]; then
    echo "Error: No Step_X_sample.pdb files found in $output_pdb_dir"
    exit 1
fi

echo "Found $file_count PDB files"

first_file="${pdb_files[0]}"
first_index=$(basename "$first_file" | sed -n 's/^Step_\([0-9]*\)_sample\.pdb$/\1/p')

if [ -z "$first_index" ]; then
    echo "Error: Could not parse index from filename: $first_file"
    exit 1
fi

cp "$first_file" "$tran_out_pdb_dir/model_${first_index}.pdb" || { echo "Failed to copy initial file"; exit 1; }

final_state="${pdb_files[-1]}"

for ((i=0; i < file_count-1; i++)); do
    file1="${pdb_files[$i]}"
    file2="${pdb_files[$i+1]}"
    
    index2=$(basename "$file2" | sed -n 's/^Step_\([0-9]*\)_sample\.pdb$/\1/p')

    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
        echo "Warning: File $file1 or $file2 does not exist, skipping"
        continue
    fi
    
    "$tmscore_path" "$final_state" "$file2" > "$tran_files_dir/tmscore_tran" || { echo "TMscore (final vs current) failed"; continue; }
    "$tmscore_path" "$file1" "$file2" > "$tran_files_dir/tmscore_qianhou" || { echo "TMscore (prev vs current) failed"; continue; }

    python "$base_dir/conformation_align.py" "$target_path" "$index2" || { echo "conformation_align.py failed"; continue; }
done

processed_pdbs=($(ls "$tran_out_pdb_dir"/model_*.pdb 2>/dev/null | sort -V))
processed_count=${#processed_pdbs[@]}

> "$target_path/model_movie_sample0.pdb"

for pdb_file in "${processed_pdbs[@]}"; do
    idx=$(basename "$pdb_file" | sed -n 's/^model_\([0-9]*\)\.pdb$/\1/p')
    python "$base_dir/conformation_ensemble.py" "$tran_out_pdb_dir" "$idx" "$target_path" || { echo "conformation_ensemble.py failed for model $idx"; continue; }
done