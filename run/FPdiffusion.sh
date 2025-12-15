#!/bin/bash

# =============================================================================
# Configuration & Path Setup
# =============================================================================

# 1. Get the absolute path of the directory where this script is located
# This replaces: BaseDir=/storage/.../run
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. Define Project Root (Assuming this script is in project_root/run/)
# This replaces: SampleDir=/storage/.../FPdiffusion_ensemble
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. Input Arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <Output_Directory> <Protein_Name> [Dataset_Dir] [Model_Dir]"
    echo "Example: $0 ./results 1abc /path/to/dataset /path/to/checkpoints"
    exit 1
fi

OUTPUT_ROOT=$1
ProteinName=$2

# 4. External Data Paths (Allow override via arguments, otherwise use defaults)
# Users should provide these, or you can set default relative paths
DATASET_DIR=${3:-"$PROJECT_ROOT/dataset"}   # Default: dataset folder inside project
TOOL_DIR=${4:-"$PROJECT_ROOT/model_weights"} # Default: model_weights folder inside project

# 5. Derived Paths
TestDir="$OUTPUT_ROOT/$ProteinName"
Threads=10

# Print Configuration
echo "=========================================================="
echo "Running FPdiffusion Pipeline"
echo "Project Root : $PROJECT_ROOT"
echo "Script Dir   : $SCRIPT_DIR"
echo "Output Dir   : $TestDir"
echo "Dataset Dir  : $DATASET_DIR"
echo "Model Dir    : $TOOL_DIR"
echo "Protein Name : $ProteinName"
echo "=========================================================="

# Check if critical directories exist
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found at $DATASET_DIR"
    echo "Please provide the correct path as the 3rd argument."
    exit 1
fi
if [ ! -d "$TOOL_DIR" ]; then
    echo "Error: Model directory not found at $TOOL_DIR"
    echo "Please provide the correct path as the 4th argument."
    exit 1
fi

# =============================================================================
# Step 1: Run MMseqs2 Search
# =============================================================================
echo ">>> [Step 1] Running MMseqs2 search..."

mkdir -p "$TestDir/msa"
MSA_M8_FILE="$TestDir/msa/msa.m8"

if [ ! -f "$MSA_M8_FILE" ]; then
    python "$SCRIPT_DIR/make_msa_mmseq.py" \
        --query_dir "$TestDir" \
        --target_db "$DATASET_DIR/UniRef/AFDB50_fasta/target/target_afdb50" \
        --output_dir "$TestDir/msa" \
        --tmp_dir_base "$TestDir/msa" \
        --max_workers 1 \
        --threads_per_task $Threads \
        --evalue 1 \
        --sensitivity 5.7 \
        --max_seqs 10000

    if [ ! -f "$MSA_M8_FILE" ]; then
        echo "Warning: MMseqs2 did not generate msa.m8 file."
        SEQ_COUNT=0
    else
        SEQ_COUNT=$(wc -l < "$MSA_M8_FILE")
        echo "MMseqs2 search completed. Number of sequences found: $SEQ_COUNT"
    fi
else
    SEQ_COUNT=$(wc -l < "$MSA_M8_FILE")
    echo "msa.m8 already exists. Number of sequences: $SEQ_COUNT"
fi

# =============================================================================
# Step 2: Logic Check and MSA Supplementation
# =============================================================================
USE_HMMER_PIPELINE=false

if [ "$SEQ_COUNT" -lt 10 ]; then
    echo ">>> Sequence count ($SEQ_COUNT) is less than 10. Switching to JackHMMER..."
    USE_HMMER_PIPELINE=true
else
    echo ">>> Sequence count ($SEQ_COUNT) is sufficient (>=10). Continuing with MMseqs2 results..."
fi

# =============================================================================
# Step 3: Generate MSTA (Branch Processing)
# =============================================================================
mkdir -p "$TestDir/msta"

RMSD_DIR="$TestDir/msta/RMSD"
MSTA_FILE="$TestDir/msta/${ProteinName}.msta"

if [ "$USE_HMMER_PIPELINE" = true ]; then
    # ---------------- [Branch B] JackHMMER Pipeline ----------------
    MSA_HMMER_FILE="$TestDir/msa/msa.hmmer"
    
    if [ ! -f "$MSA_HMMER_FILE" ]; then
        python3 "$SCRIPT_DIR/make_msa_hmmer.py" \
            --query_file "$TestDir/seq.fasta" \
            --target_db "$DATASET_DIR/AFDB/afdb50.fasta" \
            --output_dir "$TestDir/msa" \
            --tmp_dir_base "$TestDir/msa" \
            --iterations 3 \
            --evalue 1 \
            --cpu $Threads
            
        if [ ! -f "$MSA_HMMER_FILE" ]; then
            echo "Error: JackHMMER generation failed!"
            exit 1
        fi
    fi

    if [ ! -f "$MSTA_FILE" ]; then
        python "$SCRIPT_DIR/make_msta_pdbalign.py" \
            --mode hmmer \
            --input_file "$MSA_HMMER_FILE" \
            --threads_per_task $Threads \
            --afdb_pdb_dir "$DATASET_DIR/AFDB/AFDB50_pdb" \
            --tmalign_path "$TOOL_DIR/TMalign_rmsd" \
            --output_dir "$RMSD_DIR"

        python "$SCRIPT_DIR/make_msta_Fscore_Quantile.py" \
            --mode hmmer \
            --input_file "$MSA_HMMER_FILE" \
            --RMSD_path "$RMSD_DIR" \
            --seq_path "$TestDir/seq.fasta" \
            --msta_outpath "$MSTA_FILE" \
            --afdb_pdb_dir "$DATASET_DIR/AFDB/AFDB50_pdb" \
            --nwalign_path "$TOOL_DIR/NWalign"
    fi

else
    # ---------------- [Branch A] MMseqs2 Pipeline ----------------
    if [ ! -f "$MSTA_FILE" ]; then
        python "$SCRIPT_DIR/make_msta_pdbalign.py" \
            --mode m8 \
            --input_file "$MSA_M8_FILE" \
            --threads_per_task $Threads \
            --afdb_pdb_dir "$DATASET_DIR/AFDB/AFDB50_pdb" \
            --tmalign_path "$TOOL_DIR/TMalign_rmsd" \
            --output_dir "$RMSD_DIR"

        python "$SCRIPT_DIR/make_msta_Fscore_Quantile.py" \
            --mode m8 \
            --input_file "$MSA_M8_FILE" \
            --RMSD_path "$RMSD_DIR" \
            --seq_path "$TestDir/seq.fasta" \
            --msta_outpath "$MSTA_FILE" \
            --afdb_pdb_dir "$DATASET_DIR/AFDB/AFDB50_pdb" \
            --nwalign_path "$TOOL_DIR/NWalign"
    fi
fi

if [ ! -f "$MSTA_FILE" ]; then
    echo "Error: MSTA file generation failed!"
    exit 1
fi
echo "MSTA generation successful!"

# =============================================================================
# Step 4: General Data Preparation (CSV)
# =============================================================================
CSV_FILE="$TestDir/test_data.csv"
if [ ! -f "$CSV_FILE" ]; then
    python "$SCRIPT_DIR/make_data_csv.py" \
        --fasta_file "$TestDir/seq.fasta" \
        --chain_name "$ProteinName" \
        --train_val_test test \
        --csv_file "$CSV_FILE" \
        --overwrite

    if [ ! -f "$CSV_FILE" ]; then
        echo "Error: CSV file generation failed!"
        exit 1
    fi
else
    echo "test_data.csv already exists, skipping."
fi

# =============================================================================
# Step 5: Generate ESM_repr
# =============================================================================
ESM_REPR_CSV="$TestDir/seqres_and_index.csv"
if [ ! -f "$ESM_REPR_CSV" ]; then
    # We use a subshell or explicit path to run python module
    # Ensure PYTHONPATH includes PROJECT_ROOT so imports work
    export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
    
    mkdir -p "$TestDir/ESM_repr"
    NumGpu=1

    for GpuId in $(seq 0 $(($NumGpu-1))); do
        CUDA_VISIBLE_DEVICES=$GpuId python3 -m run.pretrain_repr.esmfold.ESM_repr \
            --input-csv-path "$CSV_FILE" \
            --output-dir "$TestDir/ESM_repr" \
            --esm-ckpt-fpath "$TOOL_DIR/esmfold_3B_v1.pt" \
            --num-recycles 3 \
            --batch-size 1 \
            --num-workers $NumGpu \
            --worker-id $GpuId &
    done

    wait

    # Note: Using * globbing, make sure files exist
    cat "$TestDir"/ESM_repr/seqres_and_index.worker*.csv >> "$TestDir/ESM_repr/seqres_and_index.csv"
    mv "$TestDir/ESM_repr/seqres_and_index.csv" "$ESM_REPR_CSV"
    rm "$TestDir"/ESM_repr/seqres_and_index.worker*.csv

    if [ ! -f "$ESM_REPR_CSV" ]; then
        echo "Error: ESM_repr generation failed!"
        exit 1
    fi
    echo "ESM_repr generation successful!"
fi

# =============================================================================
# Step 6: Folding Pathway Sampling
# =============================================================================
# We need to run eval.py which is likely in PROJECT_ROOT.
# We change directory to PROJECT_ROOT to ensure config paths work as expected
# or ensure python paths are correct.

STAGE1_DIR="$TestDir/stage1"
if [ ! -d "$STAGE1_DIR" ] || [ -z "$(ls -A "$STAGE1_DIR")" ]; then
    
    echo "Starting Stage 1..."
    cd "$PROJECT_ROOT" || exit
    
    python3 eval.py \
        sampling=cfg_inference \
        data.repr_loader.data_root="$TestDir" \
        paths.guidance.cond_ckpt="$TOOL_DIR/cond_model.ckpt" \
        paths.guidance.uncond_ckpt="$TOOL_DIR/uncond_model.ckpt" \
        paths.output_dir="$TestDir" \
        data.dataset.test_gen_dataset.csv_path="$CSV_FILE" \
        data.dataset.test_gen_dataset.num_samples=1 \
        model.score_network.cfg.clsfree_guidance_strength=1.0 \
        model.score_network.msta_dir="$TestDir/msta" \
        model.stage=1

    if [ ! -d "$STAGE1_DIR" ] || [ -z "$(ls -A "$STAGE1_DIR")" ]; then
        echo "Error: Stage1 sampling failed!"
        exit 1
    fi
    echo "Stage1 sampling successful!"
fi

STAGE2_DIR="$TestDir/stage2"
if [ ! -d "$STAGE2_DIR" ] || [ -z "$(ls -A "$STAGE2_DIR")" ]; then
    
    echo "Starting Stage 2..."
    cd "$PROJECT_ROOT" || exit
    
    python3 eval.py \
        sampling=cfg_inference \
        data.repr_loader.data_root="$TestDir" \
        paths.guidance.cond_ckpt="$TOOL_DIR/cond_model.ckpt" \
        paths.guidance.uncond_ckpt="$TOOL_DIR/uncond_model.ckpt" \
        paths.output_dir="$TestDir" \
        data.dataset.test_gen_dataset.csv_path="$CSV_FILE" \
        data.dataset.test_gen_dataset.num_samples=2 \
        model.score_network.cfg.clsfree_guidance_strength=0.95 \
        model.score_network.cfg.starting_steps=150 \
        model.score_network.cfg.final_steps=155 \
        model.score_network.msta_dir="$TestDir/msta" \
        model.stage=2

    if [ ! -d "$STAGE2_DIR" ] || [ -z "$(ls -A "$STAGE2_DIR")" ]; then
        echo "Error: Stage2 sampling failed!"
        exit 1
    fi
    echo "Folding pathway sampling successful!"
fi

# =============================================================================
# Step 7: Generate Pathway Movie
# =============================================================================
# Assuming make_model_movie.sh is also in SCRIPT_DIR
bash "$SCRIPT_DIR/make_model_movie.sh" "$TestDir" "$SCRIPT_DIR"

echo "All processing steps completed successfully!"