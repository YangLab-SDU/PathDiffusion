# FPdiffusion
## Installation
```bash
# clone project
git clone https://github.com/YangLab-SDU/FPdiffusion.git
cd FPdiffusion

# create conda virtual environment
conda env create -f env.yml
conda activate FPdiffusion

# install openfold
git clone https://github.com/aqlaboratory/openfold.git
pip install -e openfold

# install openfold
conda install -c conda-forge -c bioconda mmseqs2
```

## Training
We use Hydra for configuration management. All configuration files are located in the settings/ directory.

**1. Conditional Training (Fold-based)**:
```bash
python train.py \
    --config-name cond_model \
    task_name=cond_train \
    data.train_batch_size=1 \
    paths.output_dir="./train_model"
```
The detailed training configuration can be found in `settings/cond_model.yaml`.

**2. Unconditional Training (Disorder-based)**:
```bash
python train.py \
    --config-name uncond_model \
    task_name=uncond_train \
    paths.output_dir="./train_model"
```
The detailed training configuration can be found in `settings/uncond_model.yaml`.

## Inference (Sampling)
Use eval.py to generate protein structures. The inference pipeline typically uses Classifier-Free Guidance (CFG) combining both conditional and unconditional checkpoints.
### Basic Command
```bash
python eval.py \
    experiment=clsfree_guide \
    sampling=cfg_inference \
    model.stage=inference \
    paths.output_dir="./output/inference_result" \
    paths.guidance.cond_ckpt="/path/to/cond_model.ckpt" \
    paths.guidance.uncond_ckpt="/path/to/uncond_model.ckpt" \
    model.score_network.msta_dir="/path/to/msta_dir" \
    data.dataset.test_gen_dataset.csv_path="/path/to/test_data.csv"
```


