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

### Training
FPdiffusion consists of a **sequence-conditional model** and an **unconditional model**.


To train the **conditional model**:
**Conditional Training (Fold-based)**:
```bash
python train.py \
    --config-name cond_model \
    task_name=cond_train \
    data.train_batch_size=1 \
    paths.output_dir="./train_model"
```
The detailed training configuration can be found in `configs/experiment/full_atom.yaml`.

**Conditional Training (Fold-based)**:
```bash
python train.py \
    --config-name uncond_model \
    task_name=uncond_train \
    data.train_batch_size=1 \
    paths.output_dir="./train_model"
```
The detailed training configuration can be found in `configs/experiment/uncond.yaml`.

