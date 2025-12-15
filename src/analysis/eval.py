"""Evaluation function during for val_gen training

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from src.analysis import struct_align
from src.utils import hydra_utils

logger = hydra_utils.get_pylogger(__name__)

# =============================================================================
# Functions
# =============================================================================

def eval_gen_conf(
    output_root,
    csv_fpath,
    ref_root,
    num_samples: Optional[int] = None,
    n_proc=1,
    **align_kwargs
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate generated conformations
    """
    output_root = Path(output_root)
    ref_root = Path(ref_root)

    if not os.path.exists(csv_fpath):
        logger.warning(f"Metadata CSV not found: {csv_fpath}")
        return {}, {}

    df = pd.read_csv(csv_fpath)
    if 'chain_name' not in df.columns:
        if df.index.name == 'chain_name':
            df = df.reset_index()
        else:
            logger.warning("CSV must contain 'chain_name' column")
            return {}, {}

    scores = []

    unique_chains = df['chain_name'].unique()

    logger.info(f"Evaluating {len(unique_chains)} chains in {output_root}")

    for chain_name in tqdm(unique_chains, desc="Eval"):
        ref_path = None
        for suffix in ['.pdb', '.ent']:
            p = ref_root / f"{chain_name}{suffix}"
            if p.exists():
                ref_path = p
                break

        if ref_path is None:
            continue

        gen_files = list(output_root.rglob(f"*{chain_name}*.pdb"))

        if not gen_files:
            continue

        for gen_path in gen_files:
            try:
                if gen_path.resolve() == ref_path.resolve():
                    continue

                res = struct_align.compute_align_scores(
                    test_struct=str(gen_path),
                    ref_struct=str(ref_path),
                    compute_lddt=False,
                    show_error=False,
                    **align_kwargs
                )
                res['chain_name'] = chain_name
                scores.append(res)
            except Exception as e:
                pass

    log_stats = {}
    log_dists = {}

    if len(scores) > 0:
        results_df = pd.DataFrame(scores)

        for metric in ['TMscore', 'RMSD', 'GDT-TS']:
            if metric in results_df.columns:
                log_stats[f"{metric}_mean"] = results_df[metric].mean()
                log_stats[f"{metric}_median"] = results_df[metric].median()
                log_dists[metric] = results_df[metric].tolist()
    else:
        logger.warning("No valid alignments found during evaluation.")

    return log_stats, log_dists