#!/usr/bin/env python3
"""Make ESM embedding from LM or trunk layers

----------------
[License]
SPDX-License-Identifier: Apache-2.0
----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================

from argparse import ArgumentParser

import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch

from .esmfold import ESMFold

# =============================================================================
# Constants
# =============================================================================

ESMFOLD_CKPT = None


# =============================================================================
# Functions
# =============================================================================


def canonize_chain_name(series):
    """Canonize all chain names to [pdb_id]_[chain_id]"""
    return series.str.replace(".pdb", "").str.replace(".", "_")


# =============================================================================
# Main
# =============================================================================


def main(args):
    """
    Featurizes amino acids into node and edge embeddings.
    Embeddings are stored as npy files in the target directory:
        [chain_name].lm_node_repr.npy
        [chain_name].trunk_node_repr.npy
        [chain_name].trunk_edge_repr.npy
    """
    # 加载输入数据
    df = pd.read_csv(args.input_csv_path)

    # 明确使用"chain_name"和"seqres"列（适配修改后的输入文件）
    chain_name_col = "chain_name"
    seqres_col = "seqres"

    # 标准化链名称格式
    df[chain_name_col] = canonize_chain_name(df[chain_name_col])
    # 仅保留必要的列
    chain_df = df[[chain_name_col, seqres_col]]

    # 检查输出目录，创建不存在的目录
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 索引文件路径（无recycle后缀）
    index_file = output_root.joinpath("seqres_and_index.csv")

    # 跳过已存在的序列
    if index_file.exists():
        existed = pd.read_csv(index_file)
        chain_df_existed = chain_df[seqres_col].isin(existed["seqres"])
        if args.worker_id == 0:
            print(
                f"\t{chain_df_existed.sum():,}/{len(chain_df_existed):,} embeddings already existed. Skip them for generation"
            )
        chain_df = chain_df[~chain_df_existed]
        existed_count = chain_df_existed.sum()
    else:
        if args.worker_id == 0:
            # 初始化索引文件
            with open(index_file, "w") as handle:
                handle.write("seqres,index\n")
        existed_count = 0

    # 移除重复序列
    chain_df = chain_df.drop_duplicates(subset=seqres_col)

    # 分配当前worker的任务
    chain_df = chain_df.iloc[args.worker_id:: args.num_workers]
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] generating embeddings for {len(chain_df)} chains"
    )

    # 加载ESMFold模型
    esm_enc = ESMFold(ckpt_fpath=args.esm_ckpt_fpath).to("cuda:0")
    assert esm_enc is not None, "Failed to load ESM model"

    # 统计变量
    failed = 0
    oom_errors = 0
    newly_completed = 0

    # 计算总批次
    if len(chain_df) % args.batch_size == 0:
        total_batches = len(chain_df) // args.batch_size
    else:
        total_batches = len(chain_df) // args.batch_size + 1

    # 记录当前worker生成的索引
    index_f = open(output_root / f"seqres_and_index.worker{args.worker_id}.csv", "w")

    # 批量处理序列
    for batch_ix in tqdm(range(total_batches), total=total_batches):
        batch_df = chain_df[
                   batch_ix * args.batch_size: (batch_ix + 1) * args.batch_size
                   ]
        if len(batch_df) == 0:
            break

        seqres_batch = list(batch_df[seqres_col])
        chain_name_batch = list(batch_df[chain_name_col])

        try:
            # 生成ESM嵌入
            lm_node_repr, trunk_node_repr, trunk_edge_repr, mask = esm_enc.infer(
                sequences=seqres_batch, num_recycles=args.num_recycles
            )
        except Exception as e:
            # 处理OOM错误
            if "out of memory" in str(e).lower():
                print(
                    f"[Worker {args.worker_id}/{args.num_workers}] CUDA OOM, skipping batch {batch_ix}: {', '.join(chain_name_batch)}",
                    flush=True,
                )
                torch.cuda.empty_cache()
                oom_errors += 1
                continue
            # 处理其他错误
            print(
                f"[Worker {args.worker_id}/{args.num_workers}] error processing batch {batch_ix} ({', '.join(chain_name_batch)}): {e}",
                flush=True,
            )
            failed += 1
            continue

        # 转换为numpy数组（CPU）
        lm_node_repr = lm_node_repr.cpu().numpy()
        trunk_node_repr = trunk_node_repr.cpu().numpy()
        trunk_edge_repr = trunk_edge_repr.cpu().numpy()
        mask = mask.cpu().numpy().astype(bool)

        # 逐个处理链
        for ix, (chain_name, seqres) in enumerate(zip(chain_name_batch, seqres_batch)):
            # 应用mask过滤有效序列（去除padding）
            lm_node_repr_ = lm_node_repr[ix, mask[ix], ...]
            trunk_node_repr_ = trunk_node_repr[ix, mask[ix], ...]
            trunk_edge_repr_ = trunk_edge_repr[ix, mask[ix], ...]

            # 验证序列长度匹配
            assert lm_node_repr_.shape[0] == len(
                seqres
            ), f"{chain_name}: length mismatch: {lm_node_repr_.shape[0]} vs {len(seqres)}"

            # 输出文件路径（直接保存到output_root，无二级子目录）
            prefix = str(output_root / chain_name)

            # 处理文件名冲突（添加随机后缀）
            if Path(f"{prefix}.lm_node_repr.npy").exists():
                suffix = ""
                while Path(f"{prefix}{suffix}.lm_node_repr.npy").exists():
                    suffix = "_" + "".join(random.choice(string.hexdigits) for _ in range(2)).lower()
                prefix = str(output_root / f"{chain_name}{suffix}")
                chain_name = f"{chain_name}{suffix}"

            # 再次检查冲突（避免极端情况）
            if Path(f"{prefix}.lm_node_repr.npy").exists():
                raise ValueError(f"{prefix}.lm_node_repr.npy already exists")

            # 保存嵌入文件（无recycle后缀）
            np.save(f"{prefix}.lm_node_repr.npy", lm_node_repr_)
            np.save(f"{prefix}.trunk_node_repr.npy", trunk_node_repr_)
            np.save(f"{prefix}.trunk_edge_repr.npy", trunk_edge_repr_)

            # 记录到索引文件（仅保存chain_name，因直接在output_root下）
            index_f.write(f"{seqres},{chain_name}\n")

        newly_completed += len(batch_df)

    # 输出统计信息
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] Summary: Existing={existed_count:,}, Newly generated={newly_completed:,}, Failed={failed:,} (OOM errors: {oom_errors:,})"
    )
    index_f.close()


if __name__ == "__main__":
    # 命令行参数解析
    parser = ArgumentParser()
    parser.add_argument("--input-csv-path", type=str, required=True,
                        help="Path to input CSV with 'chain_name' and 'seqres' columns")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output embeddings (e.g., .../ESM_repr/npy)")
    parser.add_argument("--esm-ckpt-fpath", default=None,
                        help="Path to ESMFold checkpoint file")
    parser.add_argument("--num-recycles", type=int, default=3,
                        help="Number of recycles for ESMFold inference")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per worker")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of workers for parallel processing")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="ID of current worker (0-based)")

    args, _ = parser.parse_known_args()

    main(args)