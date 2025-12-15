"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""

from pandas.core.groupby import DataFrameGroupBy
from typing import Optional, Literal, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from openfold.data import data_transforms
from openfold.np import residue_constants as rc
from openfold.utils import rigid_utils as ru
from src.utils import hydra_utils
from omegaconf import DictConfig
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import torch
import random
import os

from src.utils import hydra_utils
logger = hydra_utils.get_pylogger(__name__)
import time


pdb_parser = PDBParser(QUIET=True)


class RCSBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,  # path to metadata csv file
        mode: Literal["train", "val", "test"],
        diffuser=None,
        data_dir: Optional[str] = None,
        msta_dir=None,
        csv_processor_cfg: Optional[DictConfig] = None,
        repr_loader: Optional[DictConfig] = None,
        dynamic_batching: bool = True,
        disorder: bool = True,
        **kwargs,
    ):
        self.mode = mode
        self.disorder = disorder
        if csv_path:
            self.df = self._process_csv(
                csv_path=csv_path, csv_processor_cfg=csv_processor_cfg
            )
        self.data_dir = data_dir
        self.msta_dir = msta_dir
        self.diffuser = diffuser
        self.repr_loader = repr_loader
        self.dynamic_batching = dynamic_batching


    def _process_csv(
        self,
        csv_path: str,
        csv_processor_cfg: Optional[DictConfig] = None,
    ) -> pd.DataFrame:
        if self.disorder:
            metadata_df = pd.read_csv(csv_path)
            if 'chain_name' in metadata_df.columns:
                metadata_df.set_index('chain_name', inplace=True)
            else:
                metadata_df.set_index(metadata_df.columns[0], inplace=True)
            filtered_index = metadata_df.index
            df = metadata_df.loc[filtered_index]

            logger.info(f"filtered_disorder_index = {filtered_index}")

            if 'train_val_test' in df.columns and self.mode in ['train', 'val', 'test']:
                df = df[df.train_val_test == self.mode]
                logger.info(f"After {self.mode} filtering: {len(df)} rows")

            self.group_col = None
        else:
            metadata_df = pd.read_csv(csv_path, index_col="chain_name")

            if csv_processor_cfg is not None:
                # filter csv
                min_seqlen = csv_processor_cfg.get("min_seqlen", 0)
                max_seqlen = csv_processor_cfg.get("max_seqlen", 1e4)
                earliest_release_date = csv_processor_cfg.get(
                    "earliest_release_date", "1900-01-01"
                )
                latest_release_date = csv_processor_cfg.get(
                    "latest_release_date", "2025-04-16"
                )
                max_coil_ratio = csv_processor_cfg.get("max_coil_ratio", 1.0)
                min_valid_frame_ratio = csv_processor_cfg.get("min_valid_frame_ratio", 0.5)

                filtered_index = metadata_df[
                    (metadata_df.seqlen >= min_seqlen)
                    & (metadata_df.seqlen <= max_seqlen)
                    & (metadata_df.release_date >= earliest_release_date)
                    & (metadata_df.release_date <= latest_release_date)
                    & (metadata_df.coil_ratio <= max_coil_ratio)
                    & (metadata_df.valid_frame_ratio >= min_valid_frame_ratio)
                ].index

                df = metadata_df.loc[filtered_index]
                logger.info(
                    f"Dataset mode: {self.mode}, number of rows in filtered DataFrame: {len(df)}"
                )

                # sequence-based clustering
                if csv_processor_cfg.get("groupby", None) is not None:
                    self.group_col = group_col = csv_processor_cfg.get("groupby")
                    df = df.groupby(group_col)
                    self.group_keys = list(df.groups.keys())
                    assert len(self.group_keys) == len(df)
                else:
                    self.group_col = None

        return df

    @property
    def name(self):
        return "rcsb"

    def __len__(self):
        return len(self.df)

    def load_pdb(self, pdb_path, chain_id, seqlen):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        struct = pdb_parser.get_structure("", pdb_path)

        if self.disorder:
            chain = struct[0]['A']
        else:
            chain = struct[0][
                chain_id
            ]  # each PDB file contains a single conformation, i.e., model 0

        # load atomic coordinates
        atom_coords = (
            np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        )  # (seqlen, 37, 3)
        for residue in chain:
            seq_idx = residue.id[1] - 1  # zero-based indexing
            for atom in residue:
                if atom.name in rc.atom_order.keys():
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord

        return atom_coords

    def _get_pdb_fpath(self, chain_name):
        pdb_id, model_id, chain_id = chain_name.split("_")
        return (
            os.path.join(
                self.data_dir,
                pdb_id[1:3],
                pdb_id,
                f"{pdb_id}_{model_id}_{chain_id}.pdb",
            ),
            chain_id,
        )

    def load_msta(self, fold_priority_file: str, seq_len: int) -> torch.Tensor:
        folding_priority = []
        with open(fold_priority_file, 'r') as f:
            for line in f:
                _, priority = line.strip().split()
                folding_priority.append(float(priority))

        folding_priority = torch.tensor(folding_priority)[:seq_len]
        return folding_priority

    def _get_msta_fpath(self, chain_name):
        pdb_id, model_id, chain_id = chain_name.split("_")
        return (
            os.path.join(
                self.msta_dir,
                pdb_id[1:3],
                pdb_id,
                f"{pdb_id}_{model_id}_{chain_id}.msta",
            )
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(self.df, pd.DataFrame):
            # sample from regular DataFrame
            row = self.df.iloc[idx]
        else:
            # sample from clusters
            assert isinstance(self.df, pd.core.groupby.DataFrameGroupBy)
            sampled_group = self.df.get_group(self.group_keys[idx])
            row = sampled_group.sample(n=1).iloc[0]

        seqres = row.seqres
        chain_name = row.name

        if self.disorder:
            pdb_fpath = os.path.join(self.data_dir, f"{chain_name}.pdb")
            atom_coords = self.load_pdb(pdb_fpath, chain_name, row.seqlen)
        else:
            pdb_fpath, chain_id = self._get_pdb_fpath(chain_name)
            atom_coords = self.load_pdb(pdb_fpath, chain_id, row.seqlen)

        fname = Path(pdb_fpath).stem
        aatype = torch.LongTensor(
            [rc.restype_order_with_x.get(res, 20) for res in seqres]
        )


        # remove center of mass
        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)
        all_atom_positions = torch.from_numpy(atom_coords)  # (seqlen, 37, 3)
        all_atom_mask = torch.all(
            ~torch.isnan(all_atom_positions), dim=-1
        )  # (seqlen, 37)

        torch.set_printoptions(threshold=torch.inf)

        all_atom_positions = torch.nan_to_num(
            all_atom_positions, 0.0
        )  # convert NaN to zero
        # ground truth backbone atomic coordinates
        gt_bb_coords = all_atom_positions[:, [0, 1, 2, 4], :]  # (seqlen, 4, 3)
        bb_coords_mask = all_atom_mask[:, [0, 1, 2, 4]]  # (seqlen, 4)

        # OpenFold data transformation
        openfold_feat_dict = {
            "aatype": aatype.long(),
            "all_atom_positions": all_atom_positions.double(),
            "all_atom_mask": all_atom_mask.double(),
        }

        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles()(
            openfold_feat_dict
        )

        # ground truth rigids
        rigids_0 = ru.Rigid.from_tensor_4x4(
            openfold_feat_dict["rigidgroups_gt_frames"]
        )[:, 0]

        rigids_mask = openfold_feat_dict["rigidgroups_gt_exists"][:, 0]
        assert rigids_mask.sum() == torch.all(all_atom_mask[:, [0, 1, 2]], dim=-1).sum()

        t_one = torch.ones(row.seqlen)
        base_t = max(0.01, random.random())

        if self.disorder:
            t = t_one - base_t
        else:
            msta_fpath = os.path.join(self.msta_dir, f"{chain_name}.msta")
            folding_priority = self.load_msta(msta_fpath, row.seqlen)
            t = t_one - base_t * (1 + 1 * folding_priority)

        t = torch.clamp(t, min=0.01)


        diffused_feat_dict = self.diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=rigids_mask.numpy(),
            as_tensor_7=False,
        )

        rigids_t = diffused_feat_dict["rigids_t"]

        for key, value in diffused_feat_dict.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.float64):
                diffused_feat_dict[key] = torch.tensor(value)

        data_dict = {
            "chain_name": chain_name,
            "fname": fname,
            "cluster_id": row[self.group_col] if self.group_col is not None else "NA",
            "aatype": aatype.long(),
            "rigids_0": rigids_0.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_t": rigids_t.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_mask": rigids_mask.float(),  # (seqlen,)
            "t": t.float(),  # (,)
            "rot_score": diffused_feat_dict["rot_score"].float(),  # (seqlen, 3)
            "trans_score": diffused_feat_dict["trans_score"].float(),  # (seqlen, 3)
            "rot_score_norm": diffused_feat_dict["rot_score_scaling"].float(),  # (,)
            "trans_score_norm": diffused_feat_dict[
                "trans_score_scaling"
            ].float(),  # (,)
            "gt_torsion_angles": openfold_feat_dict[
                "torsion_angles_sin_cos"
            ].float(),  # (seqlen,7,2)
            "torsion_angles_mask": openfold_feat_dict[
                "torsion_angles_mask"
            ].float(),  # (seqlen,7)
            "rigidgroups_gt_frames": openfold_feat_dict[
                "rigidgroups_gt_frames"
            ].float(),
            "rigidgroups_alt_gt_frames": openfold_feat_dict[
                "rigidgroups_alt_gt_frames"
            ].float(),
            "rigidgroups_gt_exists": openfold_feat_dict[
                "rigidgroups_gt_exists"
            ].float(),
            "atom14_gt_positions": openfold_feat_dict["atom14_gt_positions"].float(),
            "atom14_alt_gt_positions": openfold_feat_dict[
                "atom14_alt_gt_positions"
            ].float(),
            "atom14_atom_is_ambiguous": openfold_feat_dict[
                "atom14_atom_is_ambiguous"
            ].float(),
            "atom14_gt_exists": openfold_feat_dict["atom14_gt_exists"].float(),
            "atom14_alt_gt_exists": openfold_feat_dict["atom14_alt_gt_exists"].float(),
            "atom14_atom_exists": openfold_feat_dict["atom14_atom_exists"].float(),
            "gt_bb_coords": gt_bb_coords.float(),  # (seqlen, 4, 3)
            "bb_coords_mask": bb_coords_mask.float(),  # (seqlen, 4)
        }

        if self.repr_loader is not None:
            pretrained_repr = self.repr_loader.load(seqres=seqres)
            data_dict["pretrained_node_repr"] = pretrained_repr.get(
                "pretrained_node_repr", None
            )
            data_dict["pretrained_edge_repr"] = pretrained_repr.get(
                "pretrained_edge_repr", None
            )
        return data_dict

    def collate(self, batch_list):

        batch = {"gt_feat": {}}
        gt_feat_name = [
            "rot_score",
            "trans_score",
            "rot_score_norm",
            "trans_score_norm",
            "gt_torsion_angles",
            "torsion_angles_mask",
            "gt_bb_coords",
            "bb_coords_mask",
            "rigids_0",
            "atom14_gt_positions",
            "atom14_alt_gt_positions",
            "atom14_atom_is_ambiguous",
            "atom14_gt_exists",
            "atom14_alt_gt_exists",
            "atom14_atom_exists",
            "rigidgroups_gt_frames",
            "rigidgroups_alt_gt_frames",
            "rigidgroups_gt_exists",
        ]
        if "cluster_id" in batch_list[0].keys() and batch_list[0]["cluster_id"] != "NA":
            assert (
                np.array([feat_dict["cluster_id"] for feat_dict in batch_list]).std()
                == 0
            )

        if self.dynamic_batching:
            # determine batch size and truncate batch for large proteins
            if "rigids_t" in batch_list[0].keys():
                total_num_edges = np.cumsum(
                    [feat_dict["aatype"].size(0) ** 2 for feat_dict in batch_list]
                )
                batch_size = max(1, sum(total_num_edges < 200000))
                batch_list = batch_list[:batch_size]

        lengths = torch.tensor(
            [feat_dict["aatype"].shape[0] for feat_dict in batch_list],
            requires_grad=False,
        )
        max_L = max(lengths)
        padding_mask = torch.arange(max_L).expand(
            len(lengths), max_L
        ) < lengths.unsqueeze(1)

        for key, val in batch_list[0].items():
            if (val is None) or (key in []):
                continue
            if key in [
                "chain_name",
                "output_name",
                "dataset_name",
                "cluster_id",
                "fname",
            ]:
                batched_val = [feat_dict[key] for feat_dict in batch_list]
            elif val.dim() == 0:
                batched_val = torch.stack([feat_dict[key] for feat_dict in batch_list])
            elif (val.dim() < 3) or (key not in ["pretrained_edge_repr"]):
                batched_val = pad_sequence(
                    [feat_dict[key] for feat_dict in batch_list],
                    batch_first=True,
                    padding_value=0,
                )
            else:
                assert key == "pretrained_edge_repr"
                batched_val = []
                C = batch_list[0]["pretrained_edge_repr"].shape[2]
                for feat_dict in batch_list:
                    edge = feat_dict["pretrained_edge_repr"]
                    L = edge.shape[0]
                    pad = torch.zeros(max_L, max_L, C)
                    pad[:L, :L, :] = edge
                    batched_val.append(pad[None, :])
                batched_val = torch.cat(batched_val, dim=0)
            if key in ["rigids_0", "rigids_t"]:
                bsz, seqlen = batched_val.shape[:2]
                batched_val = batched_val + torch.cat(
                    [~padding_mask[..., None], torch.zeros(bsz, seqlen, 6)], dim=-1
                )

            if key in gt_feat_name:
                batch["gt_feat"][key] = batched_val
            else:
                batch[key] = batched_val
            batch["padding_mask"] = padding_mask

        return batch


class GenDataset(torch.utils.data.Dataset):
    """Dataset for conformation generation"""

    def __init__(
            self,
            csv_path: str,
            num_samples=10,
            repr_loader: Optional[DictConfig] = None,
            batch_size: Optional[int] = None,
            data_dir=None,
            msta_dir=None,
    ):
        self.csv_path = csv_path
        self.num_samples = num_samples
        self.batch_size = batch_size

        if csv_path:
            df = pd.read_csv(csv_path, index_col=None)
            df = df.loc[df.index.repeat(num_samples)].reset_index(drop=True)
            df["sample_id"] = df.groupby(df.columns.tolist()[:-1]).cumcount()
        else:
            df = pd.DataFrame()

        self.df = df
        self.repr_loader = repr_loader
        self.data_dir = data_dir
        self.msta_dir = msta_dir
        self.dynamic_batching = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        seqres = row.seqres

        data_dict = {
            "output_name": f"{row.chain_name}_sample{row.sample_id}.pdb",
            "chain_name": row.chain_name,
        }

        # pretrained representations
        if self.repr_loader is not None:
            pretrained_repr = self.repr_loader.load(seqres=seqres)
            data_dict["pretrained_node_repr"] = pretrained_repr.get(
                "pretrained_node_repr", None
            )
            data_dict["pretrained_edge_repr"] = pretrained_repr.get(
                "pretrained_edge_repr", None
            )

        aatype = torch.LongTensor(
            [rc.restype_order_with_x.get(res, 20) for res in seqres]
        )
        data_dict["aatype"] = aatype.long()
        return data_dict

    def collate(self, batch_list):
        return RCSBDataset.collate(self, batch_list)
