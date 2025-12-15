"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""
import torch
from torch import nn
from src.models.full_atom.utils.all_atom import atom14_to_atom37
from openfold.utils import rigid_utils as ru
from tqdm import tqdm
from src.models.full_atom.score_network import BaseScoreNetwork
from torch import einsum

import numpy as np
import os
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial.transform import Rotation

from openfold.data import data_transforms
from openfold.np import residue_constants as rc
from openfold.np.protein import Protein
from openfold.np import protein
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from src.utils import hydra_utils

logger = hydra_utils.get_pylogger(__name__)
pdb_parser = PDBParser(QUIET=True)

class GuidanceScoreNetwork(BaseScoreNetwork):
    def __init__(
        self,
        cond_model_nn,
        cond_ckpt_path,
        cfg,
        msta_dir,
        uncond_model_nn=None,
        uncond_ckpt_path=None,
        **kwargs
    ):
        super(GuidanceScoreNetwork, self).__init__(cond_model_nn, cfg)
        self.diffuser = None
        if cond_ckpt_path:
            cond_ckpt = torch.load(cond_ckpt_path, map_location="cpu")["state_dict"]
            cond_state_dict = {}
            for key in cond_ckpt.keys():
                if key.startswith("score_network.model_nn."):
                    cond_state_dict[key[len("score_network.model_nn.") :]] = cond_ckpt[key]

            self.model_nn.load_state_dict(cond_state_dict, strict=False)
            del cond_ckpt
        for param in self.model_nn.parameters():
            param.requires_grad = False

        self.uncond_model_nn = uncond_model_nn
        self.msta_dir = msta_dir

        if uncond_ckpt_path:
            uncond_ckpt = torch.load(uncond_ckpt_path, map_location="cpu")["state_dict"]
            uncond_state_dict = {}
            for key in uncond_ckpt.keys():
                if key.startswith("score_network.model_nn"):
                    uncond_state_dict[key[len("score_network.model_nn.") :]] = uncond_ckpt[key]

            self.uncond_model_nn.load_state_dict(uncond_state_dict, strict=False)
            del uncond_ckpt

    @property
    def device(self):
        return self.model_nn.device

    def forward(
        self,
        aatype,
        t,
        rigids_t,
        rigids_mask,
        padding_mask,
        gt_feat,
        res_idx=None,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ):
        self.diffuser._so3_diffuser.use_cached_score = False
        rigids_t = ru.Rigid.from_tensor_7(rigids_t)
        rigids_mask = rigids_mask * padding_mask

        model_out = self.model_nn(
            aatype=aatype,
            padding_mask=padding_mask,
            t=t,
            rigids_t=rigids_t,
            rigids_mask=rigids_mask,
            res_idx=res_idx,
            pretrained_node_repr=pretrained_node_repr,
            pretrained_edge_repr=pretrained_edge_repr,
        )

        input_feat = {
            "aatype": aatype,
            "t": t,
            "rigids_t": rigids_t,
            "rigids_mask": rigids_mask,
            "padding_mask": padding_mask,
            "gt_feat": gt_feat,
            "res_idx": res_idx,
            "pretrained_node_repr": pretrained_node_repr,
            "pretrained_edge_repr": pretrained_edge_repr,
        }

        loss = self.loss_fn(input_feat, model_out)
        return loss, {"total": loss.item()}

    def loss_fn(self, input_feat, model_out, **kwargs):
        return torch.tensor(0)

    def forward_guidance_model(self, input_feat):
        model_out = self.model_nn(**input_feat)
        return model_out


    def reverse_sample_stage1(
            self,
            aatype,
            padding_mask,
            chain_name,
            output_dir,
            pretrained_node_repr=None,
            pretrained_edge_repr=None,
            **kwargs,
    ):
        assert not self.model_nn.training
        self.diffuser._so3_diffuser.use_cached_score = True

        """ Reverse sampling. """
        rigids_mask = padding_mask.float()

        batch_size, seq_len = aatype.shape[:2]

        msta_fpath = os.path.join(self.msta_dir, f"{chain_name[0]}.msta")
        folding_priority = self.load_msta(msta_fpath, seq_len).to(aatype.device)  # [L]
        folding_priority = folding_priority.unsqueeze(0)  # [1, L]

        base_dt = 1.0 / (self.cfg.stage1_diffusion_steps + self.cfg.base_steps)
        dt_scale_coeff = ((self.cfg.stage1_diffusion_steps + self.cfg.base_steps) / self.cfg.base_steps) - 1
        dt_scale_first = 1 + dt_scale_coeff * folding_priority
        dt = base_dt * dt_scale_first  # [1, L]

        t = torch.ones(batch_size, seq_len, device=aatype.device)  # [B, L]

        rigids_t = self.diffuser.sample_ref(
            n_samples=batch_size, seq_len=seq_len, device=aatype.device
        )

        for step_t in tqdm(range(self.cfg.stage1_diffusion_steps + self.cfg.base_steps)):
            input_feat = {
                "aatype": aatype,
                "t": t,
                "rigids_t": rigids_t,
                "rigids_mask": rigids_mask,
                "padding_mask": padding_mask,
                "pretrained_node_repr": pretrained_node_repr,
                "pretrained_edge_repr": pretrained_edge_repr,
            }

            model_out = self.forward_guidance_model(
                input_feat=input_feat,
            )

            active_mask = (t > 0.01).float()
            current_rigids_mask = rigids_mask * active_mask

            pred_rot_score = (
                    self.diffuser.calc_rot_score(
                        rigids_t.get_rots(),
                        model_out["pred_rigids_0"].get_rots(),
                        t,
                    )
                    * current_rigids_mask[..., None]
            )

            pred_trans_score = (
                    self.diffuser.calc_trans_score(
                        rigids_t.get_trans(),
                        model_out["pred_rigids_0"].get_trans(),
                        t,
                        use_torch=True,
                    )
                    * current_rigids_mask[..., None]
            )

            rigids_s = self.diffuser.reverse(
                rigids_t=rigids_t,
                rot_score=pred_rot_score,
                trans_score=pred_trans_score,
                t=t,
                dt=dt,
                diffuse_mask=active_mask
            )

            rigids_t = rigids_s

            update_mask = (t > 0.01).float()
            t = t - dt * update_mask
            t = torch.clamp(t, min=0.01)

            output_dir_allatom = os.path.join(output_dir, "stage1")
            os.makedirs(output_dir_allatom, exist_ok=True)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids_t,
                model_out["pred_torsions"],
                aatype,
            )
            model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                torch.fmod(aatype, 20),
            )
            pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)
            if step_t - self.cfg.base_steps >= 0:
                for i in range(batch_size):
                    padding_mask_i = padding_mask[i]
                    aatype_i = aatype[i][padding_mask_i].cpu().numpy()
                    atom37_i = pred_atom37[i][padding_mask_i].detach().cpu().numpy()
                    atom37_mask_i = atom37_mask[i][padding_mask_i].detach().cpu().numpy()
                    res_idx = np.arange(aatype_i.shape[0])
                    gen_protein = Protein(
                        aatype=aatype_i,
                        atom_positions=atom37_i,
                        atom_mask=atom37_mask_i,
                        residue_index=res_idx + 1,
                        chain_index=np.zeros_like(aatype_i),
                        b_factors=np.zeros_like(atom37_mask_i),
                    )
                    pdb_path = os.path.join(output_dir_allatom, f"stage1_step_{step_t - self.cfg.base_steps}_sample{i}.pdb")
                    with open(pdb_path, "w") as fp:
                        fp.write(protein.to_pdb(gen_protein))


    def reverse_sample_stage2(
            self,
            aatype,
            padding_mask,
            chain_name,
            output_dir,
            pretrained_node_repr=None,
            pretrained_edge_repr=None, **kwargs,
    ):
        assert not self.model_nn.training
        self.diffuser._so3_diffuser.use_cached_score = True

        """ Reverse sampling (modified version with three-region guidance). """
        rigids_mask = padding_mask.float()

        batch_size, seq_len = aatype.shape[:2]
        base_dt = 1.0 / self.cfg.stage2_diffusion_steps

        for step_num in tqdm(
                range(self.cfg.starting_steps + self.cfg.base_steps, self.cfg.final_steps + self.cfg.base_steps)):
            logger.info(f"step {step_num}")

            stage1_pdb_dir = os.path.join(output_dir, "stage1")
            current_step_in_stage1 = step_num - self.cfg.base_steps

            current_pdb_path = os.path.join(stage1_pdb_dir, f"stage1_step_{current_step_in_stage1}_sample0.pdb")
            final_stage1_pdb_path = os.path.join(stage1_pdb_dir, f"stage1_step_{self.cfg.stage1_diffusion_steps - 1}_sample0.pdb")

            for pdb_path in [current_pdb_path, final_stage1_pdb_path]:
                if not os.path.exists(pdb_path):
                    raise FileNotFoundError(f"The stage1 pdb file does not exist: {pdb_path}")

            ca_coords_current = self._get_ca_coords(current_pdb_path, seq_len, aatype)
            ca_coords_final = self._get_ca_coords(final_stage1_pdb_path, seq_len, aatype)

            distances = torch.norm(ca_coords_current - ca_coords_final, dim=1)

            dynamic_weight = torch.full(
                size=(seq_len,),
                fill_value=1 - self.cfg.clsfree_guidance_strength,
                device=aatype.device
            )

            cond1_mask = distances < 1.0
            cond1_segments = self._find_continuous_segments(cond1_mask, seq_len)
            for start, end in cond1_segments:
                dynamic_weight[start:end+1] = self.cfg.clsfree_guidance_strength

            cond2_mask = (distances > 1.0) & (distances <= 5.0)
            cond2_segments = self._find_continuous_segments(cond2_mask, seq_len)
            for start, end in cond2_segments:
                for i in range(start, end+1):
                    if dynamic_weight[i] == (1 - self.cfg.clsfree_guidance_strength):
                        dynamic_weight[i] = 0.5

            dynamic_weight = dynamic_weight.unsqueeze(0)  # [1, L]
            dynamic_weight = torch.clamp(dynamic_weight, min=0.0, max=1.0)

            t = torch.ones(batch_size, seq_len, device=aatype.device)  # [B, L]
            dt = base_dt * torch.ones_like(t)  # [B, L]

            if not os.path.exists(current_pdb_path):
                raise FileNotFoundError(f"The stage1 pdb file does not exist: {current_pdb_path}")
            logger.info(f"Initialize rigids_t from the stage1 pdb: {current_pdb_path}")

            rigids_t = self.initialize_rigids_from_pdb(
                pdb_path=current_pdb_path,
                seq_len=seq_len,
                device=aatype.device,
                batch_size=batch_size,
                aatype=aatype
            )

            for step_t in tqdm(range(self.cfg.stage2_diffusion_steps)):
                active_mask = (t > 0.01).float()
                current_rigids_mask = rigids_mask * active_mask

                if self.uncond_model_nn and self.cfg.clsfree_guidance_strength <= 1:
                    uncond_model_out = self.uncond_model_nn(
                        aatype=aatype,
                        padding_mask=padding_mask,
                        t=t,
                        rigids_t=rigids_t,
                        rigids_mask=rigids_mask,
                        res_idx=None,
                        pretrained_node_repr=None,
                        pretrained_edge_repr=None,
                    )

                    uncond_pred_rot_score = (
                            self.diffuser.calc_rot_score(
                                rigids_t.get_rots(),
                                uncond_model_out["pred_rigids_0"].get_rots(),
                                t,
                            )
                            * current_rigids_mask[..., None]
                    )

                    uncond_pred_trans_score = (
                            self.diffuser.calc_trans_score(
                                rigids_t.get_trans(),
                                uncond_model_out["pred_rigids_0"].get_trans(),
                                t,
                                use_torch=True,
                            )
                            * current_rigids_mask[..., None]
                    )
                else:
                    uncond_pred_rot_score, uncond_pred_trans_score = 0, 0

                input_feat = {
                    "aatype": aatype,
                    "t": t,
                    "rigids_t": rigids_t,
                    "rigids_mask": rigids_mask,
                    "padding_mask": padding_mask,
                    "pretrained_node_repr": pretrained_node_repr,
                    "pretrained_edge_repr": pretrained_edge_repr,
                }

                model_out = self.forward_guidance_model(
                    input_feat=input_feat,
                )

                pred_rot_score = (
                        self.diffuser.calc_rot_score(
                            rigids_t.get_rots(),
                            model_out["pred_rigids_0"].get_rots(),
                            t,
                        )
                        * current_rigids_mask[..., None]
                )

                pred_trans_score = (
                        self.diffuser.calc_trans_score(
                            rigids_t.get_trans(),
                            model_out["pred_rigids_0"].get_trans(),
                            t,
                            use_torch=True,
                        )
                        * current_rigids_mask[..., None]
                )

                target_shape = pred_rot_score.shape

                dynamic_weight_expanded = dynamic_weight.view(*dynamic_weight.shape, 1).expand(target_shape)

                pred_rot_score = (dynamic_weight_expanded * self.cfg.clsfree_guidance_strength) * pred_rot_score + \
                                 (1 - dynamic_weight_expanded * self.cfg.clsfree_guidance_strength) * uncond_pred_rot_score

                pred_trans_score = (dynamic_weight_expanded * self.cfg.clsfree_guidance_strength) * pred_trans_score + \
                                   (1 - dynamic_weight_expanded * self.cfg.clsfree_guidance_strength) * uncond_pred_trans_score

                rigids_s = self.diffuser.reverse(
                    rigids_t=rigids_t,
                    rot_score=pred_rot_score,
                    trans_score=pred_trans_score,
                    t=t,
                    dt=dt,
                    diffuse_mask=active_mask
                )
                rigids_t = rigids_s

                update_mask = (t > 0.01).float()
                t = t - dt * update_mask
                t = torch.clamp(t, min=0.01)

            output_dir_allatom = os.path.join(output_dir, "stage2")
            os.makedirs(output_dir_allatom, exist_ok=True)

            if "pred_atom14" not in model_out:
                all_frames_to_global = self.torsion_angles_to_frames(
                    rigids_t,
                    model_out["pred_torsions"],
                    aatype,
                )
                model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    torch.fmod(aatype, 20),
                )

            pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)

            sample_info = []

            for i in range(batch_size):
                padding_mask_i = padding_mask[i]
                aatype_i = aatype[i][padding_mask_i].cpu().numpy()
                atom37_i = pred_atom37[i][padding_mask_i].detach().cpu().numpy()
                atom37_mask_i = atom37_mask[i][padding_mask_i].detach().cpu().numpy()
                res_idx = np.arange(aatype_i.shape[0])
                gen_protein = Protein(
                    aatype=aatype_i,
                    atom_positions=atom37_i,
                    atom_mask=atom37_mask_i,
                    residue_index=res_idx + 1,
                    chain_index=np.zeros_like(aatype_i),
                    b_factors=np.zeros_like(atom37_mask_i),
                )
                pdb_path = os.path.join(output_dir_allatom,
                                        f"Step_{step_num - self.cfg.base_steps}_sample{i}.pdb")
                with open(pdb_path, "w") as fp:
                    fp.write(protein.to_pdb(gen_protein))

                clash_count = self.calculate_backbone_clashes(pdb_path)
                sample_info.append((pdb_path, clash_count))

            if sample_info:
                sample_info.sort(key=lambda x: x[1])
                best_pdb_path, best_clash_count = sample_info[0]
                best_output_path = os.path.join(output_dir_allatom, f"Step_{step_num - self.cfg.base_steps}_sample.pdb")

                with open(best_pdb_path, 'r') as f_in, open(best_output_path, 'w') as f_out:
                    f_out.write(f_in.read())

    def _get_ca_coords(self, pdb_path, seq_len, aatype):
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("prot", pdb_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDB {pdb_path}: {str(e)}")

        ca_coords = []
        residue_ids = []
        for chain in structure.get_chains():
            for res in chain.get_residues():
                if res.get_id()[0] == ' ':
                    try:
                        ca_atom = res['CA']
                        ca_coords.append(ca_atom.get_coord())
                        residue_ids.append(res.get_id()[1])
                    except KeyError:
                        raise ValueError(f"Residue {res.get_id()} missing CA atom（PDB: {pdb_path}）")

        if len(ca_coords) != seq_len:
            raise ValueError(f"PDB {pdb_path} contains {len(ca_coords)} valid residues, expected to contain {seq_len}")

        return torch.tensor(ca_coords, dtype=torch.float32, device=aatype.device)

    def _find_continuous_segments(self, mask, seq_len, min_length=3):
        segments = []
        start_idx = None
        for i in range(seq_len):
            if mask[i] and start_idx is None:
                start_idx = i
            elif not mask[i] and start_idx is not None:
                if (i - 1) - start_idx + 1 >= min_length:
                    segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None and (seq_len - 1) - start_idx + 1 >= min_length:
            segments.append((start_idx, seq_len - 1))
        return segments

    def calculate_backbone_clashes(self, pdb_path):
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", pdb_path)
        except:
            return float('inf')

        backbone_atoms = []
        for atom in structure.get_atoms():
            if atom.name in ['N', 'CA', 'C', 'O'] and atom.element != 'H':
                backbone_atoms.append(atom)

        if len(backbone_atoms) == 0:
            return 0

        ns = NeighborSearch(backbone_atoms)
        clash_count = 0
        vdw_radii = {'N': 1.55, 'CA': 1.7, 'C': 1.7, 'O': 1.52}

        for i, atom1 in enumerate(backbone_atoms):
            neighbors = ns.search(atom1.coord, 5.0, level='A')
            for atom2 in neighbors:
                if id(atom1) >= id(atom2):
                    continue
                res1 = atom1.get_parent()
                res2 = atom2.get_parent()
                chain1 = res1.get_parent()
                chain2 = res2.get_parent()

                if (res1.get_id()[1] == res2.get_id()[1] and chain1.id == chain2.id):
                    continue
                if abs(res1.get_id()[1] - res2.get_id()[1]) <= 1 and chain1.id == chain2.id:
                    continue

                r1 = vdw_radii.get(atom1.name, 1.7)
                r2 = vdw_radii.get(atom2.name, 1.7)
                distance = np.linalg.norm(atom1.coord - atom2.coord)

                if distance < 0.75 * (r1 + r2):
                    clash_count += 1
        return clash_count

    def initialize_rigids_from_pdb(self, pdb_path, seq_len, device, batch_size, aatype):
        atom_coords = self.load_pdb(pdb_path, seq_len)
        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)

        all_atom_positions_single = torch.from_numpy(atom_coords).to(device)
        all_atom_mask_single = torch.all(~torch.isnan(all_atom_positions_single), dim=-1)
        all_atom_positions_single = torch.nan_to_num(all_atom_positions_single, nan=0.0)

        aatype_seq_single = torch.LongTensor([
            rc.restype_order_with_x.get(res.item(), 20) for res in aatype[0]
        ]).to(device)

        openfold_feat_dict = {
            "aatype": aatype_seq_single.long(),
            "all_atom_positions": all_atom_positions_single.double(),
            "all_atom_mask": all_atom_mask_single.double(),
        }

        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles()(openfold_feat_dict)

        rigids_single = ru.Rigid.from_tensor_4x4(openfold_feat_dict["rigidgroups_gt_frames"])[:, 0]
        rots_tensor_single = rigids_single.get_rots().get_rot_mats()
        trans_single = rigids_single.get_trans()

        rots_tensor_batch = rots_tensor_single.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        trans_batch = trans_single.unsqueeze(0).expand(batch_size, -1, -1).clone()

        rots_batch = ru.Rotation(rot_mats=rots_tensor_batch)
        rigids_batch = ru.Rigid(rots=rots_batch, trans=trans_batch)

        return rigids_batch

    def load_msta(self, fold_priority_file: str, seq_len: int) -> torch.Tensor:
        folding_priority = []
        with open(fold_priority_file, 'r') as f:
            for line in f:
                _, priority = line.strip().split()
                folding_priority.append(float(priority))
        return torch.tensor(folding_priority)[:seq_len]

    def load_pdb(self, pdb_path, seqlen=None):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("", pdb_path)
        atom_coords = np.full((seqlen, rc.atom_type_num, 3), np.nan, dtype=np.float32)
        model = struct[0]
        chain = next(model.get_chains())

        for residue in chain:
            res_id = residue.id[1]
            seq_idx = res_id - 1
            if seq_idx >= seqlen:
                continue
            for atom in residue:
                if atom.name in rc.atom_order:
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
        return atom_coords

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer("default_frames",
                                 torch.tensor(rc.restype_rigid_group_default_frame, dtype=float_dtype, device=device,
                                              requires_grad=False), persistent=False)
        if not hasattr(self, "group_idx"):
            self.register_buffer("group_idx",
                                 torch.tensor(rc.restype_atom14_to_rigid_group, device=device, requires_grad=False),
                                 persistent=False)
        if not hasattr(self, "atom_mask"):
            self.register_buffer("atom_mask", torch.tensor(rc.restype_atom14_mask, dtype=float_dtype, device=device,
                                                           requires_grad=False), persistent=False)
        if not hasattr(self, "lit_positions"):
            self.register_buffer("lit_positions",
                                 torch.tensor(rc.restype_atom14_rigid_group_positions, dtype=float_dtype, device=device,
                                              requires_grad=False), persistent=False)

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(r, f, self.default_frames, self.group_idx, self.atom_mask,
                                                             self.lit_positions)
