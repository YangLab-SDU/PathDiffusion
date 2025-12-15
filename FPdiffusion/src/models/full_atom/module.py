"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""

from typing import Dict, Any

import torch
from lightning import LightningModule, Fabric, seed_everything
from torchmetrics import MeanMetric, MinMetric
from openfold.np.protein import Protein
from openfold.np import protein
from src.utils import hydra_utils
from src.analysis.eval import eval_gen_conf
import os
import numpy as np
import torch.distributed as dist

logger = hydra_utils.get_pylogger(__name__)


class FullAtomLitModule(LightningModule):
    # ... (__init__, setup, forward, sampling, training_step 等保持不变) ...

    def __init__(
        self,
        score_network,
        diffuser,
        optimizer,
        scheduler,
        lr_warmup_steps: int,
        val_gen_every_n_epochs: int,
        output_dir: str,
        stage: int,
        log_loss_name=[
            "total",
            "rot",
            "trans",
            "bb_coords",
            "bb_dist_map",
            "torsion",
            "fape",
        ],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["score_network"])
        self.output_dir = output_dir
        self.val_output_dir = output_dir
        self.test_output_dir = output_dir
        self.score_network = score_network
        self.score_network.diffuser = diffuser
        self.loss_name = log_loss_name
        self.stage = stage

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                setattr(self, f"{split}_{loss_name}", MeanMetric())
        self.best_val_total = MinMetric()

    def setup(self, stage: str) -> None:
        # broadcast output_dir from rank 0
        if dist.is_available() and dist.is_initialized():
            fabric = Fabric()
            fabric.launch()
            self.output_dir = fabric.broadcast(self.output_dir, src=0)
            self.val_output_dir = fabric.broadcast(self.val_output_dir, src=0)
            self.test_output_dir = fabric.broadcast(self.test_output_dir, src=0)

    def forward(self, batch: Dict[str, Any]):
        return self.score_network(**batch)

    def sampling(self, batch, output_dir):
        if self.stage == 1:
            self.score_network.reverse_sample_stage1(**batch, output_dir=output_dir)
        elif self.stage == 2:
            self.score_network.reverse_sample_stage2(**batch, output_dir=output_dir)
        else:
            raise ValueError(f"Unsupported stage number: {self.stage}，only 1 or 2 are supported")

    def on_train_start(self):
        if dist.is_available() and dist.is_initialized():
            local_rank = int(dist.get_rank())
        else:
            local_rank = 0

        seed_everything(42 + local_rank, workers=True)

        self.val_total_loss = torch.tensor(float('inf'), device=self.device)
        self.log("val_total_loss", self.val_total_loss, sync_dist=True)

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                getattr(self, f"{split}_{loss_name}").reset()
        self.best_val_total.reset()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, aux_info = self.forward(batch)
        for loss_name in self.loss_name:
            getattr(self, f"train_{loss_name}").update(aux_info[loss_name])
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        if batch is None or len(batch) == 0:
            logger.error("Received empty validation batch!")
            return None

        try:
            loss, aux_info = self.forward(batch)

            self.log("val_total_loss", aux_info["total"], sync_dist=True, prog_bar=True)
            self.log("val/total_loss", aux_info["total"], sync_dist=True)

            for loss_name in self.loss_name:
                getattr(self, f"val_{loss_name}").update(aux_info[loss_name])
            return loss
        except Exception as e:
            logger.error(f"Validation step failed: {str(e)}")
            return None

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, sync_dist=True)

        log_dict = {}

        for loss_name in self.loss_name:
            train_metric_name = f"train_{loss_name}"
            if hasattr(self, train_metric_name):
                metric = getattr(self, train_metric_name)
                try:
                    val = metric.compute().item()
                    log_dict[f"train/{loss_name}"] = val
                    metric.reset()
                except Exception:
                    pass

            val_metric_name = f"val_{loss_name}"
            if hasattr(self, val_metric_name):
                metric = getattr(self, val_metric_name)
                try:
                    val = metric.compute().item()
                    log_dict[f"val/{loss_name}"] = val
                    metric.reset()
                except Exception:
                    pass

        log_msg = f"Current epoch: {self.current_epoch}, step: {self.global_step}, lr: {lr:.8f}, "
        for name, val in log_dict.items():
            log_msg += f"{name}: {val:.8f}, "

        logger.info(log_msg)
        # =================================================================

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # evaluate val_gen results
        output_dir = f"{self.val_output_dir}/epoch{self.current_epoch}"
        if self.trainer.is_global_zero and os.path.exists(output_dir):
            try:
                if hasattr(self.trainer.datamodule, 'val_gen_dataset') and self.trainer.datamodule.val_gen_dataset:
                    log_stats, log_dist = eval_gen_conf(
                        output_root=output_dir,
                        csv_fpath=self.trainer.datamodule.val_gen_dataset.csv_path,
                        ref_root=self.trainer.datamodule.val_gen_dataset.data_dir,
                        num_samples=self.trainer.datamodule.val_gen_dataset.num_samples,
                        n_proc=1,
                    )
                    log_stats = {
                        f"val_gen/cameo/{name}": val for name, val in log_stats.items()
                    }
                    self.log_dict(log_stats, rank_zero_only=True, sync_dist=True)
            except Exception as e:
                logger.warn(f"Failed to evaluate val_gen results: {e}")

        torch.cuda.empty_cache()

    def on_test_start(self):
        if dist.is_available() and dist.is_initialized():
            local_rank = int(dist.get_rank())
        else:
            local_rank = 0
        seed_everything(42 + local_rank, workers=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        self.sampling(batch, os.path.join(self.output_dir, self.test_output_dir))

    def on_test_end(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @property
    def is_epoch_based(self):
        return type(self.trainer.val_check_interval) == float and self.trainer.val_check_interval <= 1.0

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, 'func') and self.hparams.optimizer.func.__name__ == "DeepSpeedCPUAdam":
             optimizer = self.hparams.optimizer(model_params=self.parameters())
        else:
             optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is None:
            return optimizer

        scheduler = self.hparams.scheduler(optimizer=optimizer)

        logger.info(f"Initialized scheduler: {scheduler.__class__.__name__}")

        config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            logger.info("Configuring ReduceLROnPlateau with relaxed settings")
            config.update({
                "monitor": "val_total_loss",
                "strict": False,
            })

        return {
            "optimizer": optimizer,
            "lr_scheduler": config
        }

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.lr_warmup_steps
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.optimizer.keywords["lr"]
        super().optimizer_step(*args, **kwargs)