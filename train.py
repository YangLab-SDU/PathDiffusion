"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""

import os
from typing import List, Optional, Tuple, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import Logger, CSVLogger
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    RichModelSummary
)

from src.utils import hydra_utils
from src.utils.misc.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

log = hydra_utils.get_pylogger(__name__)

def load_state_dict_from_checkpoint(ckpt_path):
    if os.path.isdir(ckpt_path):
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {key.replace("_forward_module.", ""): val for key, val in state_dict.items()}
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
    return state_dict

@hydra_utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict, Dict]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    if "SLURM_JOB_GPUS" in os.environ:
        log.info(f"Slurm分配的GPU: {os.environ['SLURM_JOB_GPUS']}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Logger
    log.info("Instantiating loggers (Hardcoded CSVLogger)...")
    save_dir = cfg.paths.output_dir if "paths" in cfg and "output_dir" in cfg.paths else os.getcwd()
    logger: List[Logger] = [CSVLogger(save_dir=save_dir, name="csv_logs", version="")]

    # Callbacks
    log.info("Instantiating callbacks (Hardcoded)...")
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    callbacks: List[Callback] = []
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_model_epoch{epoch:d}_step{step:d}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    ))
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="model_epoch{epoch:d}_step{step:d}",
        monitor=None,
        every_n_epochs=100,
        save_top_k=-1,
        auto_insert_metric_name=False
    ))
    callbacks.append(EarlyStopping(
        monitor="val/total_loss",
        patience=100,
        mode="min",
        verbose=True,
        strict=True,
        check_finite=True
    ))
    callbacks.append(RichProgressBar(leave=True))
    callbacks.append(RichModelSummary(max_depth=1))

    # Trainer Config
    log.info("Configuring Trainer (Hardcoded defaults + Experiment overrides)...")
    trainer_args = {
        "accelerator": "gpu",
        "devices": "auto",
        "strategy": "ddp",
        "precision": 32,
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "check_val_every_n_epoch": 1,
        "num_nodes": 1,
        "enable_progress_bar": True,
        "inference_mode": False,
        "detect_anomaly": False,
        "default_root_dir": save_dir,
    }

    if "trainer" in cfg and cfg.trainer is not None:
        exp_trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
        exp_trainer_args = {k: v for k, v in exp_trainer_args.items() if v is not None}
        if exp_trainer_args:
            log.info(f"Overriding trainer args from experiment config: {exp_trainer_args}")
            trainer_args.update(exp_trainer_args)

    if torch.cuda.is_available():
        trainer_args["accelerator"] = "gpu"

    log.info(f"Instantiating Trainer with args: {trainer_args}")
    trainer = Trainer(
        **trainer_args,
        callbacks=callbacks,
        logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Distributed Env Setup
    if trainer.strategy == "ddp":
        import time
        time.sleep(1)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12345'

    if logger:
        log.info("Logging hyperparameters!")
        hydra_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("ckpt_path") is not None:
        if cfg.get("load_state_dict_only", False):
            log.info(f"Loading model state_dict from {cfg.ckpt_path}")
            state_dict = load_state_dict_from_checkpoint(cfg.ckpt_path)
            model.load_state_dict(state_dict=state_dict, strict=True)
            log.info("Start training!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
        else:
            log.info(f"Resume training from checkpoint {cfg.ckpt_path}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        log.info("Start training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict

@hydra.main(version_base=None, config_path="./settings", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    metric_dict, _ = train(cfg)
    metric_value = hydra_utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == "__main__":
    main()