"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import shutil
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.loggers import Logger
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary

from src.utils import hydra_utils
from train import load_state_dict_from_checkpoint
from src.utils.misc.misc import replace_dot_key_val, get_git_commit

log = hydra_utils.get_pylogger(__name__)


@hydra_utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger = []

    log.info("Instantiating callbacks (Hardcoded)...")
    callbacks = [
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=1)
    ]

    log.info(f"Configuring Trainer (Hardcoded defaults + Experiment overrides)...")

    trainer_args = {
        "accelerator": "gpu",
        "devices": 1,
        "precision": 32,
        "inference_mode": True,
        "enable_progress_bar": True,
        "default_root_dir": cfg.paths.output_dir
    }

    if "trainer" in cfg and cfg.trainer is not None:
        exp_trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
        if exp_trainer_args.get("enable_progress_bar") is False:
             exp_trainer_args["enable_progress_bar"] = True

        exp_trainer_args = {k: v for k, v in exp_trainer_args.items() if v is not None}
        trainer_args.update(exp_trainer_args)

    log.info(f"Instantiating Trainer with args: {trainer_args}")
    trainer = Trainer(**trainer_args, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.ckpt_path is not None:
        ckpt_path = list(Path(cfg.paths.log_dir).joinpath("checkpoints").glob("best_model*.ckpt"))[0] \
                    if cfg.ckpt_path == "infer" else cfg.ckpt_path

        if not os.path.exists(ckpt_path):
            log.warning(f"Checkpoint path not found: {ckpt_path}.")

        if os.path.exists(ckpt_path):
            log.info(f"Loading model state_dict from {ckpt_path}")
            state_dict = load_state_dict_from_checkpoint(ckpt_path)
            model.load_state_dict(state_dict=state_dict, strict=False)
        else:
             log.warning(f"Failed to load checkpoint from {ckpt_path}")
    else:
        log.warning("No checkpoint provided for evaluation")

    trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def clean_cfg(cfg):
    """Process cfg for evaluation"""
    assert cfg.task_name is not None, 'Please set task_name'

    with open_dict(cfg):
        try:
            cfg.commit = commit = get_git_commit()
            log.info(f"Commit: {commit}")
        except:
            pass
        train_keys = ['data.train_dataset', 'data.val_dataset', 'data.val_gen_dataset']
        for key in train_keys:
            replace_dot_key_val(cfg, dot_key=key, replace_to=None, inplace=True, ignore_error=True)
    return cfg


@hydra.main(version_base=None, config_path="./settings", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    cfg = clean_cfg(cfg)
    hydra_utils.print_config_tree(cfg, resolve=True, save_to_file=False)
    save_cfg = rank_zero_only(OmegaConf.save)
    runtime_output_dir = str(HydraConfig.get().runtime.output_dir)
    save_cfg(OmegaConf.to_container(cfg, resolve=True), f"{runtime_output_dir}/eval.yaml")
    evaluate(cfg)


if __name__ == "__main__":
    main()