import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import Callback

from utils import Logger
from utils import ModelSaver
from .decomp_trainer import DecompTrainer
from .resnet_trainer import ResNetTrainer


class SaveHyparamsCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def on_init_start(self, trainer):
        self.logger.log_hyperparams()


def build_decomp_trainer(config, seed):

    ModelClass = DecompTrainer
    automatic_optimization = True

    monitoring_metrics = [
        'epoch', 'iteration', 'total_loss', 'n_rec_loss', 'e_rec_loss',
        'seg_loss', 'consistent_loss', 'h_nac_vae_loss', 'h_aac_vae_loss',
        'h_aac_norm_loss', 'd_aac_vae_loss', 'd_aac_margin_loss',
        'h_nac_vae', 'h_aac_vae', 'h_aac_norm', 'd_aac_vae', 'd_aac_margin',
        'Background', 'NET', 'ED', 'ET'
    ]

    logger = Logger(save_dir=config.save.save_dir_path,
                    config=config,
                    seed=seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics)

    save_dir_path = logger.log_dir

    checkpoint_callback = ModelSaver(
        limit_num=config.save.save_limit_num,
        save_interval=config.save.save_interval,
        monitor=None,
        dirpath=save_dir_path,
        filename='ckpt-{epoch:04d}-{total_loss:.2f}',
        save_top_k=-1,
        save_last=False,
    )

    if config.run.resume_checkpoint:
        model = ModelClass.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
            automatic_optimization=automatic_optimization,
        )

    else:
        model = ModelClass(
            config,
            save_dir_path,
            automatic_optimization=automatic_optimization,
        )

    trainer = pl.Trainer(gpus=config.run.visible_devices,
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=1,
                         accelerator='gpu',
                         strategy=DDPPlugin(find_unused_parameters=False) if config.run.distributed_backend == 'ddp' else config.run.distributed_backend,
                         deterministic=False,
                         logger=logger,
                         sync_batchnorm=config.run.sync_batchnorm,
                         callbacks=[checkpoint_callback, SaveHyparamsCallback(logger)],
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         num_sanity_val_steps=config.run.num_sanity_val_steps,
                         limit_val_batches=config.run.limit_val_batches)

    return trainer, model


def build_resnet_trainer(config, seed):

    ModelClass = ResNetTrainer
    automatic_optimization = True

    monitoring_metrics = [
        'epoch', 'iteration', 'total_loss',
        'accuracy', 'precision', 'recall', 'specificity'
    ]

    logger = Logger(save_dir=config.save.save_dir_path,
                    config=config,
                    seed=seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics)

    save_dir_path = logger.log_dir

    checkpoint_callback = ModelSaver(
        limit_num=config.save.save_limit_num,
        save_interval=config.save.save_interval,
        monitor=None,
        dirpath=save_dir_path,
        filename='ckpt-{epoch:04d}-{total_loss:.2f}',
        save_top_k=-1,
        save_last=False,
    )

    if config.run.resume_checkpoint:
        model = ModelClass.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
            monitoring_metrics=monitoring_metrics,
            automatic_optimization=True,
        )

    else:
        model = ModelClass(config,
                           save_dir_path,
                           monitoring_metrics,
                           automatic_optimization=True)

    trainer = pl.Trainer(gpus=config.run.visible_devices,
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=1,
                         accelerator=config.run.distributed_backend,
                         deterministic=True,
                         logger=logger,
                         sync_batchnorm=config.run.sync_batchnorm,
                         plugins=DDPPlugin(find_unused_parameters=True) if config.run.distributed_backend == 'ddp' else None,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         num_sanity_val_steps=config.run.num_sanity_val_steps,
                         limit_val_batches=config.run.limit_val_batches)

    return trainer, model
