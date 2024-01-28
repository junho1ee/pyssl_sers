import os
import random
import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from tqdm import tqdm

import builders
import utils
from nets.resnet import ResNet

torch.set_float32_matmul_precision("medium")


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def forward(self, x1, x2):
        return self.model(x1.float(), x2.float())

    def training_step(self, batchs, batch_idx):
        loss = self.forward(batchs[0], batchs[1])
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


def get_model(cfg):
    # initialize params
    hidden_sizes = [cfg.hidden_size] * cfg.n_layers
    num_blocks = [cfg.block_size] * cfg.n_layers
    input_dim = cfg.input_dim
    in_channels = cfg.in_channels
    n_classes = cfg.n_classes
    # transformations = utils.get_transformation(
    # perturbation_mode=cfg.perturbation_mode, p=1.0
    # )
    transformations = utils.get_trans_from_augtype(cfg.augtype, p=1.0)

    backbone = ResNet(
        hidden_sizes,
        num_blocks,
        input_dim=input_dim,
        in_channels=in_channels,
        n_classes=n_classes,
        encodeout="flatten",  # flatten
    )
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()  # type: ignore

    # initialize ssl method
    ssl_model = getattr(builders, cfg.pre)(
        backbone, feature_size, transformations=transformations
    )
    return ssl_model


def get_trainer(cfg):
    # logger
    result_dir = f"./results/bacteria-id/pretraining/{cfg.augtype}/{cfg.pre}/"
    os.makedirs(result_dir, exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(result_dir)  # type: ignore
    logger.log_hyperparams(cfg)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{loss:.4f}",
        monitor="loss",
        mode="min",
        save_last=True,
        save_weights_only=True,
        save_top_k=-1,
        every_n_epochs=500,
    )

    # trainer
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        devices="auto",
        precision="16-mixed" if cfg.fp16 is True else "32",
        strategy="auto",
        min_epochs=cfg.n_epochs,  # type: ignore
        max_epochs=cfg.n_epochs,
        logger=logger,
        log_every_n_steps=30,
        callbacks=[
            checkpoint_callback,
        ],
    )
    return trainer


if __name__ == "__main__":
    # get args
    args = utils.get_args()

    # get config
    cfg = OmegaConf.load(
        f"./configs/bacteria-id/pretraining/{args.augtype}/{args.pre}.yaml"
    )
    cfg = OmegaConf.merge(cfg, args.__dict__)

    # set seed
    utils.seed_all(cfg.seed)
    pl.seed_everything(cfg.seed)

    # get dataloader
    loader = utils.get_ssl_loader(
        cfg.X_fn, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # get trainer
    trainer = get_trainer(cfg)

    ssl_model = get_model(cfg)
    ssl_learner = SelfSupervisedLearner(cfg, ssl_model)

    # train
    trainer.fit(ssl_learner, loader)
