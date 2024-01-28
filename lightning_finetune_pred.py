import json
import os
import random
import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import OmegaConf
from torch import nn, optim
from tqdm import tqdm

import builders
import lightning_pretrain_ssl as lps
import utils
from nets.resnet import ResNet


class SupervisedLearner(pl.LightningModule):
    def __init__(self, cfg, model, transformation=None):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.transformation = transformation
        self.criterion = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def forward(self, batch, training=True):
        x, y = batch
        x = x.float()
        y = y.long()
        if training:
            if self.transformation:
                x = self.transformation(x)

        out = self.model(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(dim=1) == y).float().mean() * 100
        return loss, acc

    def training_step(self, batchs, batch_idx):
        loss, acc = self.forward(batchs, training=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "acc": acc}

    def validation_step(self, batchs, batch_idx):
        loss, acc = self.forward(batchs, training=False)
        self.log(
            "valid_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "acc": acc}

    def test_step(self, batchs, batch_idx):
        loss, acc = self.forward(batchs, training=False)
        y_pred = self.model(batchs[0].float())
        y_pred = y_pred.argmax(dim=1)
        y_true = batchs[1].long()
        self.test_step_outputs.append(
            {"loss": loss, "acc": acc, "y_pred": y_pred, "y_true": y_true}
        )  # "representations": reprs})
        return {"loss": loss, "acc": acc}  # ), "representations": reprs}

    def configure_optimizers(self):
        if self.cfg.get("pretrained_model", False):
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.model.conv1.parameters(), "lr": self.cfg.lr},
                    {"params": self.model.encoder.parameters(), "lr": self.cfg.lr},
                    {"params": self.model.fc.parameters(), "lr": self.cfg.lr},
                ],
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return optimizer

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in self.test_step_outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in self.test_step_outputs], dim=0)
        y_true = torch.cat([x["y_true"] for x in self.test_step_outputs], dim=0)
        y_results = {"y_pred": y_pred, "y_true": y_true}
        torch.save(
            y_results,
            os.path.join(self.trainer.checkpoint_callback.dirpath, "y_results.pt"),
        )

        self.log("test_loss", avg_loss)
        self.log("test_acc", avg_acc)


class SupervisedDataModule(pl.LightningDataModule):
    def __init__(self, cfg, transformations=None):
        super().__init__()
        self.cfg = cfg
        self.transformations = transformations
        print(f"transformations: {self.transformations}")

    def setup(self, stage=None):
        y = np.load(self.cfg.y_fn)
        idx_tr, idx_val = utils.get_split_idx(y, self.cfg.fold, seed=self.cfg.seed)

        self.train_loader = utils.get_sl_loader(
            self.cfg.X_fn,
            self.cfg.y_fn,
            idxs=idx_tr,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            transformation=self.transformations,
        )
        self.val_loader = utils.get_sl_loader(
            self.cfg.X_fn,
            self.cfg.y_fn,
            idxs=idx_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            transformation=None,
        )
        self.test_loader = utils.get_sl_loader(
            self.cfg.X_te_fn,
            self.cfg.y_te_fn,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            transformation=None,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def get_model(cfg):
    hidden_sizes = [cfg.hidden_size] * cfg.n_layers
    num_blocks = [cfg.block_size] * cfg.n_layers
    input_dim = cfg.input_dim
    in_channels = cfg.in_channels
    n_classes = cfg.n_classes
    if cfg.get("use_augmentation", False):
        # transformations = utils.get_transformation(
        # perturbation_mode=cfg.perturbation_mode, p=cfg.transition_prob
        # )
        transformations = utils.get_trans_from_augtype(
            cfg.augtype, p=cfg.transition_prob
        )
    else:
        transformations = None

    # get model
    backbone = ResNet(
        hidden_sizes,
        num_blocks,
        input_dim=input_dim,
        in_channels=in_channels,
        n_classes=n_classes,
        encodeout="flatten",  # "flatten",
    )
    feature_size = backbone.fc.in_features

    ssl_model = lps.get_model(cfg)
    ssl_learner = lps.SelfSupervisedLearner(cfg, ssl_model)

    # load pretrained model
    if cfg.get("use_pretrained", False):
        print("Loading pretrained model...")
        cfg.pretrained_model = f"./results/bacteria-id/pretraining/{cfg.augtype}/{cfg.pre}/lightning_logs/version_0/checkpoints/last.ckpt"
        ssl_learner.load_state_dict(
            torch.load(cfg.pretrained_model, map_location="cuda:0")["state_dict"]
        )
    else:
        print("No pretrained model! Training from scratch...")

    backbone = ssl_learner.model.backbone

    if cfg.get("linear_eval", False):
        backbone.fc = nn.Linear(feature_size, n_classes)
    else:
        backbone.fc = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, n_classes),
        )

    del ssl_learner
    del ssl_model

    return backbone, transformations


def get_trainer(cfg):
    # logger
    if cfg.get("linear_eval", False):
        result_dir = (
            f"./results/bacteria-id/lineareval/{cfg.task}/{cfg.augtype}/{cfg.pre}/"
        )
    else:
        result_dir = (
            f"./results/bacteria-id/finetuning/{cfg.task}/{cfg.augtype}/{cfg.pre}/"
        )
    os.makedirs(result_dir, exist_ok=True)

    # logger = pl.loggers.TensorBoardLogger(result_dir, name=cfg.pre)
    logger = pl.loggers.CSVLogger(result_dir, name=f"cv{cfg.fold}", version=0)
    logger.log_hyperparams(cfg)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        filename="best",
        dirpath=os.path.join(result_dir, f"cv{cfg.fold}"),
        monitor="valid_loss",
        mode="min",
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
    )
    earlystop_callback = EarlyStopping(
        monitor="valid_loss", patience=cfg.patience, verbose=True, mode="min"
    )

    # trainer
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        devices="auto",  # cfg.devices
        precision="16-mixed" if cfg.fp16 is True else "32",
        strategy="auto",
        max_epochs=cfg.n_epochs,
        logger=logger,
        log_every_n_steps=30,
        callbacks=[checkpoint_callback, earlystop_callback],
    )
    return trainer


if __name__ == "__main__":
    # get args
    args = utils.get_args()

    # get config
    yaml_path = f"./configs/bacteria-id/finetuning/{args.task}/ssl.yaml"
    cfg = OmegaConf.load(yaml_path)

    cfg = OmegaConf.merge(cfg, args.__dict__)
    print(f"linear_eval: {cfg.get('linear_eval', False)}")
    print(f"n_epochs: {cfg.n_epochs}")

    if cfg.pre == "no_pre":
        cfg.use_pretrained = False

    # set seed
    utils.seed_all(cfg.seed)
    pl.seed_everything(cfg.seed)

    # get model
    backbone, transformations = get_model(cfg)

    # initialize model
    for module in backbone.fc.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # freeze backbone if linear eval
    if cfg.get("linear_eval", False):
        for param in backbone.conv1.parameters():
            param.requires_grad = False
        for param in backbone.encoder.parameters():
            param.requires_grad = False

    # get dataloader
    dm = SupervisedDataModule(cfg, transformations)

    # get trainer
    trainer = get_trainer(cfg)

    # wrap model
    sl_learner = SupervisedLearner(cfg, backbone)

    # train
    trainer.fit(sl_learner, dm)

    # load best model
    model_ckpt = torch.load(
        os.path.join(
            trainer.checkpoint_callback.dirpath,
            trainer.checkpoint_callback.best_model_path,
        ),
        map_location="cuda:0",
    )
    sl_learner.load_state_dict(model_ckpt["state_dict"])

    # test
    test_results = trainer.test(sl_learner, dm)
    with open(
        os.path.join(trainer.checkpoint_callback.dirpath, "test_results.json"), "w"
    ) as f:
        json.dump(test_results, f)
