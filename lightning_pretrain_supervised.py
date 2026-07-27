import os

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn

import utils
from nets.resnet import ResNet


torch.set_float32_matmul_precision("medium")


class ReferenceSupervisedLearner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x.float())
        loss = self.criterion(out, y.long())
        acc = (out.argmax(dim=1) == y.long()).float().mean() * 100
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x.float())
        loss = self.criterion(out, y.long())
        acc = (out.argmax(dim=1) == y.long()).float().mean() * 100
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.cfg.get("optimizer", "adamw").lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.lr,
                betas=(
                    self.cfg.get("adam_beta1", 0.9),
                    self.cfg.get("adam_beta2", 0.999),
                ),
                weight_decay=self.cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_model(cfg):
    hidden_sizes = [cfg.hidden_size] * cfg.n_layers
    num_blocks = [cfg.block_size] * cfg.n_layers
    return ResNet(
        hidden_sizes,
        num_blocks,
        input_dim=cfg.input_dim,
        in_channels=cfg.in_channels,
        n_classes=cfg.n_classes,
        encodeout="flatten",
    )


def apply_data_variant(cfg):
    data_variant = cfg.get("data_variant", "full") or "full"
    cfg.data_variant = data_variant

    if data_variant == "full":
        base_dir = "./data/bacteria-id/preprocessed"
        input_dim = 696
    elif data_variant == "minimal":
        base_dir = "./data/bacteria-id/preprocessed_minimal"
        input_dim = 696
    elif data_variant == "github_551":
        base_dir = "./data/bacteria-id/preprocessed_github_551"
        input_dim = 551
    else:
        raise ValueError(f"Unknown data_variant: {data_variant}")

    cfg.X_fn = f"{base_dir}/X_reference.npy"
    cfg.y_fn = f"{base_dir}/y_reference.npy"
    cfg.input_dim = input_dim
    return cfg


def get_trainer(cfg):
    result_dir = f"./results/bacteria-id/pretraining/{cfg.augtype}/supervised/"
    os.makedirs(result_dir, exist_ok=True)

    logger_version = os.environ.get("LOGGER_VERSION")
    if logger_version is not None and logger_version.isdigit():
        logger_version = int(logger_version)
    logger = pl.loggers.TensorBoardLogger(result_dir, version=logger_version)  # type: ignore
    logger.log_hyperparams(cfg)

    checkpoint_callback = ModelCheckpoint(
        filename="best",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    earlystop_callback = EarlyStopping(
        monitor="valid_loss",
        patience=cfg.patience,
        verbose=True,
        mode="min",
    )

    return pl.Trainer(
        default_root_dir=result_dir,
        devices="auto",
        precision="16-mixed" if cfg.fp16 is True else "32",
        strategy="auto",
        max_epochs=cfg.n_epochs,
        logger=logger,
        log_every_n_steps=30,
        callbacks=[checkpoint_callback, earlystop_callback],
    )


if __name__ == "__main__":
    args = utils.get_args()
    cfg = OmegaConf.load("./configs/bacteria-id/pretraining/supervised.yaml")
    cfg = OmegaConf.merge(cfg, utils.get_arg_overrides(args))
    cfg.augtype = args.augtype
    cfg = apply_data_variant(cfg)

    utils.seed_all(cfg.seed)
    pl.seed_everything(cfg.seed)

    transformations = None
    if cfg.get("use_augmentation", False):
        transformations = utils.get_trans_from_augtype(cfg.augtype, p=cfg.transition_prob)

    y = np.load(cfg.y_fn)
    idx_tr, idx_val = utils.get_split_idx(
        y,
        0,
        seed=cfg.seed,
        split_mode="random_holdout",
        valid_size=cfg.valid_size,
    )

    train_loader = utils.get_sl_loader(
        cfg.X_fn,
        cfg.y_fn,
        idxs=idx_tr,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        transformation=transformations,
    )
    val_loader = utils.get_sl_loader(
        cfg.X_fn,
        cfg.y_fn,
        idxs=idx_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        transformation=None,
    )

    model = get_model(cfg)
    learner = ReferenceSupervisedLearner(cfg, model)
    trainer = get_trainer(cfg)
    trainer.fit(learner, train_loader, val_loader)
