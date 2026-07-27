import argparse
import random

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import augmentations as augs
import datasets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", "-p", type=str, default="simclrv2")
    parser.add_argument("--augtype", "-a", type=str, default="phys")
    parser.add_argument("--fold", "-f", type=int, default=0)
    parser.add_argument("--task", "-t", type=str, default="class30")
    parser.add_argument(
        "--data_variant", choices=["full", "minimal", "github_551"], default=None
    )
    parser.add_argument("--pretrained_version", type=str, default=None)
    parser.add_argument("--pretrained_ckpt_name", type=str, default=None)
    parser.add_argument(
        "--use_pretrained", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--use_augmentation", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--reuse_pretrained_classifier",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--split_mode", choices=["stratified_kfold", "random_holdout"])
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--valid_size", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "adam"])
    parser.add_argument("--adam_beta1", type=float, default=None)
    parser.add_argument("--adam_beta2", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--linear_eval", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def get_arg_overrides(args):
    return {key: value for key, value in vars(args).items() if value is not None}


def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ssl_loader(X_fn, idxs=None, batch_size=128, num_workers=4, transformation=None):
    x = np.load(X_fn)
    dataset = datasets.SSLSpectralDataset(x, idxs=idxs, transform=transformation)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return loader


def get_sl_loader(
    X_fn,
    y_fn,
    idxs=None,
    batch_size=128,
    num_workers=4,
    shuffle=True,
    transformation=None,
):
    x = np.load(X_fn)
    y = np.load(y_fn)
    dataset = datasets.SpectralDataset(x, y, idxs=idxs, transform=transformation)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return loader


def get_trans_from_augtype(augtype, p=1.0):
    if augtype == "phys":
        perturbation_mode = [
            "powerline_noise",
            "emg_noise",
            "baseline_shift",
            "baseline_wander",
        ]
        transform = augs.get_transformation(perturbation_mode=perturbation_mode, p=p)
    elif augtype == "crop":
        perturbation_mode = ["random_resized_crop", "freqout"]
        transform = augs.get_transformation(perturbation_mode=perturbation_mode, p=p)
    return transform


def get_transformation(perturbation_mode=None, p=None):
    transform = augs.get_transformation(perturbation_mode=perturbation_mode, p=p)
    return transform


def get_split_idx(
    y,
    fold,
    seed=0,
    split_mode="stratified_kfold",
    n_splits=10,
    valid_size=0.1,
):
    n_samples = len(y)
    idxs = np.arange(n_samples)

    if split_mode == "stratified_kfold":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kfold.split(idxs, y))
        if fold >= len(splits):
            raise ValueError(f"fold={fold} is outside n_splits={n_splits}")
        idx_tr, idx_val = splits[fold]
    elif split_mode == "random_holdout":
        idx_tr, idx_val = train_test_split(
            idxs,
            test_size=valid_size,
            random_state=seed + fold,
            stratify=y,
        )
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    return idx_tr, idx_val


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(args.get("linear_eval", False))
