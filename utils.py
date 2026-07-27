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
    parser.add_argument("--n_labels_per_class", type=int, default=None)
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
    n_labels_per_class=0,
):
    """Build train/validation indices for the fine-tuning subset.

    When ``n_labels_per_class`` is a positive integer the labelled pool is first
    reduced to that many spectra per class (drawn without replacement, stratified,
    with a fold-dependent seed) and the train/validation split is then performed
    on the reduced pool. ``n_labels_per_class=0`` keeps every labelled spectrum
    and reproduces the original behaviour exactly.
    """
    y = np.asarray(y)
    idxs = np.arange(len(y))

    n_keep = int(n_labels_per_class or 0)
    if n_keep > 0:
        rng = np.random.RandomState(seed + 1000 * fold)
        kept = []
        for cls in np.unique(y):
            cls_idx = idxs[y == cls]
            take = min(n_keep, len(cls_idx))
            kept.append(rng.choice(cls_idx, size=take, replace=False))
        idxs = np.sort(np.concatenate(kept))

    y_pool = y[idxs]
    pos = np.arange(len(idxs))

    if split_mode == "stratified_kfold":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kfold.split(pos, y_pool))
        if fold >= len(splits):
            raise ValueError(f"fold={fold} is outside n_splits={n_splits}")
        pos_tr, pos_val = splits[fold]
    elif split_mode == "random_holdout":
        pos_tr, pos_val = train_test_split(
            pos,
            test_size=valid_size,
            random_state=seed + fold,
            stratify=y_pool,
        )
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    return idxs[pos_tr], idxs[pos_val]


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(args.get("linear_eval", False))
