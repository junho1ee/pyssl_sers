import argparse
import random

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

import augmentations as augs
import datasets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", "-p", type=str, default="simclrv2")
    parser.add_argument("--augtype", "-a", type=str, default="phys")
    parser.add_argument("--fold", "-f", type=int, default=0)
    parser.add_argument("--task", "-t", type=str, default="class30")
    parser.add_argument("--linear_eval", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


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


def get_split_idx(y, fold, seed=0):
    n_samples = len(y)
    idxs = list(range(n_samples))

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    idx_tr, idx_val = list(kfold.split(idxs, y))[fold]
    return idx_tr, idx_val


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(args.get("linear_eval", False))
