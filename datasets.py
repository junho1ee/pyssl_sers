import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# from torchvision import transforms


class SpectralDataset(Dataset):
    def __init__(self, X_fn, y_fn, idxs=None, transform=None):
        if type(X_fn) == str:
            self.X = np.load(X_fn)
        else:
            self.X = X_fn
        if type(y_fn) == str:
            self.y = np.load(y_fn)
        else:
            self.y = y_fn
        if idxs is None:
            idxs = np.arange(len(self.y))

        self.idxs = idxs
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x, y = self.X[i], self.y[i]
        x = np.expand_dims(x, axis=0)
        if self.transform:
            x = self.transform(x)
        return (x, y)


class SSLSpectralDataset(Dataset):
    def __init__(self, X_fn, idxs=None, transform=None):
        if type(X_fn) == str:
            self.X = np.load(X_fn)
        else:
            self.X = X_fn

        if idxs is None:
            idxs = np.arange(len(self.X))

        self.idxs = idxs
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x = self.X[i]
        x1 = np.expand_dims(x, axis=0)
        x2 = x1.copy()

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return (x1, x2)
