import numpy as np
import torch
from torch import nn


class RamanNet(nn.Module):
    def __init__(self, w_len, n_windows, n_classes):
        super(RamanNet, self).__init__()
        self.w_len = w_len
        self.n_windows = n_windows
        self.n_classes = n_classes

        self.inps = nn.ModuleList()
        self.features = nn.ModuleList()

        for i in range(n_windows):
            inp = nn.Linear(w_len, 25)
            self.inps.append(inp)

            feat = nn.Sequential(nn.BatchNorm1d(25), nn.LeakyReLU())
            self.features.append(feat)

        self.comb = nn.Sequential(nn.Dropout(0.50))

        self.top1 = nn.Sequential(
            nn.Linear(25 * n_windows, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
        )

        self.top2 = nn.Sequential(nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.25))

        self.classification = nn.Sequential(
            nn.Linear(256, n_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        inps = []
        features = []

        for i in range(self.n_windows):
            # inp = self.inps[i](x[:,i*self.w_len:(i+1)*self.w_len])
            inp = self.inps[i](x[:, i, :].squeeze(1))
            inps.append(inp)

            feat = self.features[i](inp)
            features.append(feat)

        comb = self.comb(torch.cat(features, dim=1))

        top = self.top1(comb)

        ## torch l2 normalize
        # torch version 1
        emb = top / torch.norm(top, p=2, dim=1, keepdim=True)
        emb[torch.isnan(emb)] = 0.0

        # torch version 2
        # emb = nn.functional.normalize(top, p=2, dim=1)

        top = self.top2(top)

        classification = self.classification(top)

        return emb, classification


class RamanNet_preprocess(nn.Module):
    def __init__(self, w_len, n_windows):
        super().__init__()
        self.w_len = w_len
        self.n_windows = n_windows

        self.inps = nn.ModuleList()
        self.features = nn.ModuleList()

        for i in range(n_windows):
            inp = nn.Linear(w_len, 25)
            self.inps.append(inp)

            feat = nn.Sequential(nn.BatchNorm1d(25), nn.LeakyReLU())
            self.features.append(feat)

        self.comb = nn.Sequential(nn.Dropout(0.50))

        self.top1 = nn.Sequential(
            nn.Linear(25 * n_windows, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        inps = []
        features = []

        for i in range(self.n_windows):
            # inp = self.inps[i](x[:,i*self.w_len:(i+1)*self.w_len])
            inp = self.inps[i](x[:, i, :].squeeze(1))
            inps.append(inp)

            feat = self.features[i](inp)
            features.append(feat)

        comb = self.comb(torch.cat(features, dim=1))

        top = self.top1(comb)

        ## torch l2 normalize
        # torch version 1
        emb = top / torch.norm(top, p=2, dim=1, keepdim=True)
        emb[torch.isnan(emb)] = 0.0

        # torch version 2
        # emb = nn.functional.normalize(top, p=2, dim=1)

        return emb, top


class RamanNet_postprocess(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.top2 = nn.Sequential(nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.25))

        self.classification = nn.Sequential(
            nn.Linear(256, n_classes),
            # nn.Softmax(dim=-1)
            # nn.Softmax()
        )

    def forward(self, top):
        top = self.top2(top)

        classification = self.classification(top)
        return classification


class RamanNet_split(nn.Module):
    def __init__(self, w_len, n_windows, n_classes):
        super().__init__()
        self.w_len = w_len
        self.n_windows = n_windows
        self.n_classes = n_classes

        self.preprocess = RamanNet_preprocess(w_len, n_windows)
        self.postprocess = RamanNet_postprocess(n_classes)

    def forward(self, x):
        emb, top = self.preprocess(x)
        classification = self.postprocess(top)

        return emb, classification
