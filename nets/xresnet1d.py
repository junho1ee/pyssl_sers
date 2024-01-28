import torch
import torch.nn as nn
import torch.nn.functional as F


class XResNet1d(nn.Sequential):
    def __init__(self, hidden_sizes, num_blocks, in_channels=64, n_classes=30):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.stem = nn.Sequential(
            nn.Conv1d(
                1, self.in_channels // 2, kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.BatchNorm1d(self.in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(
                self.in_channels // 2,
                self.in_channels // 2,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(self.in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(
                self.in_channels // 2,
                self.in_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(),
        )

        self.maxpool1 = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        layers = []
        strides = [1] + [1] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(
                self._make_layer(hidden_size, num_blocks[idx], stride=strides[idx])
            )

        self.blocks = nn.Sequential(*layers)

        head = [
            AdaptiveConcatPool1d(sz=1),
            nn.Flatten(),
            nn.BatchNorm1d(hidden_sizes[-1] * 2),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_sizes[-1] * 2, self.n_classes),
        ]

        self.head = nn.Sequential(*head)

        super().__init__(
            self.stem,
            self.maxpool1,
            self.blocks,
            self.head,
        )
        init_cnn(self)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        hidden_channels = 64

        # Layers
        self.convpath = nn.Sequential(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )

        idpath = []

        if in_channels != out_channels:
            idpath.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                )
            )
            idpath.append(nn.BatchNorm1d(out_channels))

        self.idpath = nn.Sequential(*idpath)
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act_fn(self.convpath(x) + self.idpath(x))


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)
