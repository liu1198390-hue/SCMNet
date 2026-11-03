"""Convolution, normalization and pyramid-pooling layers."""

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _gn_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, *, padding: int | None = None
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPM(nn.Module):
    def __init__(self, in_channels: int, channels: int, pool_scales: Iterable[int] = (1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(int(scale)),
                    ConvGNAct(in_channels, channels, kernel_size=1, padding=0),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = ConvGNAct(
            in_channels + len(self.branches) * channels, channels, kernel_size=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        priors = [x]
        for branch in self.branches:
            priors.append(
                F.interpolate(branch(x), size=(h, w), mode="bilinear", align_corners=False)
            )
        return self.bottleneck(torch.cat(priors, dim=1))


class DropPath(nn.Module):
    """Per-sample stochastic depth, compatible with the original VSS path."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
        if not 0 <= self.drop_prob < 1:
            raise ValueError("drop_prob must be in [0,1)")

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
